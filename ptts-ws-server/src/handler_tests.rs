use super::*;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll};

struct StalledSocket(Arc<tokio::sync::Notify>);

impl futures_util::Sink<Message> for StalledSocket {
    type Error = std::io::Error;

    fn poll_ready(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.0.notify_one();
        Poll::Pending
    }

    fn start_send(self: Pin<&mut Self>, _: Message) -> Result<(), Self::Error> {
        unreachable!("the socket never becomes ready")
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Pending
    }

    fn poll_close(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
}

struct EndlessAudio {
    produced: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
    full: Arc<tokio::sync::Notify>,
}

impl Iterator for EndlessAudio {
    type Item = ptts::Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.produced.fetch_add(1, Ordering::SeqCst) + 1;
        if n == REPLY_QUEUE + AUDIO_QUEUE + 3 {
            self.full.notify_one();
        }
        Some(Ok(vec![0.1; 480]))
    }
}

impl Drop for EndlessAudio {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn a_slow_socket_bounds_audio_and_disconnect_joins_the_producer() {
    let (reply_tx, reply_rx) = mpsc::channel(REPLY_QUEUE);
    let (audio_tx, mut audio_rx) = mpsc::channel(AUDIO_QUEUE);
    let (cancel_tx, mut cancel_rx) = watch::channel(false);
    let stalled = Arc::new(tokio::sync::Notify::new());
    let forwarder = tokio::spawn(forward_replies(
        StalledSocket(stalled.clone()),
        reply_rx,
        cancel_rx.clone(),
        SEND_TIMEOUT,
    ));
    let produced = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    let full = Arc::new(tokio::sync::Notify::new());
    let stream =
        EndlessAudio { produced: produced.clone(), dropped: dropped.clone(), full: full.clone() };
    let generation = tokio::spawn(async move {
        let producer = tokio::task::spawn_blocking(move || drain_audio(stream, audio_tx));
        let mut encoder = Encoder::new(Format::default(), 480, 24000).unwrap();
        let outcome =
            forward_audio(&mut audio_rx, 480, 0, &mut encoder, &reply_tx, &mut cancel_rx).await;
        drop(audio_rx);
        producer.await.unwrap().unwrap();
        outcome.unwrap();
    });
    tokio::time::timeout(Duration::from_secs(2), stalled.notified()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), full.notified()).await.unwrap();
    // One chunk is in each forwarding stage and one is blocked in the producer.
    assert_eq!(produced.load(Ordering::SeqCst), REPLY_QUEUE + AUDIO_QUEUE + 3);
    cancel_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(2), generation).await.unwrap().unwrap();
    forwarder.await.unwrap().unwrap();
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn a_stalled_socket_times_out_and_closes_the_reply_channel() {
    let (tx, rx) = mpsc::channel(REPLY_QUEUE);
    let (_cancel_tx, cancel_rx) = watch::channel(false);
    tx.send(TtsReply::EndOfStream).await.unwrap();
    let error = forward_replies(
        StalledSocket(Arc::new(tokio::sync::Notify::new())),
        rx,
        cancel_rx,
        Duration::from_millis(10),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("stopped reading"));
    assert!(tx.is_closed());
}

#[tokio::test]
async fn normal_end_of_stream_drains_queued_replies_in_order() {
    let output = Arc::new(std::sync::Mutex::new(Vec::new()));
    let socket = futures_util::sink::unfold(output.clone(), |output, msg| async move {
        let Message::Text(text) = msg else { panic!("expected text reply") };
        output.lock().unwrap().push(text.to_string());
        Ok::<_, std::convert::Infallible>(output)
    });
    let (tx, rx) = mpsc::channel(REPLY_QUEUE);
    let (_cancel_tx, cancel_rx) = watch::channel(false);
    tx.send(TtsReply::Flushed { flush_id: 7 }).await.unwrap();
    tx.send(TtsReply::EndOfStream).await.unwrap();
    drop(tx);
    forward_replies(Box::pin(socket), rx, cancel_rx, SEND_TIMEOUT).await.unwrap();
    let output = output.lock().unwrap();
    assert_eq!(output.len(), 2);
    assert!(matches!(serde_json::from_str(&output[0]).unwrap(), TtsReply::Flushed { flush_id: 7 }));
    assert!(matches!(serde_json::from_str(&output[1]).unwrap(), TtsReply::EndOfStream));
}

#[test]
fn pending_text_is_limited_in_characters_and_a_rejected_append_preserves_it() {
    let mut buffer = "é".repeat(MAX_INPUT_CHARS - 1);
    assert!(append_text(&mut buffer, "🙂"));
    let before = buffer.clone();
    assert!(!append_text(&mut buffer, "x"));
    assert_eq!(buffer, before);
    buffer.clear();
    assert!(append_text(&mut buffer, "Hello."));
}
