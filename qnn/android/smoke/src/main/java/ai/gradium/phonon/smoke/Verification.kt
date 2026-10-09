package ai.gradium.phonon.smoke

import android.content.Context
import android.util.Log
import ai.gradium.phonon.PhononTTS
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Runs without audio playback so a remote device can save output and measure inference. */
internal object Verification {
    fun run(context: Context, bundle: File, lang: String) {
        PhononTTS.load(context, bundle, lang).use { model ->
            val rate = model.sampleRate
            check(model.voices.isNotEmpty()) { "No voices in bundle" }
            val output = File(context.getExternalFilesDir(null), "phonon-verification.wav")
            repeat(5) { iteration ->
                var samples = 0L
                val file = if (iteration == 4) RandomAccessFile(output, "rw").apply {
                    setLength(0); write(ByteArray(44))
                } else null
                try {
                    val result = model.speak("Hello from Phonon on the Snapdragon NPU.") { pcm ->
                        check(pcm.isNotEmpty() && pcm.all { it.isFinite() }) { "Invalid PCM" }
                        samples += pcm.size
                        file?.let { wav ->
                            val bytes = ByteBuffer.allocate(pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                            for (value in pcm) bytes.putShort((value.coerceIn(-1f, 1f) * 32767).toInt().toShort())
                            wav.write(bytes.array())
                        }
                        true
                    }
                    check(!result.cancelled && samples > 0 && result.samples == samples) { "Incomplete generation: $result" }
                    check(result.ttfaMs >= 0 && result.totalMs >= result.ttfaMs) { "Invalid timing: $result" }
                    val rtf = result.totalMs / (samples * 1000.0 / rate)
                    Log.i("PhononSmoke", "QNN_VERIFY_RUN $iteration $result rtf=$rtf")
                    file?.let { wav ->
                        val size = (samples * 2).toInt()
                        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
                        header.put("RIFF".toByteArray(Charsets.US_ASCII)).putInt(36 + size)
                        header.put("WAVEfmt ".toByteArray(Charsets.US_ASCII)).putInt(16)
                        header.putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
                        header.put("data".toByteArray(Charsets.US_ASCII)).putInt(size)
                        wav.seek(0); wav.write(header.array())
                    }
                } finally { file?.close() }
            }
            Log.i("PhononSmoke", "QNN_VERIFY_OK wav=${output.absolutePath}")
        }
    }
}
