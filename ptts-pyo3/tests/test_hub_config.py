"""Check config requirements through the Hub transport without model downloads."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import ptts


@pytest.fixture
def hub(tmp_path, monkeypatch):
    state = SimpleNamespace(repo="test/phonon", revision="b" * 40, requests=[], denied=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, body):
            path = urlsplit(self.path).path
            state.requests.append(path)
            if "/tree/" in path:
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                if body:
                    self.wfile.write(b"[]")
                return
            self.send_response(401 if state.denied else 404)
            if not state.denied:
                self.send_header("X-Error-Code", "EntryNotFound")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def load(hub):
    return ptts.TTS(config=hub.repo, revision=hub.revision, device="cpu", quant="q8", lang="en")


def test_phonon_hub_requires_config_before_fetching_weights(hub):
    with pytest.raises(LookupError, match="config.json"):
        load(hub)
    assert not any(p.endswith((".gguf", ".safetensors")) for p in hub.requests)


@pytest.mark.parametrize("repo", ["kyutai/pocket-tts", "kyutai/pocket-tts-without-voice-cloning"])
def test_known_pocket_repos_keep_the_config_fallback(hub, repo):
    hub.repo = repo
    with pytest.raises(LookupError, match="no weights"):
        load(hub)


def test_config_authentication_errors_remain_io_errors(hub):
    hub.denied = True
    with pytest.raises(OSError):
        load(hub)
    assert not any(p.endswith((".gguf", ".safetensors")) for p in hub.requests)
