"""The one way out of a submission's install: an HTTPS tunnel to the package index and the code host, and nothing else.

Installing a submission runs its build code, which has no network of its own (:func:`hosted.install_submission`): it
reaches the world only through this proxy, as ``HTTPS_PROXY``, which answers ``CONNECT`` to :data:`HOSTS` on port 443 and
refuses everything else. It reads nothing but the request line and headers of each connection, and the tunnels it opens
carry the client's own TLS. The verifier starts one for an install and stops it when the install ends, before any
instance exists, so that nothing a build left behind can reach it afterwards.
"""

from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Callable

HOSTS = frozenset({
    "pypi.org", "files.pythonhosted.org",
    "github.com", "api.github.com", "codeload.github.com", "raw.githubusercontent.com",
    "objects.githubusercontent.com", "release-assets.githubusercontent.com",
})
MAX_HEADER = 8192
MAX_TUNNELS = 64
IDLE_S = 120.0

log = logging.getLogger(__name__)


def _target(request: bytes) -> tuple[str, int] | None:
    """Where a ``CONNECT host:port HTTP/1.x`` request asks to go, or None for any other request."""
    parts = request.split(b"\r\n", 1)[0].decode("ascii", "replace").split()
    if len(parts) != 3 or parts[0] != "CONNECT" or not parts[2].startswith("HTTP/1."):
        return None
    host, _, port = parts[1].rpartition(":")
    return (host.lower(), int(port)) if host and port.isascii() and port.isdigit() else None


class ConnectFilter:
    """Serve on ``127.0.0.1:<port>``; ``port`` 0 takes a free one. ``note`` hears each decision, for the install's log,
    and ``tunnel_port`` is the one port a tunnel may reach."""

    def __init__(self, hosts: frozenset[str] = HOSTS, port: int = 0, note: Callable[[str], None] = log.info,
                 tunnel_port: int = 443):
        self.hosts, self.note, self.tunnel_port = hosts, note, tunnel_port
        self._listener = socket.create_server(("127.0.0.1", port))
        self._slots = threading.BoundedSemaphore(MAX_TUNNELS)
        self._stopped = threading.Event()

    @property
    def port(self) -> int:
        return self._listener.getsockname()[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> ConnectFilter:
        threading.Thread(target=self._serve, daemon=True, name="connect-filter").start()
        return self

    def stop(self) -> None:
        self._stopped.set()
        self._listener.close()

    def __enter__(self) -> ConnectFilter:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _serve(self) -> None:
        while not self._stopped.is_set():
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            if not self._slots.acquire(blocking=False):
                client.close()
                continue
            threading.Thread(target=self._tunnel, args=(client,), daemon=True).start()

    def _tunnel(self, client: socket.socket) -> None:
        upstream = None
        try:
            client.settimeout(IDLE_S)
            request = b""
            while b"\r\n\r\n" not in request and len(request) < MAX_HEADER:
                chunk = client.recv(MAX_HEADER)
                if not chunk:
                    return
                request += chunk
            target = _target(request)
            if target is None or target[0] not in self.hosts or target[1] != self.tunnel_port:
                self.note(f"refused {request[:120]!r}")
                client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            upstream = socket.create_connection(target, timeout=30)
            upstream.settimeout(IDLE_S)
            self.note(f"tunnel to {target[0]}")
            client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            other = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
            other.start()
            _pump(client, upstream)
            other.join()
        except OSError as error:
            self.note(f"tunnel failed: {type(error).__name__}")
        finally:
            for sock in (client, upstream):
                if sock is not None:
                    sock.close()
            self._slots.release()


def _pump(source: socket.socket, sink: socket.socket) -> None:
    try:
        while data := source.recv(65536):
            sink.sendall(data)
    except OSError:
        pass
    finally:
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass
