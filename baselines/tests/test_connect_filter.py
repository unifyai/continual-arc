"""The install's one way out (systems/connect_filter.py): a tunnel to the hosts it names, on one port, and nothing else."""

import socket

import pytest

from continual_arc_baselines.systems.connect_filter import HOSTS, ConnectFilter


def ask(proxy: ConnectFilter, request: bytes) -> tuple[bytes, socket.socket]:
    client = socket.create_connection(("127.0.0.1", proxy.port), timeout=5)
    client.sendall(request)
    return client.recv(200), client


@pytest.fixture
def server():
    listener = socket.create_server(("127.0.0.1", 0))
    yield listener
    listener.close()


def test_a_named_host_gets_a_tunnel_that_carries_both_ways(server):
    port = server.getsockname()[1]
    with ConnectFilter(hosts=frozenset({"localhost"}), tunnel_port=port) as proxy:
        answer, client = ask(proxy, f"CONNECT localhost:{port} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
        assert answer.startswith(b"HTTP/1.1 200")
        upstream, _ = server.accept()
        client.sendall(b"hello")
        assert upstream.recv(5) == b"hello"
        upstream.sendall(b"there")
        assert client.recv(5) == b"there"
        client.close()
        upstream.close()


@pytest.mark.parametrize("request_line", [
    b"CONNECT example.com:443 HTTP/1.1", b"CONNECT 1.1.1.1:443 HTTP/1.1", b"CONNECT github.com:22 HTTP/1.1",
    b"CONNECT github.com:80 HTTP/1.1", b"CONNECT evil.github.com.example.org:443 HTTP/1.1", b"CONNECT github.com HTTP/1.1",
    b"GET http://github.com/ HTTP/1.1", b"GET / HTTP/1.1", b"not http at all", b"CONNECT github.com:443", b"CONNECT :443 HTTP/1.1",
])
def test_anything_else_is_refused(request_line):
    notes = []
    with ConnectFilter(note=notes.append) as proxy:
        answer, client = ask(proxy, request_line + b"\r\n\r\n")
        client.close()
    assert answer.startswith(b"HTTP/1.1 403") and any(note.startswith("refused") for note in notes)


def test_it_names_the_index_and_the_code_host_alone():
    assert {"pypi.org", "files.pythonhosted.org", "github.com"} <= HOSTS
    assert all(h.endswith(("pypi.org", "pythonhosted.org", "github.com", "githubusercontent.com")) for h in HOSTS)


def test_a_header_without_end_is_cut_off():
    with ConnectFilter() as proxy:
        client = socket.create_connection(("127.0.0.1", proxy.port), timeout=5)
        try:
            for _ in range(64):
                client.sendall(b"x" * 8192)
        except OSError:
            pass  # closed on us, which is the point
        client.settimeout(5)
        assert client.recv(200).startswith(b"HTTP/1.1 403")
        client.close()
