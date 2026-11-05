"""The Linux sandbox (systems/sandbox.py, bubblewrap) as an allowlist: an empty root with the operating system's programs
and libraries and what the system is declared to need bound in, so that a harness cannot read other runs' transcripts
wherever they lie or reach a daemon's socket, a filter on what it may ask of the kernel, and limits on what it consumes."""

import json
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from continual_arc_baselines.systems import seccomp
from continual_arc_baselines.systems.sandbox import (
    SandboxSpec,
    bwrap_command,
    resolve_backend,
    start_bridge,
    wrap_command,
)

BUBBLEWRAP = pytest.mark.skipif(platform.system() != "Linux" or shutil.which("bwrap") is None,
                                reason="bubblewrap on Linux only")


def _pairs(cmd, flag):
    return [(cmd[i + 1], cmd[i + 2]) for i, arg in enumerate(cmd[:-2]) if arg == flag]


def test_the_command_starts_from_nothing_and_binds_back_only_the_declared_paths(tmp_path):
    home, tmp = tmp_path / "home", tmp_path / "tmp"
    runtime = home / "u" / ".local" / "share" / "python-3.11.16"
    (runtime / "bin").mkdir(parents=True)
    minor = home / "u" / ".local" / "share" / "python-3.11"
    minor.symlink_to(runtime)  # the minor-version link a venv's interpreter goes through
    install = home / "u" / ".local" / "share" / "agent"
    (install / "bin").mkdir(parents=True)
    (install / "bin" / "agent").write_text("#!/bin/sh\n")
    launcher = home / "u" / ".local" / "bin" / "agent"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(install / "bin" / "agent")
    state = home / "u" / "runs" / "r1" / "users" / "u000" / "system" / "state"
    state.mkdir(parents=True)
    bridge = state.parent / "netbridge.py"
    bridge.write_text("")
    bench = home / "u" / "bench"
    (bench / "src").mkdir(parents=True)
    private_tmp = tmp / "arc-pa-1"
    private_tmp.mkdir(parents=True)
    spec = SandboxSpec(read_paths=[install, minor, bench], write_paths=[state], tmp_dir=private_tmp,
                       mask_paths=[bench / "src"])
    argv = [str(launcher), "--cwd", str(state), str(bridge)]
    cmd = bwrap_command(argv, spec)
    assert cmd[0] == "prlimit" and cmd[cmd.index("bwrap") + 1:cmd.index("bwrap") + 4] == [
        "--unshare-user", "--unshare-ipc", "--unshare-pid"]
    assert "--unshare-net" in cmd and "--unshare-uts" in cmd and "--die-with-parent" in cmd
    ro, rw, links = _pairs(cmd, "--ro-bind"), _pairs(cmd, "--bind"), _pairs(cmd, "--symlink")
    assert ("/", "/") not in ro and ("/usr", "/usr") in ro and ("/etc", "/etc") in ro
    assert not any(dest in ("/run", "/var", "/home", "/root", "/sys", "/boot") for _, dest in ro)
    assert (str(install), str(install)) in ro and (str(runtime), str(runtime)) in ro and (str(bench), str(bench)) in ro
    assert (str(bridge), str(bridge)) in ro  # named on the command line
    # links stay links, so a program that finds itself through one sees what it sees on the host
    assert (str(runtime), str(minor)) in links and (str(install / "bin" / "agent"), str(launcher)) in links
    assert rw == [(str(state), str(state)), (str(private_tmp), str(private_tmp))]
    # writable paths come after every read-only bind and link, so a writable path wins where they meet
    assert cmd.index("--bind") > max(i for i, a in enumerate(cmd) if a in ("--ro-bind", "--symlink"))
    # a benchmark tree under something bound back is hidden again, last, and the root is read-only after everything
    tmpfs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--tmpfs"]
    assert tmpfs[-1] == str(bench / "src") and cmd.index(str(bench / "src")) > cmd.index("--bind")
    assert cmd.index("--remount-ro") > cmd.index(str(bench / "src"))
    # nothing else of the home directory comes back: no other run, no parent directory
    bound = {dest for _, dest in ro + rw + links}
    assert not any(str(b) in bound for b in (home / "u", home / "u" / "runs", state.parent))
    assert cmd[len(cmd) - len(argv) - 1:] == ["--", *argv]


def test_a_link_in_what_the_system_wrote_decides_nothing(tmp_path):
    """A submission's build writes its own environment: a link there, on the command line or under it, must not bind what
    it names, or the build chooses what the sandbox shows."""
    secret = tmp_path / "rules"
    secret.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(secret)
    state = tmp_path / "state"
    state.mkdir()
    (state / "link").symlink_to(secret)
    spec = SandboxSpec(opaque_paths=[venv], write_paths=[state])
    cmd = bwrap_command([str(venv / "bin" / "python"), str(state / "link")], spec)
    binds = _pairs(cmd, "--ro-bind") + _pairs(cmd, "--bind") + _pairs(cmd, "--symlink")
    assert (str(venv), str(venv)) in binds and not any(str(secret) in pair for pair in binds)


@BUBBLEWRAP
def test_inside_the_sandbox_only_the_declared_paths_exist(tmp_path):
    secret = tmp_path / "other-run" / "turns.jsonl"  # under /tmp: another system's files
    secret.parent.mkdir()
    secret.write_text("the answer")
    state = tmp_path / "state"
    state.mkdir()
    readable = tmp_path / "install"
    readable.mkdir()
    (readable / "tool.txt").write_text("tool\n")
    spec = SandboxSpec(read_paths=[readable], write_paths=[state])
    script = (f"test ! -e {secret} && echo hidden; cat {readable}/tool.txt; echo ok > {state}/written && echo wrote; "
              f"ls -A {Path.home()} | wc -l; ls -A /tmp | wc -l")
    out = subprocess.run(bwrap_command(["/bin/sh", "-c", script], spec), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.split()
    assert lines[:3] == ["hidden", "tool", "wrote"]
    assert (state / "written").read_text() == "ok\n"
    # the home directory holds only what was bound back into it (nothing, unless tmp_path lies under it)
    assert int(lines[3]) <= 1 and int(lines[4]) <= 1


def test_a_bridge_returns_once_its_own_socket_exists_not_one_an_earlier_bridge_left(tmp_path):
    """An adapter may start a bridge for every session in the same private directory, and an ended bridge
    leaves its socket file. Were start_bridge to find that file at once, bubblewrap would race the new bridge, which
    removes it before binding its own ("bwrap: Can't bind mount .../proxy.sock ... No such file or directory"). When
    start_bridge returns, the file must be the new bridge's socket, and it must reach the proxy."""
    proxy = socket.create_server(("127.0.0.1", 0))  # stands in for the tracking proxy
    short = Path(tempfile.mkdtemp(prefix="arc-t-", dir="/tmp"))  # UNIX socket paths must be short
    sock_path = short / "proxy.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(sock_path))  # what an ended bridge leaves: a socket file nothing listens on; held open, so the new
    stale_inode = os.stat(sock_path).st_ino  # socket cannot be given the same inode number
    host = None
    try:
        _, host = start_bridge(["true"], proxy.getsockname()[1], sock_path, tmp_path / "netbridge.py")
        assert os.stat(sock_path).st_ino != stale_inode
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + 5
        while True:  # the file exists once the bridge has bound it; it listens a moment later
            try:
                client.connect(str(sock_path))
                break
            except ConnectionRefusedError:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        proxy.settimeout(5)
        upstream, _ = proxy.accept()
        client.sendall(b"ping")
        upstream.settimeout(5)
        assert upstream.recv(4) == b"ping"
        client.close()
        upstream.close()
    finally:
        if host is not None:
            host.kill()
            host.wait(timeout=10)
        stale.close()
        proxy.close()
        shutil.rmtree(short, ignore_errors=True)


PROBE = r"""
import ctypes, json, os, socket, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


def call(number, *args):
    ctypes.set_errno(0)
    result = libc.syscall(ctypes.c_long(number), *[ctypes.c_long(a) for a in args])
    return 0 if result >= 0 else ctypes.get_errno()


def family(domain, kind=socket.SOCK_STREAM, protocol=0):
    try:
        socket.socket(domain, kind, protocol).close()
        return 0
    except OSError as error:
        return error.errno


pid = os.fork()
if pid == 0:
    libc.shmget(0x52454454, 4096, 0o1666)
    os._exit(0)
os.waitpid(pid, 0)
json.dump({
    "unshare": call(272, 0x10000000), "io_uring_setup": call(425, 1, 0), "bpf": call(321, 0, 0, 0),
    "mount": call(165, 0, 0, 0, 0, 0), "ptrace": call(101, 0, 0, 0, 0), "keyctl": call(250, 0, 0, 0, 0, 0),
    "clone3": call(435, 0, 0), "x32": call(0x40000000 + 39), "new": call(600),
    "clone_newuser": call(56, 0x10000000 | 17, 0, 0, 0, 0),
    "personality_aslr": call(135, 0x0040000), "personality_read": call(135, 0xFFFFFFFF),
    "alg": family(38, socket.SOCK_SEQPACKET), "vsock": family(40, socket.SOCK_DGRAM),
    "netlink_uevent": family(16, socket.SOCK_RAW, 15), "netlink_route": family(16, socket.SOCK_RAW, 0),
    "unix": family(socket.AF_UNIX), "inet": family(socket.AF_INET),
    "getpid": call(39), "shared_segments": len(open("/proc/sysvipc/shm").read().splitlines()) - 1,
    "paths": {p: os.path.lexists(p) for p in ("/run/user", "/run/dbus", "/var/log", "/var/lib", "/home", "/root", "/boot",
                                              "/sys/fs/cgroup", "/sys/kernel", "/mnt", "/srv", "/snap")},
    "hostname": os.uname().nodename, "pids": len([p for p in os.listdir("/proc") if p.isdigit()]),
    "writable_root": os.access("/", os.W_OK),
}, sys.stdout)
"""


@BUBBLEWRAP
def test_what_a_process_in_the_sandbox_can_ask_of_the_kernel_and_see_of_the_host(tmp_path):
    """The filter (systems/seccomp.py) and the namespaces, from inside: what no learner needs is refused, what every
    runtime needs is not, and the host's sockets, processes, shared memory and name are not there."""
    if platform.machine() != "x86_64":
        pytest.skip("the filter has x86-64's numbers")
    script = tmp_path / "probe.py"
    script.write_text(PROBE)
    argv = wrap_command("bwrap", ["/usr/bin/python3", "-I", str(script)], SandboxSpec(), tmp_path / "filter")
    out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    seen = json.loads(out.stdout)
    assert not any(seen["paths"].values()), seen["paths"]
    assert seen["hostname"] == "sandbox" and seen["pids"] < 5 and not seen["writable_root"]
    assert seen["shared_segments"] == 1  # its own segment and no one else's: the IPC namespace is its own
    eperm, enosys, eafnosupport = 1, 38, 97
    for name in ("unshare", "io_uring_setup", "bpf", "mount", "ptrace", "keyctl", "clone_newuser", "personality_aslr"):
        assert seen[name] == eperm, name
    for name in ("clone3", "x32", "new"):
        assert seen[name] == enosys, name
    for name in ("alg", "vsock", "netlink_uevent"):
        assert seen[name] == eafnosupport, name
    assert seen["netlink_route"] == seen["unix"] == seen["inet"] == seen["getpid"] == seen["personality_read"] == 0


@BUBBLEWRAP
def test_a_32_bit_call_past_the_filter_kills_the_process(tmp_path):
    """The 32-bit entry numbers calls differently, so a filter that read only x86-64's numbers would miss them."""
    if platform.machine() != "x86_64":
        pytest.skip("the filter has x86-64's numbers")
    script = tmp_path / "probe.py"
    script.write_text("""
import ctypes, mmap
code = bytes([0x53, 0xb8, 0x14, 0, 0, 0, 0xcd, 0x80, 0x5b, 0xc3])  # push rbx; mov eax, 20 (i386 getpid); int 0x80; pop rbx; ret
page = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
page.write(code)
ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.addressof(ctypes.c_char.from_buffer(page)))()
print("not stopped")
""")
    argv = wrap_command("bwrap", ["/usr/bin/python3", "-I", str(script)], SandboxSpec(), tmp_path / "filter")
    out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert "not stopped" not in out.stdout and out.returncode != 0


@BUBBLEWRAP
def test_the_limits_hold_inside(tmp_path):
    script = ("import resource as r; print(*(r.getrlimit(w)[0] for w in (r.RLIMIT_CORE, r.RLIMIT_NOFILE, r.RLIMIT_NPROC, "
              "r.RLIMIT_FSIZE)))")
    argv = wrap_command("bwrap", ["/usr/bin/python3", "-I", "-c", script], SandboxSpec(), tmp_path / "filter")
    out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    core, files, processes, size = (int(value) for value in out.stdout.split())
    assert core == 0 and 0 < files <= 65536 and 0 < processes <= 8192 and 0 < size <= 64 * 2**30


def test_a_filter_is_a_valid_program():
    program = seccomp.program()
    if program is None:
        pytest.skip("the filter has x86-64's numbers")
    assert len(program) % 8 == 0 and len(program) / 8 < 4096
    code, _, _, last = struct.unpack("<HBBI", program[-8:])
    assert code == seccomp.RET and last == seccomp.RET_KILL_PROCESS


def test_inside_a_sandbox_auto_means_the_one_around_it(monkeypatch):
    """A harness adapter run whole in a submission's sandbox cannot make a sandbox of its own, the namespaces being
    refused, and needs none."""
    monkeypatch.setenv("ARC_CONFINED", "1")
    assert resolve_backend("auto") == "none"
    assert sys.platform and resolve_backend("none") == "none"
