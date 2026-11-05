"""Confine a system's process tree so the benchmark cannot leak into it.

Systems execute model-written code. Left alone, that code can read the benchmark
source, import the Re-ARC or ARC-GEN generators from the very environment that
launched it, or reach the internet: a harness unconfined can locate ``continual_arc``
in the baselines' environment and call the task's verifier to compute the ground
truth before submitting.

The rule is therefore: a system sees only what it needs, and the tracking proxy is
its only way out.

* read: the system's own installation, its Python runtime, the OS's programs and libraries;
* read and write: its private state under the attempt directory, a private HOME
  and a private temporary directory;
* network: TCP to the proxy on the loopback interface, and UNIX sockets under its
  private temporary directory (for anything the system spawns for itself);
* nothing else. Under bubblewrap nothing else of the host's tree exists inside the sandbox, so neither the
  repositories, the virtual environments, nor the sockets of the daemons outside it (a process that reaches
  the systemd manager's socket under ``/run/user`` starts what it likes outside every namespace) are reachable.

Backends: ``bwrap`` (bubblewrap on Linux, :func:`bwrap_command`), which also gives the system private user, network,
pid, IPC, UTS and cgroup namespaces, a filter on what it may ask of the kernel (:mod:`continual_arc_baselines.systems.seccomp`)
and limits on what it may consume (:data:`LIMITS`), and ``seatbelt`` (macOS ``sandbox-exec``,
:func:`seatbelt_profile`); each wraps the command in place and applies to every descendant process.
:func:`resolve_backend` never falls back to no sandbox silently: ``sandbox="none"`` has to be spelled out and is recorded
in the run parameters, unless the process is itself inside a sandbox (:data:`CONFINED`).
"""

from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from continual_arc_baselines.systems import seccomp

SEATBELT = "/usr/bin/sandbox-exec"
BACKENDS = ("auto", "seatbelt", "bwrap", "none")
CONFINED = "ARC_CONFINED"
"""Set in the environment of a process already inside a sandbox: a harness adapter run whole in a submission's sandbox
starts its harness with no sandbox of its own, since the filter refuses the namespaces another would need and the one
around it already holds."""

# A confined harness's environment is an allowlist: of the host's, these, its locale (``LC_*``) and the variables an
# adapter is told to pass through; with a proxy, the proxy's address and placeholder key (PROXY_ENV); and its own key
# variable (KEY_VARIABLE), which its provider entry reads, holding the placeholder.
INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ")
PROXY_ENV = ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY", "OPENROUTER_BASE_URL",
             "OPENROUTER_API_KEY", "ARC_LLM_BASE_URL")
KEY_VARIABLE = "ARC_LLM_API_KEY"

# Operating-system paths any process needs; nothing under /Users or /home.
_SYSTEM_READ = (
    "/usr", "/bin", "/sbin", "/System", "/Library", "/private/etc", "/etc",
    "/private/var/db", "/private/var/select", "/var/db", "/var/select", "/dev",
    "/Applications/Xcode.app/Contents/Developer/usr",  # harmless if absent
)


class SandboxUnavailable(RuntimeError):
    """No sandbox backend for this platform, and none was explicitly waived."""


@dataclass
class SandboxSpec:
    """What a confined process tree may touch. All paths are made absolute and
    symlink-free, because the sandbox checks the resolved path."""

    read_paths: list[Path] = field(default_factory=list)
    write_paths: list[Path] = field(default_factory=list)
    tmp_dir: Path | None = None
    home: Path | None = None
    mask_paths: list[Path] = field(default_factory=list)
    """Paths hidden from the system (bubblewrap only; seatbelt denies by default).
    The benchmark source, its data with the generators and verifiers, and the
    virtual environments that hold them."""
    loopback_ports: list[int] = field(default_factory=list)
    """Loopback TCP ports the system may connect to (the proxy)."""
    listen_ports: list[int] = field(default_factory=list)
    """Loopback TCP ports the system may listen on and connect to (a gateway it
    runs for itself, such as openclaw's)."""
    unix_socket_dirs: list[Path] = field(default_factory=list)
    opaque_paths: list[Path] = field(default_factory=list)
    """Trees written by the code being confined, a submission's environment and checkout: bound whole and read-only, and
    no link inside them is followed to decide what else to bind, since a link there may name anything on the host."""

    def normalised(self) -> SandboxSpec:
        def fix(paths):
            out = []
            for p in paths:
                p = Path(p).expanduser()
                if p.exists():
                    out.append(p.resolve())
            return out

        return SandboxSpec(
            read_paths=fix(self.read_paths), write_paths=fix(self.write_paths),
            mask_paths=fix(self.mask_paths), opaque_paths=fix(self.opaque_paths),
            tmp_dir=self.tmp_dir.resolve() if self.tmp_dir else None,
            home=self.home.resolve() if self.home else None,
            loopback_ports=list(self.loopback_ports),
            listen_ports=list(self.listen_ports),
            unix_socket_dirs=fix(self.unix_socket_dirs),
        )


def resolve_backend(requested: str = "auto") -> str:
    """Pick the sandbox backend: seatbelt on macOS, bubblewrap on Linux, never a
    silent fallback to none."""
    if requested not in BACKENDS:
        raise ValueError(f"sandbox must be one of {BACKENDS}, got {requested!r}")
    if requested == "none" or (requested == "auto" and os.environ.get(CONFINED)):
        return "none"
    system = platform.system()
    if requested == "seatbelt" or (requested == "auto" and system == "Darwin"):
        if system == "Darwin" and os.path.exists(SEATBELT):
            return "seatbelt"
        raise SandboxUnavailable("seatbelt (sandbox-exec) is only available on macOS")
    if requested == "bwrap" or (requested == "auto" and system == "Linux"):
        if shutil.which("bwrap"):
            return "bwrap"
        raise SandboxUnavailable("bubblewrap (bwrap) is not installed; apt install bubblewrap, "
                                 "or pass sandbox=\"none\" explicitly (recorded in the run)")
    raise SandboxUnavailable(f"no sandbox backend for {system}; pass sandbox=\"none\" explicitly")


def default_mask_paths() -> list[Path]:
    """What a system must not be able to read on a Linux node, even under a folder bound back for it: this repository
    (the benchmark, the baselines, their virtual environments and every run's attempt under them), the pinned data
    (generators and verifiers), and a separate checkout of the baselines, its source, environments and runs. A harness
    that walks the whole tree searching for its task id would otherwise find other runs' state, transcripts, records
    and schedules; a run's own state is bound back in by :func:`bwrap_command`."""
    home = Path.home()
    candidates = [
        home / "continual-arc", home / ".cache" / "continual-arc", home / "prime-rl",
        home / "continual-arc-baselines" / "src", home / "continual-arc-baselines" / "envs",
        home / "continual-arc-baselines" / ".venv", home / "continual-arc-baselines" / ".workbench",
    ]
    return [p for p in candidates if p.exists()]


NETBRIDGE = r'''"""Forward between a loopback TCP port and a UNIX socket (stdlib only).

Two roles, one script: on the host, ``unix2tcp SOCK PORT`` accepts connections on the
UNIX socket and relays them to the tracking proxy on 127.0.0.1:PORT; inside a sandbox
with a private network namespace, ``tcp2unix PORT SOCK`` listens on 127.0.0.1:PORT and
relays to the UNIX socket, which crosses the namespace boundary as a file. With
``--`` and a command, ``tcp2unix`` starts the relay in the background and execs the
command, so the system finds the proxy at the same address it would on the host.
"""
import os
import socket
import sys
import threading


def _pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _serve(listener, connect):
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        try:
            upstream = connect()
        except OSError:
            client.close()
            continue
        threading.Thread(target=_pump, args=(client, upstream), daemon=True).start()
        threading.Thread(target=_pump, args=(upstream, client), daemon=True).start()


def unix2tcp(sock_path, port):
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path)
    os.chmod(sock_path, 0o600)
    listener.listen(64)

    def connect():
        return socket.create_connection(("127.0.0.1", port))

    _serve(listener, connect)


def tcp2unix(port, sock_path):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(64)

    def connect():
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
        return s

    _serve(listener, connect)


def main(argv):
    role = argv[0]
    if role == "unix2tcp":
        unix2tcp(argv[1], int(argv[2]))
        return 0
    if role == "tcp2unix":
        tcp2unix(int(argv[1]), argv[2])
        return 0
    raise SystemExit(f"unknown role {role}")


if __name__ == "__main__":
    # tcp2unix with a command: keep a relay process alive alongside the command.
    if len(sys.argv) > 4 and sys.argv[1] == "tcp2unix" and sys.argv[4] == "--":
        port, sock_path, cmd = int(sys.argv[2]), sys.argv[3], sys.argv[5:]
        if os.fork() == 0:
            tcp2unix(port, sock_path)
            os._exit(0)
        os.execvp(cmd[0], cmd)
    sys.exit(main(sys.argv[1:]))
'''


def proxy_port(url: str) -> int | None:
    """The loopback port of the tracking proxy in ``url``, or None for a remote URL."""
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.hostname in ("127.0.0.1", "localhost", "::1") and u.port:
        return int(u.port)
    return None


def start_bridge(argv: list[str], port: int | None, sock_path: Path, script_path: Path):
    """For a private network namespace: write the forwarder, start its host half
    (UNIX socket -> proxy port) and return ``(argv wrapped to start the sandbox
    half, host process)`` once the host half's socket exists, since the sandbox
    binds that file in as it starts. With no loopback proxy there is nothing to bridge.

    A socket file left at ``sock_path`` by an earlier bridge is removed first: a bridge that is ended leaves its file,
    and an adapter may start a bridge for every session in the same private directory. The wait would find that file
    at once, so bubblewrap would start while the new bridge was removing it and binding its own, and fail with "Can't
    bind mount .../proxy.sock ... No such file or directory" or "Unable to find .../proxy.sock (deleted) in mount
    table", in a persistent row about one session in a hundred."""
    if port is None:
        return list(argv), None
    script_path.write_text(NETBRIDGE)
    sock_path.unlink(missing_ok=True)
    host = subprocess.Popen([sys.executable, str(script_path), "unix2tcp", str(sock_path), str(port)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while not sock_path.exists() and host.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if not sock_path.exists():
        host.kill()
        raise RuntimeError(f"network bridge did not come up on {sock_path} (UNIX socket paths must be "
                           "short; keep it under the run's /tmp directory)")
    inner = ["/usr/bin/python3", str(script_path), "tcp2unix", str(port), str(sock_path), "--", *argv]
    return inner, host


# What a Linux sandbox shows of the host, read-only: its programs and their libraries, /etc and /opt. Nothing else of
# the host's tree exists inside: not its home directories, container volumes or temporary directory, and not /run or
# /var, where the sockets of the daemons outside it live.
SYSTEM_ROOTS = ("/usr", "/etc", "/opt", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
CPU_TOPOLOGY = "/sys/devices/system/cpu"  # how many processors there are, which numeric libraries read
# What a sandbox may consume, for ``prlimit``: no core files, at most this many open files, processes and threads (counted
# in its own user namespace on Linux 5.14 and later, and across the user's sandboxes above it), and bytes in one file.
LIMITS = {"nofile": (resource.RLIMIT_NOFILE, 65_536), "nproc": (resource.RLIMIT_NPROC, 8_192),
          "fsize": (resource.RLIMIT_FSIZE, 64 * 2**30), "core": (resource.RLIMIT_CORE, 0)}


def _limits() -> list[str]:
    """``prlimit``'s options for :data:`LIMITS`, each no higher than the hard limit this process itself has."""
    options = []
    for name, (which, wanted) in LIMITS.items():
        hard = resource.getrlimit(which)[1]
        value = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
        options.append(f"--{name}={value}:{value}")
    return options


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _links_to(path: str | Path) -> tuple[list[tuple[str, Path]], Path]:
    """The symlinks a lookup of ``path`` passes through, as ``(target as written, link path)`` in the order it meets
    them, and the real path it ends at. Recreated inside the sandbox as links, they keep what a program learns about
    itself through them as it is on the host: a launcher in ``~/.local/bin`` that finds its release through
    ``/proc/self/exe``, a venv interpreter that finds its runtime (bound in place of its link, prime-agent looks for
    its package.json in ``~/.local/bin`` and does not start)."""
    links: list[tuple[str, Path]] = []
    pending = list(Path(os.path.abspath(Path(path).expanduser())).parts[1:])
    current = Path("/")
    while pending and len(links) < 40:
        current = current / pending.pop(0)
        if os.path.islink(current):
            target = os.readlink(current)
            links.append((target, current))
            # the rest of the lookup continues from the target (a relative one from the link's directory, which is
            # already free of links)
            pending = list(Path(os.path.normpath(current.parent / target)).parts[1:]) + pending
            current = Path("/")
    return links, current


def bwrap_command(argv: list[str], spec: SandboxSpec, share_net: bool = False, seccomp_file: Path | None = None) -> list[str]:
    """Bubblewrap, as an allowlist like the seatbelt profile. The sandbox starts from an empty root and shows only
    :data:`SYSTEM_ROOTS`, read-only, and what the system is declared to need: its read paths and any existing absolute
    path its command line names (its executable, the bridge script) read-only, each followed through its links unless it
    lies in what the system can write or in an opaque path, then its opaque paths read-only, then its private state, HOME
    and temporary directory writable. The
    benchmark trees stay hidden wherever they lie, even under something bound back. Private user, network, pid, IPC, UTS
    and cgroup namespaces: loopback only (bubblewrap brings ``lo`` up), so the system reaches the tracking proxy through
    the bridge of :func:`start_bridge` and nothing else, and no process, shared memory or message queue of the host's or
    another sandbox's is in reach. ``share_net`` (the host's network) is for a system's own maintenance commands that must
    reach a gateway it runs (never for a model-driven turn). ``seccomp_file`` holds the compiled filter
    (:mod:`continual_arc_baselines.systems.seccomp`), which bubblewrap loads from a file descriptor, opened here by a
    shell so that the command stays one argument list.

    An allowlist rather than the whole filesystem with a list of trees hidden: under such a list a harness reads whatever
    it does not name, other runs' transcripts and the host's ``.env`` among them, and reaches any socket it does not
    name, the systemd manager's among them (a hostile learner did, from a test copy of a verification runner, on 5
    November 2026, and so started a process outside every namespace)."""
    raw = spec
    spec = spec.normalised()
    shown = lambda p: any(_under(p, Path(root)) for root in SYSTEM_ROOTS)  # noqa: E731
    # what the confined code can write, or wrote, decides nothing: its links are not followed
    unfollowed = [*spec.opaque_paths, *spec.write_paths, *(p for p in (spec.tmp_dir, spec.home) if p),
                  *(Path(os.path.abspath(p)) for p in [*raw.opaque_paths, *raw.write_paths])]
    cmd = ["bwrap", "--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-uts", "--unshare-cgroup-try",
           "--hostname", "sandbox", "--die-with-parent", "--share-net" if share_net else "--unshare-net"]
    if seccomp_file is not None:
        cmd += ["--seccomp", "9"]
    for root in SYSTEM_ROOTS:
        if os.path.islink(root):
            cmd += ["--symlink", os.readlink(root), root]
        elif os.path.isdir(root):
            cmd += ["--ro-bind", root, root]
    cmd += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/var/tmp", "--dir", "/run"]
    if os.path.isdir(CPU_TOPOLOGY):
        cmd += ["--ro-bind", CPU_TOPOLOGY, CPU_TOPOLOGY]
    reals: set[Path] = set()
    links: dict[Path, str] = {}
    named = [*raw.read_paths, *(a for a in argv if a.startswith("/"))]
    if share_net:
        named.append("/etc/resolv.conf")  # on most systems a link into /run, where the resolver writes it
    for path in named:
        if any(_under(Path(os.path.abspath(Path(path).expanduser())), u) for u in unfollowed):
            continue
        chain, real = _links_to(path)
        if not real.exists():
            continue
        if not shown(real):
            reals.add(real)
        links.update({link: target for target, link in chain if not shown(link)})
    ro: list[Path] = list(spec.opaque_paths)
    for real in sorted(reals, key=lambda p: (len(p.parts), str(p))):
        # parents first; a path inside a directory already bound back is visible through it
        if not any(d.is_dir() and _under(real, d) for d in ro):
            ro.append(real)
    for real in ro:
        cmd += ["--ro-bind", str(real), str(real)]
    # a link inside a directory bound back is already there, as on the host
    for link, target in sorted(links.items(), key=lambda kv: (len(kv[0].parts), str(kv[0]))):
        if not any(d.is_dir() and _under(link, d) and link != d for d in ro):
            cmd += ["--symlink", target, str(link)]
    rw = list(spec.write_paths) + [p for p in (spec.tmp_dir, spec.home) if p]
    for p in rw:
        cmd += ["--bind", str(p), str(p)]
    for m in spec.mask_paths:
        if any(_under(m, d) and m != d for d in [*ro, *rw]):
            cmd += ["--tmpfs", str(m)]
    cmd += ["--remount-ro", "/", "--", *argv]
    limited = ["prlimit", *_limits(), *cmd]
    return ["/bin/sh", "-c", 'exec 9<"$1"; shift; exec "$@"', "sh", str(seccomp_file), *limited] if seccomp_file else limited


SELF_CHECK = r"""
import ctypes, json, os, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
refused = {}
for name, number, args in (("unshare", 272, (0x10000000,)), ("mount", 165, (0, 0, 0, 0, 0)), ("bpf", 321, (0, 0, 0)),
                           ("io_uring_setup", 425, (1, 0)), ("keyctl", 250, (0, 0, 0, 0, 0))):
    result = libc.syscall(ctypes.c_long(number), *[ctypes.c_long(a) for a in args])
    refused[name] = result < 0 and ctypes.get_errno() == 1
json.dump({"refused": refused, "pids": len([p for p in os.listdir("/proc") if p.isdigit()]), "host": os.uname().nodename,
           "shown": [p for p in ("/run/user", "/run/dbus", "/run/systemd", "/var/log", "/var/lib", "/root", "/boot", "/sys/fs/cgroup",
                                 "/proc/kcore", "/dev/kmsg") if os.path.lexists(p) and not (p == "/proc/kcore")]}, sys.stdout)
"""


def confinement_problems(profile_path: Path) -> list[str]:
    """What a process in this machine's sandbox can do that it should not, from inside: a check that fails loudly where a
    machine was set up with a bubblewrap, a kernel or a filter that holds less than this module says. Empty when the
    sandbox holds."""
    backend = resolve_backend("bwrap")
    argv = wrap_command(backend, ["/usr/bin/python3", "-I", "-c", SELF_CHECK], SandboxSpec(), profile_path)
    done = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    if done.returncode != 0:
        return [f"the sandbox did not start (exit {done.returncode}): {done.stderr[-300:]}"]
    seen = json.loads(done.stdout)
    problems = [f"{name} was not refused" for name, was in seen["refused"].items() if not was]
    problems += [f"{path} shows in the sandbox" for path in seen["shown"]]
    if seen["pids"] > 3 or seen["host"] != "sandbox":
        problems.append(f"the sandbox shares the host's processes or name ({seen['pids']} pids, {seen['host']})")
    return problems


def _sb_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(spec: SandboxSpec, share_net: bool = False) -> str:
    """Render a Seatbelt profile: deny by default, then the allowances in ``spec``. ``share_net`` adds outbound
    connections anywhere, as bubblewrap's ``--share-net`` does (:func:`bwrap_command`)."""
    spec = spec.normalised()
    rw = list(spec.write_paths)
    if spec.tmp_dir:
        rw.append(spec.tmp_dir)
    if spec.home:
        rw.append(spec.home)
    socket_dirs = list(spec.unix_socket_dirs) + ([spec.tmp_dir] if spec.tmp_dir else [])
    reads = [Path(p) for p in _SYSTEM_READ if Path(p).exists()] + spec.read_paths + spec.opaque_paths + rw

    lines = [
        "(version 1)",
        "(deny default)",
        ";; process bookkeeping every runtime needs; none of it reaches user data",
        "(allow process-exec process-fork process-info*)",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow ipc-posix*)",
        "(allow file-ioctl)",
        "(allow pseudo-tty)",
        "(allow file-map-executable)",
        ";; path resolution needs stat() on parents; this leaks names, never contents",
        "(allow file-read-metadata)",
        ";; readable: the operating system, the system's own install and runtime, its state",
        '(allow file-read* (literal "/"))',  # dyld reads the root directory; every exec aborts without it
    ]
    for p in reads:
        lines.append(f"(allow file-read* (subpath {_sb_str(str(p))}))")
    lines.append(";; writable: only the system's private state, HOME and temporary directory")
    for p in rw:
        lines.append(f"(allow file-write* (subpath {_sb_str(str(p))}))")
    lines.append("(allow file-write* (literal \"/dev/null\") (literal \"/dev/tty\") (regex #\"^/dev/tty[a-z0-9]+$\"))")
    lines.append(";; network: the tracking proxy on loopback, and private UNIX sockets")
    for port in spec.loopback_ports:
        lines.append(f'(allow network-outbound (remote ip "localhost:{int(port)}"))')
    for port in spec.listen_ports:
        lines.append(f'(allow network-bind (local ip "localhost:{int(port)}"))')
        lines.append(f'(allow network-inbound (local ip "localhost:{int(port)}"))')
        lines.append(f'(allow network-outbound (remote ip "localhost:{int(port)}"))')
    for d in socket_dirs:
        s = _sb_str(str(d))
        lines.append(f"(allow network-outbound (remote unix-socket (subpath {s})))")
        lines.append(f"(allow network-bind (local unix-socket (subpath {s})))")
        lines.append(f"(allow network-inbound (local unix-socket (subpath {s})))")
    if share_net:
        lines.append(";; the host's network: a command that is not a model-driven turn (see bwrap_command)")
        lines.append("(allow network-outbound)")
    return "\n".join(lines) + "\n"


def wrap_command(backend: str, argv: list[str], spec: SandboxSpec, profile_path: Path,
                 share_net: bool = False) -> list[str]:
    """Return the command line that runs ``argv`` inside the sandbox. ``profile_path`` is where the sandbox's profile goes
    (Seatbelt's, or the compiled filter for bubblewrap), which must be outside everything the sandbox can write."""
    if backend == "none":
        return list(argv)
    if backend == "seatbelt":
        profile_path.write_text(seatbelt_profile(spec, share_net=share_net))
        return [SEATBELT, "-f", str(profile_path), *argv]
    if backend == "bwrap":
        program = seccomp.program()
        if program is None:
            return bwrap_command(argv, spec, share_net=share_net)
        profile_path.write_bytes(program)
        return bwrap_command(argv, spec, share_net=share_net, seccomp_file=profile_path)
    raise ValueError(f"unknown sandbox backend {backend!r}")


def python_runtime_root(python: Path) -> Path | None:
    """The installation a venv's interpreter comes from (``home`` in pyvenv.cfg): for a venv this package made or trusts.
    A submission's build code can write its own venv's pyvenv.cfg, so it never names what a sandbox binds."""
    cfg = Path(python).absolute().parent.parent / "pyvenv.cfg"  # the venv, not the resolved binary
    if not cfg.is_file():
        return None
    for line in cfg.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "home" and value.strip():
            home = Path(value.strip())
            return home.parent if home.name == "bin" else home
    return None


def darwin_user_dirs() -> list[Path]:
    """The per-user temporary and cache directories macOS libraries use regardless
    of ``TMPDIR`` (``/var/folders/.../T`` and ``.../C``). Empty elsewhere."""
    if platform.system() != "Darwin":
        return []
    out = []
    for name in ("DARWIN_USER_TEMP_DIR", "DARWIN_USER_CACHE_DIR"):
        try:
            value = subprocess.run(["getconf", name], capture_output=True, text=True,
                                   timeout=10, check=False).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
        if value and Path(value).is_dir():
            out.append(Path(value))
    return out


def package_manager_roots(path: Path) -> list[Path]:
    """Extra read roots an executable under a package manager prefix needs.

    Homebrew binaries link against libraries in ``<prefix>/opt/*`` and
    ``<prefix>/Cellar/*`` rather than next to themselves, so a formula's own
    directory is not enough to run it."""
    resolved = Path(path).resolve()
    out: list[Path] = []
    for prefix in (Path("/opt/homebrew"), Path("/usr/local"), Path("/home/linuxbrew/.linuxbrew")):
        if prefix in resolved.parents:
            for sub in ("opt", "Cellar", "lib", "bin", "etc", "share"):
                if (prefix / sub).exists():
                    out.append(prefix / sub)
    return out


def install_root(executable: str) -> Path | None:
    """The directory tree an executable lives in: the folder of the file it resolves to, symlinks followed, or that
    folder's parent when it is a ``bin``.

    prime-agent's TypeScript installer links ``~/.local/bin/prime-agent`` to ``~/.local/share/prime-agent/bin/prime-agent``,
    itself a link to the release's executable, so its tree is the release,
    ``~/.local/share/prime-agent/releases/<version>-<platform>-<digest>``; install-rust.sh's ``<prefix>/bin/prime-agent``
    is a launcher script, not a link, so its tree is ``<prefix>``, which holds the release in ``share/prime-agent``.
    """
    found = shutil.which(executable) if os.path.sep not in executable else executable
    if not found:
        return None
    real = Path(found).resolve()
    parent = real.parent
    return parent.parent if parent.name == "bin" else parent
