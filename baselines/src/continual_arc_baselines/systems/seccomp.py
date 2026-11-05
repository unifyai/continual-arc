"""The system calls a sandboxed process tree may not make, as the compiled filter bubblewrap loads (``--seccomp``).

A sandbox shares the host's kernel, so what its code can ask of that kernel is the last wall: every call below reaches
code that no learner needs and that has given root to unprivileged processes before, above all user namespaces, which
open the kernel's network and filesystem code to an unprivileged caller, and io_uring, BPF and the new mount calls. The
list follows what Docker's default profile refuses (https://docs.docker.com/engine/security/seccomp/), with the same
treatment of ``clone3``, whose flags a filter cannot read: it answers ENOSYS, so C libraries fall back to ``clone``,
whose flags it can.

A call numbered past :data:`NEWEST_KNOWN` also answers ENOSYS, so a newer kernel opens no new door, and a 32-bit
process (``int 0x80``, whose numbers mean other calls) or an x32 call is refused whole: the filter reads only x86-64's
numbers, and on any other machine :func:`program` is None and the sandbox says it has no filter.
"""

from __future__ import annotations

import errno
import platform
import struct

AUDIT_ARCH_X86_64 = 0xC000003E
X32_BIT = 0x40000000
NEWEST_KNOWN = 469  # x86-64's file_setattr; every call Linux had when this list was read

# x86-64's numbers (asm/unistd_64.h), the calls no sandboxed process may make: it answers EPERM.
DENIED = {
    "ptrace": 101, "syslog": 103, "uselib": 134, "ustat": 136, "sysfs": 139, "modify_ldt": 154, "pivot_root": 155,
    "_sysctl": 156, "adjtimex": 159, "chroot": 161, "acct": 163, "settimeofday": 164, "mount": 165, "umount2": 166,
    "swapon": 167, "swapoff": 168, "reboot": 169, "sethostname": 170, "setdomainname": 171, "iopl": 172, "ioperm": 173,
    "create_module": 174, "init_module": 175, "delete_module": 176, "get_kernel_syms": 177, "query_module": 178,
    "quotactl": 179, "nfsservctl": 180, "getpmsg": 181, "putpmsg": 182, "afs_syscall": 183, "tuxcall": 184,
    "security": 185, "lookup_dcookie": 212, "clock_settime": 227, "vserver": 236, "mbind": 237, "set_mempolicy": 238,
    "get_mempolicy": 239, "kexec_load": 246, "add_key": 248, "request_key": 249, "keyctl": 250, "migrate_pages": 256,
    "unshare": 272, "vmsplice": 278, "move_pages": 279, "perf_event_open": 298, "fanotify_init": 300,
    "name_to_handle_at": 303, "open_by_handle_at": 304, "clock_adjtime": 305, "setns": 308, "process_vm_readv": 310,
    "process_vm_writev": 311, "kcmp": 312, "finit_module": 313, "kexec_file_load": 320, "bpf": 321, "userfaultfd": 323,
    "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427, "open_tree": 428, "move_mount": 429,
    "fsopen": 430, "fsconfig": 431, "fsmount": 432, "fspick": 433, "pidfd_getfd": 438, "process_madvise": 440,
    "mount_setattr": 442, "quotactl_fd": 443, "memfd_secret": 447, "process_mrelease": 448,
    "set_mempolicy_home_node": 450,
}
SYS_IOCTL, SYS_SOCKET, SYS_SOCKETPAIR, SYS_CLONE, SYS_PERSONALITY, SYS_CLONE3 = 16, 41, 53, 56, 135, 435

# clone()'s flags that make a namespace, of which the sandbox has all it needs: mount, cgroup, UTS, IPC, user, PID, network.
CLONE_NEW = 0x00020000 | 0x02000000 | 0x04000000 | 0x08000000 | 0x10000000 | 0x20000000 | 0x40000000
AF_UNIX, AF_INET, AF_NETLINK, AF_INET6 = 1, 2, 16, 10
NETLINK_ROUTE = 0
# Pushing input into a terminal (TIOCSTI) and the console's own calls (TIOCLINUX) reach the session outside.
DENIED_IOCTLS = (0x5412, 0x541C)
# What a process may ask of personality(): Linux's ordinary values, without ADDR_NO_RANDOMIZE, the exploit's helper.
PERSONALITIES = (0x0, 0x8, 0x20000, 0x20008, 0xFFFFFFFF)

RET_ALLOW, RET_KILL_PROCESS, RET_ERRNO = 0x7FFF0000, 0x80000000, 0x00050000
LD_W_ABS, JEQ, JGE, JGT, JSET, RET = 0x20, 0x15, 0x35, 0x25, 0x45, 0x06
NR, ARCH = 0, 4  # offsets of the call's number and architecture in struct seccomp_data


def _arg(index: int) -> int:
    """The offset of a call's argument's low 32 bits (little-endian) in struct seccomp_data."""
    return 16 + 8 * index


def _assemble(lines: list) -> bytes:
    """Classic BPF from a list of ``(code, k, true target, false target)`` and label strings, a target being a label or
    None for the next instruction. A jump reaches at most 255 instructions, which :func:`program`'s layout keeps to."""
    labels: dict[str, int] = {}
    position = 0
    for line in lines:
        if isinstance(line, str):
            labels[line] = position
        else:
            position += 1
    out, position = [], 0
    for line in lines:
        if isinstance(line, str):
            continue
        code, k, true, false = line
        position += 1
        offset = [0 if target is None else labels[target] - position for target in (true, false)]
        if not all(0 <= value <= 255 for value in offset):
            raise ValueError(f"a jump out of reach at instruction {position}: {offset}")
        out.append(struct.pack("<HBBI", code, offset[0], offset[1], k & 0xFFFFFFFF))
    return b"".join(out)


def _load(offset: int) -> tuple:
    return (LD_W_ABS, offset, None, None)


def _is(value: int, target: str) -> tuple:
    return (JEQ, value, target, None)


def program() -> bytes | None:
    """The filter as bubblewrap reads it, an array of ``struct sock_filter``; None on a machine it has no numbers for."""
    if platform.machine() != "x86_64":
        return None
    errno_of = lambda value: RET_ERRNO | value  # noqa: E731
    return _assemble([
        _load(ARCH),
        (JEQ, AUDIT_ARCH_X86_64, None, "kill"),
        _load(NR),
        (JGE, X32_BIT, "nosys", None),
        (JGT, NEWEST_KNOWN, "nosys", None),
        _is(SYS_CLONE3, "nosys"),
        _is(SYS_CLONE, "clone"),
        _is(SYS_SOCKET, "socket"),
        _is(SYS_SOCKETPAIR, "socketpair"),
        _is(SYS_IOCTL, "ioctl"),
        _is(SYS_PERSONALITY, "personality"),
        *(_is(number, "denied") for number in sorted(DENIED.values())),
        (RET, RET_ALLOW, None, None),
        "clone",
        _load(_arg(0)),
        (JSET, CLONE_NEW, "denied", "allow"),
        "socket",
        _load(_arg(0)),
        _is(AF_UNIX, "allow"),
        _is(AF_INET, "allow"),
        _is(AF_INET6, "allow"),
        (JEQ, AF_NETLINK, None, "family"),
        _load(_arg(2)),  # netlink's routing protocol alone: libc reads the (private) interfaces through it
        (JEQ, NETLINK_ROUTE, "allow", "family"),
        "socketpair",
        _load(_arg(0)),
        (JEQ, AF_UNIX, "allow", "family"),
        "ioctl",
        _load(_arg(1)),
        *(_is(request, "denied") for request in DENIED_IOCTLS),
        (RET, RET_ALLOW, None, None),
        "personality",
        _load(_arg(0)),
        *(_is(value, "allow") for value in PERSONALITIES),
        "denied",
        (RET, errno_of(errno.EPERM), None, None),
        "allow",
        (RET, RET_ALLOW, None, None),
        "nosys",
        (RET, errno_of(errno.ENOSYS), None, None),
        "family",
        (RET, errno_of(errno.EAFNOSUPPORT), None, None),
        "kill",
        (RET, RET_KILL_PROCESS, None, None),
    ])
