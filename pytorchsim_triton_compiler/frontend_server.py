"""frontend_server: the triton-compile worker's imports paid once per environment, not once per kernel.

The worker (frontend.py) runs in a process of its own, and a kernel file inductor wrote imports
torch._inductor.runtime, which brings torch._dynamo with it: some 3s of a small kernel's ~10s
compile, every kernel. The server imports those once, then forks one child per kernel; the child
runs frontend.py from disk, as `python frontend.py ...` would, with the caller's argv, environment,
working directory and stdout/stderr. A child is a fresh copy of the imported server, so no kernel
sees another's state, and the compiler's own modules are read from disk by each child, never held
over from an older checkout.

    python frontend_server.py <socket path>      the server (started by served(), not by hand)

served() is the client: it runs the worker through the server, starting one beside the first call
it finds none for, and answers None (the caller runs the worker as a subprocess) whenever the
server is off, not up yet, or anything goes wrong on the way. This file imports only the standard
library at the top: the server must not hold the compiler's modules."""
import hashlib
import json
import os
import socket
import struct
import sys
import tempfile

ENV = "PYTORCHSIM_COMPILE_FRONTEND_SERVER"
"""Set to 0 to run every worker as a subprocess, as before the server."""

IDLE_SECONDS = 600
"""A server with no request and no child for this long exits."""

_VOLATILE = ("PWD", "OLDPWD", "_", "SHLVL")
"""Environment variables a shell changes from call to call that no import reads."""


def _key(env):
    """One server per interpreter, server code and environment: anything else its imports read."""
    h = hashlib.sha256()
    h.update(sys.executable.encode())
    with open(__file__, "rb") as f:
        h.update(f.read())
    h.update(json.dumps(sorted((k, v) for k, v in env.items() if k not in _VOLATILE)).encode())
    return h.hexdigest()[:20]


def socket_path(env):
    """The server's socket for `env`, in the job's temporary directory (a Unix socket path is at
    most 107 bytes, so /tmp when that one is too deep)."""
    name = f"ptb-frontend-{os.getuid()}-{_key(env)}.sock"
    for d in (env.get("TMPDIR"), tempfile.gettempdir(), "/tmp"):
        if d and len(os.path.join(d, name)) < 100:
            return os.path.join(d, name)
    return None


def _send_msg(sock, obj, fds=()):
    data = json.dumps(obj).encode()
    data = struct.pack("!I", len(data)) + data
    if fds:
        socket.send_fds(sock, [data], list(fds))
    else:
        sock.sendall(data)


def _recv_exact(sock, n, got=b""):
    while len(got) < n:
        chunk = sock.recv(n - len(got))
        if not chunk:
            raise ConnectionError("the other end closed")
        got += chunk
    return got


def _recv_msg(sock, maxfds=0):
    if maxfds:
        data, fds, _, _ = socket.recv_fds(sock, 65536, maxfds)
    else:
        data, fds = sock.recv(65536), []
    if not data:
        raise ConnectionError("the other end closed")
    data = _recv_exact(sock, 4, data)
    n = struct.unpack("!I", data[:4])[0]
    data = _recv_exact(sock, 4 + n, data)
    return json.loads(data[4:4 + n]), fds


# ---------------------------------------------------------------------------------------------- client

def served(cmd, env, cwd=None):
    """Run the worker command `cmd` ([python, frontend.py, args...]) in a child of the server for
    `env`: a subprocess.CompletedProcess, or None when the caller should run it as a subprocess."""
    import subprocess
    if env.get(ENV, os.environ.get(ENV, "1")) in ("0", "", "no", "false"):
        return None
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend.py")
    if len(cmd) < 2 or os.path.abspath(str(cmd[1])) != here:
        return None  # the server runs frontend.py and nothing else
    path = socket_path(env)
    if path is None:
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError:
        sock.close()
        _start(path, env)
        return None
    try:
        with tempfile.TemporaryFile(dir=env.get("TMPDIR")) as out, \
                tempfile.TemporaryFile(dir=env.get("TMPDIR")) as err:
            _send_msg(sock, {"args": [str(c) for c in cmd[2:]], "env": env,
                             "cwd": cwd or os.getcwd()}, fds=(out.fileno(), err.fileno()))
            reply, _ = _recv_msg(sock)
            out.seek(0)
            err.seek(0)
            return subprocess.CompletedProcess(cmd, reply["rc"],
                                               out.read().decode(errors="replace"),
                                               err.read().decode(errors="replace"))
    except (OSError, ValueError, KeyError, ConnectionError):
        return None
    finally:
        sock.close()


def _start(path, env):
    """A server for `path`, started in the background and left to itself; one that finds another
    already holding the path's lock exits at once."""
    import subprocess
    try:
        log = open(path + ".log", "a")
        subprocess.Popen([sys.executable, os.path.abspath(__file__), path], env=env,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                         cwd="/")
        log.close()
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------- server

def _preload():
    """What every worker imports before its first kernel is read: torch, Triton and the runtime
    modules an inductor kernel file imports, those under skip_gpu_driver_setup as the spec loads
    the file (kernel_spec's kernel())."""
    import importlib
    for name in ("torch", "triton", "triton.language", "triton.compiler.compiler", "triton._C.libtriton"):
        importlib.import_module(name)
    from torch._inductor.runtime.triton_helpers import skip_gpu_driver_setup
    with skip_gpu_driver_setup():
        for name in ("torch._inductor.runtime.triton_heuristics", "torch._inductor.runtime.hints"):
            importlib.import_module(name)


def _child(req, fds, frontend):
    """In the forked child: become `python frontend.py argv...` in the caller's environment and
    working directory, writing to the caller's stdout and stderr; never returns."""
    import runpy
    import traceback
    rc = 0
    try:
        os.dup2(fds[0], 1)
        os.dup2(fds[1], 2)
        for fd in fds:
            os.close(fd)
        os.environ.clear()
        os.environ.update(req["env"])
        os.chdir(req["cwd"])
        sys.argv = [frontend] + req["args"]
        runpy.run_path(frontend, run_name="__main__")
    except SystemExit as e:
        if e.code is None:
            rc = 0
        elif isinstance(e.code, int):
            rc = e.code
        else:
            print(e.code, file=sys.stderr)
            rc = 1
    except BaseException:  # noqa: BLE001 -- the child reports it as python would
        traceback.print_exc()
        rc = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(rc)


def serve(path):
    import fcntl
    import select
    import signal
    import time
    lock = open(path + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return  # another server has this path
    t0 = time.time()
    _preload()
    print(f"[frontend-server] {path}: imports in {time.time() - t0:.1f}s, pid {os.getpid()}", flush=True)
    frontend = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend.py")
    if os.path.exists(path):
        os.unlink(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    listener.listen(64)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    waiting = {}  # child pid -> the caller's connection
    idle_since = time.time()
    try:
        while True:
            ready, _, _ = select.select([listener], [], [], 0.2)
            if ready:
                conn, _ = listener.accept()
                try:
                    req, fds = _recv_msg(conn, maxfds=2)
                except (OSError, ValueError, ConnectionError):
                    conn.close()
                    continue
                pid = os.fork()
                if pid == 0:
                    listener.close()
                    conn.close()
                    lock.close()  # the lock is the server's: a child left running must not hold it
                    _child(req, fds, frontend)
                for fd in fds:
                    os.close(fd)
                waiting[pid] = conn
            while waiting:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                conn = waiting.pop(pid, None)
                if conn is not None:
                    rc = os.waitstatus_to_exitcode(status)
                    try:
                        _send_msg(conn, {"rc": rc})
                    except OSError:
                        pass
                    conn.close()
            if waiting or ready:
                idle_since = time.time()
            elif time.time() - idle_since > IDLE_SECONDS:
                break
    finally:
        listener.close()
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    serve(sys.argv[1])
