"""Push work-in-progress files from a Colab session to GitHub, so a recycled VM doesn't lose hours of work.

The notebook's setup cell puts the token-bearing URL in the GIT_PUSH_URL environment variable (memory only, never
written to git config or printed). Without it, syncing is a no-op, so the same scripts run unchanged locally.

Syncing never runs git in the working checkout. Git operations like `pull --rebase` replace files on disk, and a
process that keeps a file open (a sampler appending results) would silently go on writing into a detached copy. That
lost ~850 sampled prompts once. Instead, files are *copied* into a separate clone (.colab_sync/) and committed and
pushed from there, under a cross-process lock, with retries.
"""

import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

COAUTHOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
SYNC_DIR = Path(".colab_sync")
LOCK_PATH = Path(".colab_sync.lock")


def _git(*args: str, secret: str, cwd: Path = SYNC_DIR) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {args[0]} failed: " + (r.stderr or r.stdout).replace(secret, "***").strip()[-300:])
    return r


@contextmanager
def _exclusive():
    """Cross-process exclusive lock: fcntl on Linux (Colab), msvcrt on Windows (local tests)."""
    with open(LOCK_PATH, "a+") as fh:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:
            import msvcrt
            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.2)
        yield


def _ensure_clone(url: str) -> None:
    if (SYNC_DIR / ".git").exists():
        return
    shutil.rmtree(SYNC_DIR, ignore_errors=True)
    _git("clone", "-q", "--depth", "50", url, str(SYNC_DIR), secret=url, cwd=Path("."))
    for key, value in (("user.name", "ankankisku-lab"),
                       ("user.email", "239427487+ankankisku-lab@users.noreply.github.com")):
        _git("config", key, value, secret=url)


def _copy_in(path: str) -> None:
    src, dst = Path(path), SYNC_DIR / path
    if src.is_dir():
        for f in src.rglob("*"):
            if f.is_file():
                (SYNC_DIR / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, SYNC_DIR / f)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def sync(paths: list[str], message: str, attempts: int = 3) -> bool:
    """Copy `paths` into the sync clone, commit and push to main. Returns False (and doesn't raise) when syncing
    isn't configured or keeps failing, so a transient network error never kills a long run."""
    url = os.environ.get("GIT_PUSH_URL")
    if not url:
        return False
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        return False
    with _exclusive():
        for attempt in range(1, attempts + 1):
            try:
                _ensure_clone(url)
                _git("fetch", "-q", url, "main", secret=url)
                _git("reset", "-q", "--hard", "FETCH_HEAD", secret=url)  # the clone holds no work of its own
                for p in existing:
                    _copy_in(p)
                _git("add", *existing, secret=url)
                if not subprocess.run(["git", "-C", str(SYNC_DIR), "diff", "--cached", "--quiet"]).returncode:
                    return True  # nothing new
                _git("commit", "-q", "-m", message, "-m", COAUTHOR, secret=url)
                _git("push", "-q", url, "HEAD:main", secret=url)
                return True
            except RuntimeError as e:
                print(f"[sync] attempt {attempt}/{attempts} failed: {e}", flush=True)
                time.sleep(5 * attempt)
    return False
