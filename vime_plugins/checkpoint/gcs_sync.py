"""GCS checkpoint sync, ported from ai21-verl (verl/ai21/utils/checkpoint_files.py).

vime saves checkpoints locally (``--save`` for Megatron, ``--save-hf`` for HF) and has no remote
IO. This adds GCS upload in two ways, both additive (no upstream vime changes):

1. **Sidecar watcher (recommended)** — run alongside training; it periodically mirrors the local
   checkpoint dir to GCS as new checkpoints appear::

       python -m vime_plugins.checkpoint.gcs_sync watch \
           --local-dir $SAVE --gcs-dest gs://bucket/run123 --poll-interval 60

   Uses ``gsutil -m rsync -r`` (incremental), so re-runs only upload new/changed files.

2. **Programmatic hook** — :func:`sync_checkpoint_dir` / :func:`sync_paths_gs` can be called from
   a custom ``--checkpoint-sync-path`` callback if/when that core hook is added; until then the
   sidecar covers the use-case without touching vime core.

Requires the ``gcloud`` / ``gsutil`` CLIs (present in the AI21 training image). The fast/slow
fallback (``gsutil cp`` then a more conservative ``gsutil cp``) mirrors the upstream behaviour.

NOTE: ESI/preemption-triggered early checkpointing from ai21-verl is intentionally deferred.
"""

import argparse
import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VIME_LOGGING_LEVEL", os.getenv("VERL_LOGGING_LEVEL", "INFO")))


def threaded(func):
    """Run ``func`` in a background thread, re-raising thread exceptions in the main thread."""

    def handle_thread_exception(args):
        if args.exc_type is not None:
            print(f"Thread {args.thread} failed with {args.exc_type}: {args.exc_value}")
            raise args.exc_value

    threading.excepthook = handle_thread_exception

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


def sync_paths_gs(source_path: str, dest_path: str, wildcard: str, log_stdout: bool = False, fast_timeout: int = 150):
    """Copy ``source_path/wildcard`` to ``dest_path``, fast path first then a slower fallback."""
    try:
        logger.info(f"Syncing paths from {source_path}/{wildcard} to {dest_path}")
        _sync_paths_gs_fast(source_path, dest_path, wildcard, timeout=fast_timeout, log_stdout=log_stdout)
    except subprocess.CalledProcessError as e:
        logger.info(f"sync_paths_gs_fast from {source_path} to {dest_path} failed with exception: {e}")
        logger.info("Falling back to sync_paths_gs_slow")
        _sync_paths_gs_slow(source_path, dest_path, wildcard, log_stdout=log_stdout)


def sync_file_gs(source_path: str, dest_path: str, log_stdout: bool = False):
    logger.info(f"Syncing file from {source_path} to {dest_path}")
    cmd = ["gcloud", "storage", "cp", source_path, dest_path]
    _run_cmd(cmd, log_stdout=log_stdout)


def sync_dir_gs(local_dir: str, gcs_dest: str, log_stdout: bool = False):
    """Incrementally mirror a local directory tree to GCS (idempotent)."""
    logger.info(f"rsync {local_dir} -> {gcs_dest}")
    cmd = ["gsutil", "-m", "rsync", "-r", local_dir, gcs_dest]
    _run_cmd(cmd, log_stdout=log_stdout)


def sync_checkpoint_dir(local_path: str, gcs_dest: str, log_stdout: bool = False):
    """Upload one checkpoint directory to GCS (suitable as a --checkpoint-sync-path callback body)."""
    sync_dir_gs(local_path, gcs_dest, log_stdout=log_stdout)


@threaded
def sync_paths_gs_threaded(
    source_path: str, dest_path: str, wildcard: str, log_stdout: bool = False, fast_timeout: int = 150
):
    sync_paths_gs(source_path, dest_path, wildcard, log_stdout=log_stdout, fast_timeout=fast_timeout)


@threaded
def sync_file_gs_threaded(source_path: str, dest_path: str, log_stdout: bool = False):
    sync_file_gs(source_path, dest_path, log_stdout=log_stdout)


def check_file_exists_gs(path: str) -> bool:
    try:
        _run_cmd(["gsutil", "stat", path], log_stdout=False)
        return True
    except subprocess.CalledProcessError:
        return False


def watch_and_sync(local_dir: str, gcs_dest: str, poll_interval: float = 60.0, log_stdout: bool = False):
    """Sidecar loop: every ``poll_interval`` seconds, rsync ``local_dir`` to ``gcs_dest``.

    Runs until interrupted. ``gsutil rsync`` is incremental, so each pass only uploads what
    changed since the last checkpoint save.
    """
    logger.info(f"watch_and_sync: mirroring {local_dir} -> {gcs_dest} every {poll_interval}s")
    while True:
        try:
            if os.path.isdir(local_dir) and os.listdir(local_dir):
                sync_dir_gs(local_dir, gcs_dest, log_stdout=log_stdout)
        except subprocess.CalledProcessError as e:
            logger.warning(f"watch_and_sync: rsync failed (will retry): {e}")
        time.sleep(poll_interval)


def _sync_paths_gs_slow(source_path: str, dest_path: str, wildcard: str, log_stdout: bool):
    cmd = [
        "gsutil",
        "-m",
        "-o",
        "'GSUtil:parallel_thread_count=1'",
        "-o",
        "'GSUtil:sliced_object_download_max_components=8'",
        "cp",
        f"{source_path}/{wildcard}",
        f"{dest_path}",
    ]
    _run_cmd(cmd, log_stdout=log_stdout)


def _sync_paths_gs_fast(source_path: str, dest_path: str, wildcard: str, log_stdout: bool, timeout=None):
    cmd = ["gsutil", "-m", "cp", "-r", f"{source_path}/{wildcard}", f"{dest_path}"]
    if timeout:
        cmd = ["timeout", f"{timeout}s"] + cmd
    _run_cmd(cmd, log_stdout=log_stdout)


def _run_cmd(cmd: list[str], log_stdout: bool):
    stdout = None
    stderr = None
    shell = False
    if not log_stdout:
        cmd = " ".join(cmd) + " > /dev/null 2>&1"
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        shell = True

    subprocess.check_call(cmd, stdout=stdout, shell=shell, stderr=stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="GCS checkpoint sync for vime.")
    sub = parser.add_subparsers(dest="command", required=True)

    w = sub.add_parser("watch", help="Run a sidecar that mirrors a local checkpoint dir to GCS.")
    w.add_argument("--local-dir", required=True, help="Local checkpoint dir (e.g. vime --save).")
    w.add_argument("--gcs-dest", required=True, help="Destination, e.g. gs://bucket/run-id.")
    w.add_argument("--poll-interval", type=float, default=60.0)
    w.add_argument("--log-stdout", action="store_true")

    s = sub.add_parser("sync", help="Mirror a local dir to GCS once and exit.")
    s.add_argument("--local-dir", required=True)
    s.add_argument("--gcs-dest", required=True)
    s.add_argument("--log-stdout", action="store_true")

    args = parser.parse_args()
    if args.command == "watch":
        watch_and_sync(args.local_dir, args.gcs_dest, poll_interval=args.poll_interval, log_stdout=args.log_stdout)
    elif args.command == "sync":
        sync_dir_gs(args.local_dir, args.gcs_dest, log_stdout=args.log_stdout)


if __name__ == "__main__":
    main()
