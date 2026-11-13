"""Post-run draft-update evidence checks for the external-draft NPU smokes.

The smoke drivers only assert that the ray job exits 0 ("Job succeeded" in
the log). Everything the external-draft loop is supposed to do — train a
new draft every round, publish it into the live engine, and keep the next
rollout on a healthy draft — is visible only in the job log, so this module
captures that log and checks the recorded evidence.

The log must be captured at the file-descriptor level: the job is submitted
through ``vime.utils.misc.exec_command``, which runs ``subprocess.run`` with
``capture_output=False``, so the job writes straight to this process's fd 1
and 2 and ``contextlib.redirect_stdout`` never sees it.
"""

from __future__ import annotations

import ast
import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

_DRAFT_TRAIN_MARKER = "External Draft training: "
_SPEC_ACCEPT_KEY = "rollout/spec_accept_length"
_POST_LINE_RE = re.compile(r'"POST (/[\w]+) HTTP/[\d.]+" (\d{3})')
_ROLLOUT_PERF_RE = re.compile(r"perf \d+: (\{.*\})\s*$")

# Only the draft session uses start_draft_weight_update; the update/finish
# endpoints are shared with the target weight-update session.
_DRAFT_CHAIN = ("/start_draft_weight_update", "/update_weights", "/finish_weight_update")


@contextlib.contextmanager
def tee_output(log_path):
    """Mirror this process's fd 1 and 2 into ``log_path`` while it runs.

    Console output keeps streaming through ``tee``, so long smokes remain
    watchable. Descriptor-level capture is required because the submitted
    ray job inherits fd 1/2 directly.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = subprocess.Popen(["tee", str(log_path)], stdin=subprocess.PIPE)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    pipe = tee.stdin.fileno()
    os.dup2(pipe, 1)
    os.dup2(pipe, 2)
    try:
        yield log_path
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tee.stdin.close()
        tee.wait()


def check_draft_update_evidence(
    log_path,
    *,
    num_rollouts,
    min_accept_length,
    last_round_ratio=0.7,
):
    """Assert the log shows the next rollout really used the updated draft.

    Review ask: "checking that the next rollout uses the updated Draft".
    Four conditions, all against the captured job log:

    1. one draft training per rollout, ``draft_version`` incrementing
       1..num_rollouts (a silently skipped round fails here);
    2. one complete draft publish session per rollout: the draft-only
       start_draft_weight_update -> update_weights -> finish_weight_update
       chain, every link answering 200;
    3. every round's rollout spec_accept_length >= min_accept_length —
       the acceptance-collapse failure mode leaves job exit, publishing
       and train metrics green, so this floor is the actual detector;
    4. last round >= first round * last_round_ratio (slow-drift guard).
    """
    with open(log_path, errors="replace") as handle:
        lines = handle.read().splitlines()
    errors = []

    train_events = _draft_train_events(lines)
    versions = [event["draft_version"] for event in train_events]
    expected_versions = list(range(1, num_rollouts + 1))
    if versions != expected_versions:
        errors.append(
            "condition 1 (draft trained every round): expected draft_version"
            f" sequence {expected_versions}, got {versions}"
        )

    sessions = _draft_publish_sessions(_post_events(lines))
    broken = [
        session
        for session in sessions
        if [endpoint for endpoint, _ in session] != list(_DRAFT_CHAIN) or any(status != 200 for _, status in session)
    ]
    if len(sessions) != num_rollouts or broken:
        described = " ".join(
            "[" + " ".join(f"{endpoint}:{status}" for endpoint, status in session) + "]" for session in sessions
        )
        errors.append(
            f"condition 2 (draft publish session all-200): expected"
            f" {num_rollouts} sessions of {list(_DRAFT_CHAIN)}, got"
            f" {len(sessions)}: {described or 'none'}"
        )

    accept = _rollout_accept_lengths(lines)
    if len(accept) != num_rollouts:
        errors.append(
            "condition 3 (per-round acceptance floor): expected"
            f" {num_rollouts} rollout metrics lines, got {len(accept)}"
        )
    below = [round(value, 3) for value in accept if value < min_accept_length]
    if below:
        errors.append(
            f"condition 3 (per-round acceptance floor): rounds below"
            f" {min_accept_length}: {below}, all rounds:"
            f" {[round(value, 3) for value in accept]}"
        )
    if accept and accept[-1] < accept[0] * last_round_ratio:
        errors.append(
            f"condition 4 (no slow degradation): last round {accept[-1]:.3f}"
            f" < first round {accept[0]:.3f} x {last_round_ratio}"
        )

    summary = (
        f"draft_versions={versions} draft_sessions={len(sessions)}"
        f" spec_accept_length={[round(value, 3) for value in accept]}"
    )
    if errors:
        raise AssertionError(
            "post-run draft-update evidence check failed:\n  - "
            + "\n  - ".join(errors)
            + f"\n  evidence: {summary}\n  log: {log_path}"
        )
    print(f"[draft-update-check] OK: {summary}")


def _draft_train_events(lines):
    """Parsed payloads of the per-round "External Draft training" log lines."""
    events = []
    for line in lines:
        _, separator, payload = line.partition(_DRAFT_TRAIN_MARKER)
        if not separator:
            continue
        try:
            data = ast.literal_eval(payload.strip())
        except (ValueError, SyntaxError):
            continue
        if isinstance(data, dict) and "draft_version" in data:
            events.append(data)
    return events


def _post_events(lines):
    """(endpoint, status) pairs of the engine's POST access-log lines, in order."""
    events = []
    for line in lines:
        match = _POST_LINE_RE.search(line)
        if match:
            events.append((match.group(1), int(match.group(2))))
    return events


def _draft_publish_sessions(post_events):
    """Group draft publish chains: start_draft -> update -> finish, in order.

    A draft session is exactly three consecutive POSTs (see _DRAFT_CHAIN);
    a session cut short by a crash yields a short or mismatched group.
    """
    sessions = []
    index = 0
    while index < len(post_events):
        if post_events[index][0] == _DRAFT_CHAIN[0]:
            sessions.append(post_events[index : index + 3])
            index += 3
        else:
            index += 1
    return sessions


def _rollout_accept_lengths(lines):
    """Per-round rollout spec_accept_length values, in rollout order.

    Two loggers emit "perf N: {...}" lines (rollout_metrics from the
    RolloutManager, train_metric_utils from the train actor); only the
    rollout one carries the spec acceptance keys.
    """
    values = []
    for line in lines:
        if _SPEC_ACCEPT_KEY not in line:
            continue
        match = _ROLLOUT_PERF_RE.search(line)
        if match is None:
            continue
        try:
            data = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            continue
        if isinstance(data, dict) and _SPEC_ACCEPT_KEY in data:
            values.append(float(data[_SPEC_ACCEPT_KEY]))
    return values
