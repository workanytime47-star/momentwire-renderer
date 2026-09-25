from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import argparse
import hashlib
import os

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "CREATIVE_FINAL_QA_POLICY.json"
_FREEZE_RE = re.compile(r"lavfi\.freezedetect\.freeze_(start|duration|end):\s*([0-9.]+)")


def load_final_qa_policy() -> dict[str, Any]:
    data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("enabled"):
        raise ValueError("creative final QA policy is unavailable or disabled")
    return data


def _probe(path: Path) -> dict[str, Any]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=index,codec_type,start_time,duration,avg_frame_rate,r_frame_rate",
        "-show_entries", "format=start_time,duration", "-of", "json", str(path),
    ], text=True, timeout=30)
    return json.loads(raw)


def _ratio(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        a, b = value.split("/", 1)
        return float(a) / float(b) if float(b) else 0.0
    return float(value)


def _frame_timestamps(path: Path) -> list[float]:
    # Inspect packet presentation timestamps instead of asking ffprobe to decode
    # every frame. For H.264/H.265 with B-frames packet PTS may arrive in decode
    # order, so sort the presentation timestamps before measuring display gaps.
    # This preserves the timestamp-continuity gate while avoiding a full video
    # decode in ffprobe (the separate _decode_clean gate still decodes the file).
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path),
    ], text=True, timeout=45)
    out: list[float] = []
    for line in raw.splitlines():
        value = line.strip().split(",", 1)[0]
        if not value or value == "N/A":
            continue
        out.append(float(value))
    return sorted(set(out))


def _freeze_events(
    path: Path, threshold: float, viewport_aspect_ratio: str | None = None,
    noise_db: float = -50.0,
) -> list[dict[str, float]]:
    filters: list[str] = []
    if viewport_aspect_ratio:
        try:
            rw, rh = (float(x) for x in viewport_aspect_ratio.split(":", 1))
            if rw <= 0 or rh <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("invalid freeze-detection viewport aspect ratio")
        # preserve-frame-v2 puts the source at full width and centers it vertically.
        # Analyze motion in that content viewport so static safe-area bars cannot
        # dominate the freeze metric. The rest of final QA still checks the full file.
        filters.append(
            f"crop=iw:round(iw*{rh:.8f}/{rw:.8f}):0:(ih-round(iw*{rh:.8f}/{rw:.8f}))/2"
        )
    filters.extend(["scale=320:-2", f"freezedetect=n={float(noise_db):.1f}dB:d={threshold:.3f}"])
    proc = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "info", "-i", str(path),
        "-map", "0:v:0", "-vf", ",".join(filters),
        "-an", "-f", "null", "-",
    ], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise ValueError("final render freeze analysis failed")
    values = [(m.group(1), float(m.group(2))) for m in _FREEZE_RE.finditer(proc.stderr)]
    events: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for kind, value in values:
        if kind == "start":
            if current is not None:
                events.append(current)
            current = {"start": value}
        elif current is not None:
            current[kind] = value
            if kind == "end":
                events.append(current)
                current = None
    if current is not None:
        events.append(current)
    return events


def _decode_clean(path: Path) -> None:
    proc = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-",
    ], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or proc.stderr.strip():
        raise ValueError("final render decode/timestamp validation failed")


def _trimmed_source_freeze_events(
    path: Path, start_seconds: float, duration_seconds: float, threshold: float
) -> list[dict[str, float]]:
    filters = ["scale=320:-2", f"freezedetect=n=-50dB:d={threshold:.3f}"]
    proc = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "info",
        "-ss", str(max(0.0, float(start_seconds))),
        "-t", str(max(0.001, float(duration_seconds))),
        "-i", str(path), "-map", "0:v:0", "-vf", ",".join(filters),
        "-an", "-f", "null", "-",
    ], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise ValueError("selected source freeze analysis failed")
    values = [(m.group(1), float(m.group(2))) for m in _FREEZE_RE.finditer(proc.stderr)]
    events: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for kind, value in values:
        if kind == "start":
            if current is not None:
                events.append(current)
            current = {"start": value}
        elif current is not None:
            current[kind] = value
            if kind == "end":
                events.append(current)
                current = None
    if current is not None:
        events.append(current)
    return _close_open_freeze_events(events, float(duration_seconds))


def _close_open_freeze_events(events: list[dict[str, float]], duration_seconds: float) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for raw in events:
        event = dict(raw)
        start = float(event.get("start") or 0.0)
        if event.get("end") is None:
            end = max(start, float(duration_seconds))
            event["end"] = end
            event["duration"] = max(0.0, end - start)
        elif event.get("duration") is None:
            event["duration"] = max(0.0, float(event["end"]) - start)
        out.append(event)
    return out


def _freeze_event_bounds(event: dict[str, float]) -> tuple[float, float]:
    start = float(event.get("start") or 0.0)
    if event.get("end") is not None:
        end = float(event["end"])
    else:
        end = start + float(event.get("duration") or 0.0)
    return start, max(start, end)


def _freeze_matches_source(
    final_event: dict[str, float], source_events: list[dict[str, float]],
    tolerance_seconds: float, minimum_overlap_ratio: float,
) -> bool:
    fs, fe = _freeze_event_bounds(final_event)
    fd = max(0.001, fe - fs)
    for source_event in source_events:
        ss, se = _freeze_event_bounds(source_event)
        sd = max(0.001, se - ss)
        overlap = max(0.0, min(fe, se) - max(fs, ss))
        ratio = overlap / min(fd, sd)
        if abs(fs - ss) <= tolerance_seconds or ratio >= minimum_overlap_ratio:
            return True
    return False


def repair_actions_for_failures(failures: list[str]) -> list[dict[str, str]]:
    mapping = {
        "VIDEO_STREAM_START_LATE": ("REBASE_VIDEO_START", "Rebase video timestamps to start at zero before re-review."),
        "AUDIO_STREAM_START_LATE": ("REBASE_AUDIO_START", "Rebase audio timestamps to start at zero before re-review."),
        "FPS_BELOW_MINIMUM": ("RERENDER_MINIMUM_FPS", "Rerender at or above the minimum frame rate."),
        "AUDIO_VIDEO_DURATION_SKEW": ("RESYNC_AUDIO_VIDEO", "Rerender with aligned audio/video duration."),
        "VIDEO_TIMESTAMP_GAP": ("REENCODE_CONTINUOUS_TIMESTAMPS", "Reencode with continuous frame timestamps."),
        "OPENING_FREEZE_OR_STUTTER": ("REPAIR_OPENING", "Trim or rerender the opening; do not publish with a frozen/stuttering start."),
        "WHOLE_CLIP_FREEZE": ("RERENDER_MOTION_CONTINUITY", "Rerender the affected section and verify motion continuity."),
    }
    out=[]
    for failure in failures:
        action, note = mapping.get(str(failure), ("MANUAL_DIAGNOSIS", "Inspect the finished output and repair before publication."))
        out.append({"failure": str(failure), "action": action, "note": note})
    return out


def analyze_final_render(
    path: str | Path, *, source_path: str | Path | None = None,
    source_start_seconds: float = 0.0, source_duration_seconds: float | None = None,
) -> dict[str, Any]:
    p = Path(path).resolve()
    if not p.is_file() or p.stat().st_size <= 0:
        raise ValueError("final render file is missing")
    policy = load_final_qa_policy()["final_render"]
    probe = _probe(p)
    streams = probe.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise ValueError("final render must contain video and audio")
    video_start = float(video.get("start_time") or 0.0)
    audio_start = float(audio.get("start_time") or 0.0)
    video_duration = float(video.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    audio_duration = float(audio.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    fps = _ratio(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    failures: list[str] = []
    if abs(video_start) > float(policy["video_start_max_seconds"]):
        failures.append("VIDEO_STREAM_START_LATE")
    if abs(audio_start) > float(policy["audio_start_max_seconds"]):
        failures.append("AUDIO_STREAM_START_LATE")
    if fps < float(policy["minimum_fps"]):
        failures.append("FPS_BELOW_MINIMUM")
    if abs(video_duration - audio_duration) > float(policy["max_av_duration_skew_seconds"]):
        failures.append("AUDIO_VIDEO_DURATION_SKEW")
    timestamps = _frame_timestamps(p)
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    max_gap = max(gaps) if gaps else 0.0
    if max_gap > float(policy["max_video_pts_gap_seconds"]):
        failures.append("VIDEO_TIMESTAMP_GAP")
    _decode_clean(p)
    opening_window = float(policy["opening_window_seconds"])
    opening_threshold = float(policy["opening_freeze_reject_seconds"])
    whole_threshold = float(policy["whole_clip_freeze_reject_seconds"])
    freeze_events = _freeze_events(
        p,
        min(opening_threshold, whole_threshold),
        policy.get("content_aspect_ratio")
        if policy.get("freeze_detection_scope") == "CENTERED_CONTENT_VIEWPORT"
        else None,
    )
    freeze_events = _close_open_freeze_events(freeze_events, video_duration)
    source_freeze_events: list[dict[str, float]] = []
    inherited_freeze_events: list[dict[str, float]] = []
    actionable_freeze_events = list(freeze_events)
    if source_path is not None and policy.get("source_aware_freeze_comparison", False):
        source = Path(source_path).resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError("selected source file is missing")
        source_duration = float(source_duration_seconds if source_duration_seconds is not None else video_duration)
        source_freeze_events = _trimmed_source_freeze_events(
            source, float(source_start_seconds), source_duration,
            min(opening_threshold, whole_threshold),
        )
        tolerance = float(policy.get("source_freeze_match_tolerance_seconds", 0.2))
        overlap_ratio = float(policy.get("source_freeze_minimum_overlap_ratio", 0.6))
        actionable_freeze_events = []
        for event in freeze_events:
            if _freeze_matches_source(event, source_freeze_events, tolerance, overlap_ratio):
                inherited_freeze_events.append(event)
            else:
                actionable_freeze_events.append(event)
        if actionable_freeze_events and policy.get("confirm_renderer_added_freeze", False):
            viewport = (
                policy.get("content_aspect_ratio")
                if policy.get("freeze_detection_scope") == "CENTERED_CONTENT_VIEWPORT" else None
            )
            strict_events = _freeze_events(
                p, min(opening_threshold, whole_threshold), viewport,
                float(policy.get("renderer_added_freeze_confirmation_noise_db", -60.0)),
            )
            strict_events = _close_open_freeze_events(strict_events, video_duration)
            actionable_freeze_events = [
                event for event in actionable_freeze_events
                if _freeze_matches_source(event, strict_events, tolerance, overlap_ratio)
            ]
    for event in actionable_freeze_events:
        start = float(event.get("start") or 0.0)
        duration = float(event.get("duration") or max(0.0, video_duration - start))
        if start < opening_window and duration >= opening_threshold:
            failures.append("OPENING_FREEZE_OR_STUTTER")
            break
    for event in actionable_freeze_events:
        start = float(event.get("start") or 0.0)
        duration = float(event.get("duration") or max(0.0, video_duration - start))
        if duration >= whole_threshold:
            failures.append("WHOLE_CLIP_FREEZE")
            break
    failures = sorted(set(failures))
    return {
        "schema": "momentwire-final-render-qa/v1",
        "policy_contract": load_final_qa_policy()["contract_version"],
        "passed": not failures,
        "failures": failures,
        "repair_actions": repair_actions_for_failures(failures),
        "video_start_seconds": round(video_start, 6),
        "audio_start_seconds": round(audio_start, 6),
        "fps": round(fps, 4),
        "max_video_pts_gap_seconds": round(max_gap, 6),
        "av_duration_skew_seconds": round(abs(video_duration - audio_duration), 6),
        "freeze_events": freeze_events,
        "source_freeze_events": source_freeze_events,
        "inherited_freeze_events": inherited_freeze_events,
        "actionable_freeze_events": actionable_freeze_events,
        "source_aware_freeze_comparison": bool(source_path is not None and policy.get("source_aware_freeze_comparison", False)),
        "freeze_detection_scope": policy.get("freeze_detection_scope", "FULL_FRAME"),
        "content_aspect_ratio": policy.get("content_aspect_ratio"),
    }


def analyze_renderer_input(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    if not p.is_file() or p.stat().st_size <= 0:
        raise ValueError("renderer input file is missing")
    policy = load_final_qa_policy()["renderer_input"]
    probe = _probe(p)
    streams = probe.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise ValueError("renderer input must contain video and audio")
    video_start = float(video.get("start_time") or 0.0)
    audio_start = float(audio.get("start_time") or 0.0)
    failures: list[str] = []
    if abs(video_start) > float(policy["video_start_max_seconds"]):
        failures.append("VIDEO_STREAM_START_LATE")
    if abs(audio_start) > float(policy["audio_start_max_seconds"]):
        failures.append("AUDIO_STREAM_START_LATE")
    _decode_clean(p)
    freeze_events = _freeze_events(p, float(policy["opening_freeze_reject_seconds"]))
    opening = float(policy["opening_window_seconds"])
    for event in freeze_events:
        start = float(event.get("start") or 0.0)
        duration = float(event.get("duration") or 0.0)
        if start < opening and duration >= float(policy["opening_freeze_reject_seconds"]):
            failures.append("OPENING_FREEZE_OR_STUTTER")
            break
    failures = sorted(set(failures))
    return {
        "schema": "momentwire-renderer-input-qa/v1",
        "policy_contract": load_final_qa_policy()["contract_version"],
        "passed": not failures,
        "failures": failures,
        "video_start_seconds": round(video_start, 6),
        "audio_start_seconds": round(audio_start, 6),
        "freeze_events": freeze_events,
    }

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--source-start", type=float, default=0.0)
    ap.add_argument("--source-duration", type=float, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    final = Path(args.final).resolve()
    source = Path(args.source).resolve()
    qa = analyze_final_render(
        final,
        source_path=source,
        source_start_seconds=args.source_start,
        source_duration_seconds=args.source_duration,
    )
    payload = {
        "schema": "momentwire-github-strict-qa/v1",
        "policy_contract": load_final_qa_policy()["contract_version"],
        "passed": bool(qa.get("passed")),
        "failures": list(qa.get("failures") or []),
        "qa": qa,
        "final_sha256": _sha256(final),
        "final_bytes": final.stat().st_size,
        "source_sha256": _sha256(source),
        "source_bytes": source.stat().st_size,
        "github_repository": os.getenv("GITHUB_REPOSITORY", ""),
        "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "github_sha": os.getenv("GITHUB_SHA", ""),
        "execution_location": "GITHUB_ACTIONS_CLOUD",
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(45)


if __name__ == "__main__":
    _cli()
