"""Portable review pages and explicit source-timeline checks for rendered videos."""

import html
import json
import math
import shutil
import subprocess
from pathlib import Path

from .manifest import selected, verify, write_json


def probe_video(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,avg_frame_rate,width,height:frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    return {
        "fps": numerator / denominator,
        "frame_count": int(stream["nb_read_frames"]),
        "width": stream["width"],
        "height": stream["height"],
        "timestamps": [float(frame["best_effort_timestamp_time"]) for frame in data["frames"]],
    }


def check_video(path, fps, frame_count):
    result = probe_video(path)
    if result["frame_count"] != frame_count or abs(result["fps"] - fps) > 1e-4:
        raise ValueError(f"Video does not match the source timeline: {path}: {result}")
    timestamps = result["timestamps"]
    if len(timestamps) != frame_count or any(
        not math.isfinite(t) or abs(t - timestamps[0] - index / fps) > 1e-5
        for index, t in enumerate(timestamps)
    ):
        raise ValueError(f"Video frame timestamps do not match the uniform source timeline: {path}")
    return result


def build_page(manifest, output_dir):
    """Copy playable media; preserve every case and explicitly disclose missing pairing."""
    verify(manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "media").mkdir()
    cards, coverage = [], []
    copied = {}

    def video(binding, case):
        metadata = check_video(binding["path"], case["fps"], case["frame_count"])
        key = binding["sha256"]
        if key not in copied:
            suffix = Path(binding["path"]).suffix.lower()
            if suffix not in (".mp4", ".mov"):
                raise ValueError("Review pages require playable MP4/MOV files")
            destination = f"media/{key}{suffix}"
            shutil.copy2(binding["path"], output / destination)
            copied[key] = destination
        return f'<video controls preload="none" src="{copied[key]}"></video>', metadata

    for case in manifest["cases"]:
        source, _ = video(case["inputs"]["source_video"], case)
        cells = [f"<div><h3>Source video</h3>{source}</div>"]
        row = {"id": case["id"], "selected": dict(case["selected"]), "missing": []}
        for stage, title in (("human_object", "Human / object"), ("robot", "Native G1")):
            candidate = selected(case, stage)
            if candidate is None:
                row["missing"].append(stage + "_rebase")
                candidate = case["original"][stage]
                status = "Original paired reference; rebase for the selected human is missing"
            else:
                status = candidate["status"]
            notes = "; ".join(candidate.get("limitations", []))
            body = "<p>Comparison video unavailable.</p>"
            if "comparison" in candidate:
                body, _ = video(candidate["comparison"], case)
            else:
                row["missing"].append(stage + "_comparison")
            cells.append(
                f"<div><h3>{title}: {html.escape(status)}</h3>{body}<p>{html.escape(notes)}</p></div>"
            )
        cards.append(
            f'<article><h2>{html.escape(case["id"])}</h2><div class="columns">'
            + "".join(cells)
            + "</div></article>"
        )
        coverage.append(row)
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Agentic Retargeter review</title>
<style>body{background:#151719;color:#eee;font:16px system-ui;margin:2rem}article{border-top:1px solid #555;margin-top:2rem}.columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}video{width:100%;max-height:600px}p{line-height:1.5}h3{font-size:1rem}</style>
<h1>Agentic Retargeter review</h1>"""
    page += f"<p>{len(coverage)} / {manifest['expected_cases']} cases. Numerical feasibility and scoped visual improvements do not establish physical validity. Unresolved cases retain explicit original references.</p>"
    page += "".join(cards) + "</html>"
    (output / "index.html").write_text(page)
    write_json(output / "coverage.json", coverage)
    write_json(output / "manifest.json", manifest)
    return {
        "cases": len(coverage),
        "copied_videos": len(copied),
        "cases_with_missing_coverage": sum(bool(row["missing"]) for row in coverage),
    }
