"""Thin CLI dispatch for local video perception and grasp planning."""
import argparse
import json
from pathlib import Path


def add_cli(subparsers):
    parser = subparsers.add_parser("video", help="Local video grasp evidence and keyframes before editing")
    sub = parser.add_subparsers(dest="video_command", required=True)
    for name in ("extract", "analyze", "plan"):
        command = sub.add_parser(name)
        command.add_argument("--output", required=True, help="New output directory")
        if name in ("extract", "analyze"):
            command.add_argument("--video", required=True)
            command.add_argument("--object-prompt", required=True)
            command.add_argument("--detector-dir", required=True)
            command.add_argument("--segmenter-dir", required=True)
            command.add_argument("--hand-model", required=True)
            command.add_argument("--device", default="cpu")
            command.add_argument("--stride", type=int, default=1)
        if name == "plan":
            command.add_argument("--observations", required=True)
        if name in ("plan", "analyze"):
            command.add_argument("--no-render", action="store_true")
            command.add_argument("--stable-seconds", type=float, default=.12)
    return parser


def run_cli(args):
    from .video_grasps import Settings, run

    output = Path(args.output).expanduser().resolve()
    if args.video_command in ("plan", "analyze"):
        Settings(stable_seconds=args.stable_seconds).validate()
    if args.video_command == "plan":
        return run(args.observations, output, Settings(stable_seconds=args.stable_seconds), render=not args.no_render)
    from .video_perception import extract_video_observations

    if args.video_command == "analyze":
        output.mkdir(parents=True, exist_ok=False)
        observations_dir = output / "observations"
    else:
        observations_dir = output
    evidence = extract_video_observations(
        video=args.video, object_prompt=args.object_prompt, output=observations_dir,
        detector_dir=args.detector_dir, segmenter_dir=args.segmenter_dir,
        hand_model=args.hand_model, device=args.device, stride=args.stride,
    )
    if args.video_command == "extract":
        return {"observations": str(evidence), "status": "UNREVIEWED"}
    return run(observations_dir / "observations.json", output / "plan",
               Settings(stable_seconds=args.stable_seconds), render=not args.no_render)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_cli(sub)
    args = parser.parse_args(["video", *(argv if argv is not None else __import__("sys").argv[1:])])
    try:
        result = run_cli(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Video analysis error: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
