"""Run existing native-contact review dependencies in the model environment."""

import argparse

from .contracts import load_npz, read_json, write_json
from .review import review_candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = read_json(args.request)
    scene = load_npz(request["scene"])
    scene["_sha256"] = request["scene_sha256"]
    contact = read_json(request["contact_review"]) if request.get("contact_review") else None
    result = review_candidate(
        load_npz(request["parent"]),
        load_npz(request["candidate"]),
        load_npz(request["before"]),
        load_npz(request["after"]),
        scene,
        contact_spec=contact,
        evidence=request.get("evidence"),
    )
    write_json(request["output"], result)


if __name__ == "__main__":
    main()
