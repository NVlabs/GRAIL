"""Compile agent-reviewed video intervals into frozen material contact pairs.

Run from the repo: python -m scripts.standalone.pack_fixed_contacts --help
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from grail.optimization.contact_review import compile_review, fixed_schema_for_review, review_sha256
from grail.optimization.fixed_contact import FixedContactLoss, _json_object, _reject_json_constant


def pack(binding_path, geometry_path, review_path, output_path):
    def read(path):
        return json.loads(
            Path(path).read_bytes(),
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )

    binding, review = read(binding_path), read(review_path)
    contacts = compile_review(review, binding)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Version the prediction instead of overwriting {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema=fixed_schema_for_review(review),
        binding=binding,
        video_review=review,
        contacts=contacts,
        prediction=dict(
            method="agent-reviewed video intervals",
            evidence=f"{Path(review_path).resolve()} sha256:{review_sha256(review)}",
        ),
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, allow_nan=False)
        with np.load(geometry_path, allow_pickle=False) as geometry:
            FixedContactLoss.from_file(
                temporary,
                binding=binding,
                object_vertices=geometry["object_vertices"],
                object_faces=geometry["object_faces"],
                human_faces=geometry["human_faces"],
                device="cpu",
                require_video_review=True,
            )
        # Exclusive creation also protects a prediction created during validation.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, help="init_data fixed_contact_binding.json")
    parser.add_argument("--geometry", required=True, help="init_data fixed_contact_geometry.npz")
    parser.add_argument(
        "--review",
        required=True,
        help="grail.contact_review.v2 per-hand vertex pairs and video intervals",
    )
    parser.add_argument("--output", required=True, help="Stage 2 contact.prediction_path")
    args = parser.parse_args()
    print(pack(args.binding, args.geometry, args.review, args.output))


if __name__ == "__main__":
    main()
