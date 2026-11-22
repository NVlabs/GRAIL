"""Check destinations linked by a published GRAIL build against a proposed build."""
import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup


SITE = "https://nvlabs.github.io/GRAIL/"


def check_terminology(proposed, readme):
    banned = re.compile(r"\b(?:body[-‐‑– ]only|contactfirst)\b", re.IGNORECASE)
    errors = []
    for file in sorted(proposed.rglob("*.html")):
        article = BeautifulSoup(file.read_bytes(), "html.parser").find("article")
        if not article:
            continue
        for node in article.select("pre, code, script, style, .headerlink"):
            node.decompose()
        visible = article.get_text(" ", strip=True)
        visible += " " + " ".join(img.get("alt", "") for img in article.select("img"))
        if match := banned.search(visible):
            errors.append(f"{file.relative_to(proposed)}: revise prose term {match.group()!r}")
    if readme:
        sources = [readme, *sorted((readme.parent / "docs").glob("*.md"))]
        for source in sources:
            text = re.sub(r"^```[^\n]*\n.*?^```[^\n]*", "", source.read_text(), flags=re.M | re.S)
            text = re.sub(r"`[^`\n]+`", "", text)
            if match := banned.search(text):
                errors.append(f"{source.relative_to(readme.parent)}: revise prose term {match.group()!r}")
    return errors


def check(published, proposed, readme, proposed_readme=None):
    origins = defaultdict(set)
    promised_launchers = set()

    def record(url, origin):
        if not url.startswith(SITE):
            return None
        address = urlsplit(url[len(SITE):])
        path = unquote(address.path)
        if not path or path.endswith("/"):
            path += "index.html"
        if path.endswith(".html"):
            origins[path, unquote(address.fragment)].add(origin)
            return path
        return None

    for page in sorted(published.rglob("*.html")):
        relative = page.relative_to(published).as_posix()
        soup = BeautifulSoup(page.read_bytes(), "html.parser")
        for link in soup.select("a[href]"):
            path = record(urljoin(SITE + relative, link["href"]), relative)
            if (relative == "tutorials/stairs.html" and path and
                    "full-schedule launcher command" in link.get_text(" ", strip=True).lower()):
                promised_launchers.add(path)
    if readme:
        for url in re.findall(r"https://nvlabs\.github\.io/GRAIL/[^\s<>\"')]*", readme.read_text()):
            record(url, "README.md")

    missing = []
    pages = {}
    for (path, fragment), sources in sorted(origins.items()):
        target = proposed / path
        if path not in pages and target.is_file():
            pages[path] = BeautifulSoup(target.read_bytes(), "html.parser")
        if path not in pages or (fragment and not pages[path].find(id=fragment)):
            missing.append(f"{path}#{fragment} (from {', '.join(sorted(sources))})")
    for path in sorted(promised_launchers):
        example = 'bash scripts/tracking/terrain_finetune.sh "$CASE/tracking-full" "$CASE/terrain"'
        if path in pages and not any(example in code.get_text() for code in pages[path].select("pre")):
            missing.append(f"{path}: stairs tutorial links here for the full-schedule launcher")
    tracking = proposed / "tracking.html"
    if tracking.is_file():
        page = BeautifulSoup(tracking.read_bytes(), "html.parser")
        commands = "\n".join(node.get_text() for node in page.select("article pre"))
        for needle in ("gear_sonic/train_agent_trl.py", "gear_sonic/eval_agent_trl.py",
                       "gear_sonic/eval_contactfirst_trl.py", "accelerate launch", "--num_machines",
                       "manager/universal_token/hoi/pnp_table", "manager/universal_token/scene/terrain_tracking"):
            if needle not in commands:
                missing.append(f"tracking.html: missing tracking reference command or argument {needle}")
    missing.extend(check_terminology(proposed, proposed_readme))
    for item in missing:
        print(item)
    print(f"{len(origins)} distinct published destinations, {len(promised_launchers)} promised launcher checks, {len(missing)} failures")
    return not missing


def check_worker_inventory(proposed):
    """Exercise the published observation layout with the installed GRAIL CPU reader."""
    from grail.agentic_kinematic.hoi_stages import bind, load_case
    from grail.pipelines.recon_4dhoi import _origin_id

    soup = BeautifulSoup((proposed / "recon_4dhoi.html").read_bytes(), "html.parser")
    recipe, = [node.get_text() for node in soup.select("article pre")
               if "check_capture.py" in node.get_text() and "SOURCE=" in node.get_text()]
    settings = dict(re.findall(r"^(SOURCE|ROOT|VIDEO)=(\S+)$", recipe, re.MULTILINE))
    run_mount = Path("/workspace/runs")
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / Path(settings["SOURCE"]).relative_to(run_mount)
        case_dir = source.parent
        worker = Path(tmp) / Path(settings["ROOT"]).relative_to(run_mount)
        video = settings["VIDEO"]
        if "end" in video or _origin_id(video) != video:
            raise ValueError(f"Documented worker sequence does not preserve its calibration ID: {video}")
        products = worker / "generation"
        paths = {
            "source_video": source / "videos_kling" / f"{video}-start-end.mp4",
            "object_mesh": source / "mesh" / Path(video).parent / "model.obj",
            "render_config": source / "foundation_pose" / video / "first_frame_pose.pickle",
            "soma03/human_motion": products / "hmr_gem_x_native77" / f"{video}.npz",
            "object_poses": products / "object_pose_mose3_fp" / video / "pose_estimation_output/poses_in_cam.pkl",
            "mask": products / "4dhoi_recon_soma03_cache/masks" / f"{video}.npz",
            "depth": products / "4dhoi_recon_soma03_cache/depth_moge3_vitg" / f"{video}.pt",
            "depth_metadata": products / "4dhoi_recon_soma03_cache/depth_moge3_vitg" / f"{video}.pt.json",
        }
        files = {}
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())  # load_case verifies locations, sizes and hashes.
            files[name] = {**bind(path), "path": os.path.relpath(path, case_dir)}
        inventory = {"schema": "grail.cached_fitting_inputs.v1", "fps": 24, "frame_count": 2,
                     "placement": {"contact_frame": 0, "object_first_motion_frame": 1}, "files": files}
        case_path = case_dir / "case.json"
        case_path.write_text(json.dumps({"schema_version": 1, "backend": "soma03",
                                        "human_model": {"body_model": "soma03"}, "inventory": "inventory.json",
                                        "optimization_config": "fit.yaml"}))
        (case_dir / "fit.yaml").write_text("{}")
        inventory_path = case_dir / "inventory.json"
        inventory_path.write_text(json.dumps(inventory))
        load_case(case_path, require_contacts=False)

        # Reproduce the sibling-directory failure using the same bytes and hash.
        sibling = case_dir.with_name(case_dir.name + "-observations") / "mask.npz"
        sibling.parent.mkdir(exist_ok=True)
        sibling.write_bytes(paths["mask"].read_bytes())
        files["mask"]["path"] = os.path.relpath(sibling, case_dir)
        inventory_path.write_text(json.dumps(inventory))
        try:
            load_case(case_path, require_contacts=False)
        except ValueError as exc:
            if str(exc) != "Input record escapes inventory directory":
                raise
        else:
            raise AssertionError("The fitter accepted a worker record outside the inventory directory")
    print("Stage 1 inventory accepted all eight documented input paths and rejected the sibling record.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("published", type=Path)
    parser.add_argument("proposed", type=Path)
    parser.add_argument("--readme", type=Path, help="published README.md to include")
    parser.add_argument("--proposed-readme", type=Path, help="new README to check against the paper terminology")
    parser.add_argument("--check-worker-inventory", action="store_true",
                        help="also load the documented Stage 1 inventory on CPU in the GRAIL environment")
    args = parser.parse_args()
    if not all((d / "index.html").is_file() for d in (args.published, args.proposed)):
        parser.error("Both inputs must be built site directories containing index.html")
    if args.check_worker_inventory:
        check_worker_inventory(args.proposed)
    raise SystemExit(0 if check(args.published, args.proposed, args.readme, args.proposed_readme) else 1)
