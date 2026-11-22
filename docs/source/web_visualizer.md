# Web Visualizer

<span id="browse-a-motion-library"></span>

You can compare a preview in your browser without running Isaac Sim. The web viewer pairs `robot/<name>.pkl` with `vis/<name>.mp4`. Pick the video that answers your question: a kinematic retargeting preview shows the commanded motion; a policy evaluation shows what the simulated robot did.

<span id="generate"></span>

## Gallery generation

<span id="build-a-gallery-for-the-stair-reference"></span>

Complete the [stairs tutorial](tutorials/stairs.md) through the terrain pack. In the `grail` container environment, use a fresh gallery directory. This keeps the original packed library intact and adds a copy of the G1 kinematic preview under the motion's own name.

```bash
LIB=/workspace/runs/guides/stairs-preview
mkdir -p "$LIB/vis"
ln -s /workspace/runs/stairs/terrain/robot "$LIB/robot"
cp /workspace/runs/stairs/retargeting.mp4 "$LIB/vis/stairs.mp4"
python -m grail.web_visualizer.generate_manifest --motion_lib "$LIB" \
  --output /workspace/runs/guides/browser --copy_videos
```

<span id="what-the-manifest-contains"></span>

In the standard image this took under one second on an i9-14900K. The manifest reported **1 motion, 241 frames, 1 video**. The portable site contains `index.html`, `main.js`, `style.css`, `manifest.json` and `vis/stairs.mp4`.

<span id="serve"></span>

## Local gallery server

<span id="open-it-on-your-workstation"></span>

If your container follows the installation guide, open a host terminal at the checkout root and serve the output through the host's loopback interface.

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory ../grail-demo/runs/guides/browser
```

<span id="what-the-viewer-does"></span>

Open `http://127.0.0.1:8000/` on that workstation and hover over the stairs card. Stop the server with Ctrl-C. If the card has no clip, look at `num_with_video` in `manifest.json` and check the MP4 stem against the robot PKL. If the page looks empty when double-clicked, open it through the HTTP server so the browser can fetch the manifest.

## Gallery manifest and sharing

<span id="inspect-the-manifest-and-share-the-gallery"></span>

The generator uses the robot PKL stem as the motion name. It reads the frame count from the trajectory, matches the video by stem under `vis/` and copies allowlisted metadata such as the object name and table pose, quaternion and size. A motion can have a card even when it has no video. Compare the motion and video counts before sharing.

By default the local gallery uses relative video symlinks. Choose `--copy_videos` as in the example above for a self-contained site. Serve the entire output including its manifest, JavaScript, CSS and media. The viewer searches motion names by case-insensitive substring, plays muted clips on hover and follows the browser’s color preference.
