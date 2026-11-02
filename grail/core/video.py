import os
import shutil
from glob import glob

import cv2
import imageio
import numpy as np
import requests
from tqdm import tqdm

DEFAULT_MOTION_FPS = 24.0


def normalize_motion_fps(value, default=DEFAULT_MOTION_FPS):
    """Return a valid motion FPS, falling back to the project default."""
    try:
        fps = float(value)
    except (TypeError, ValueError):
        fps = float(default)
    if not np.isfinite(fps) or fps <= 0:
        fps = float(default)
    return fps


def get_motion_fps(motion_data, default=DEFAULT_MOTION_FPS):
    """Read FPS from HOI or motion data using the canonical 24-FPS fallback."""
    if not isinstance(motion_data, dict):
        return float(default)
    human_data = motion_data.get("human_data", motion_data)
    if not isinstance(human_data, dict):
        return float(default)
    value = human_data.get("mocap_frame_rate", human_data.get("fps", default))
    return normalize_motion_fps(value, default=default)


def concat_videos(input_video, result_video, topview_video, output_path):
    """
    Concatenate input, result, and top-view videos into a single comparison video.
    Vertical input -> horizontal stack; horizontal input -> vertical stack.
    Uses imageio for H.264 output compatible with VSCode.
    """
    video_paths = [v for v in [input_video, result_video, topview_video] if os.path.exists(v)]
    if len(video_paths) < 2:
        return False

    readers = []
    for vp in video_paths:
        try:
            readers.append(imageio.get_reader(vp))
        except Exception:
            for r in readers:
                r.close()
            return False

    meta = readers[0].get_meta_data()
    fps = normalize_motion_fps(meta.get("fps"))
    ref_h, ref_w = readers[0].get_data(0).shape[:2]
    vertical_input = ref_h > ref_w

    min_frames = min(r.count_frames() for r in readers)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for i in range(min_frames):
            frames = [r.get_data(i) for r in readers]
            resized = []
            if vertical_input:
                for frame in frames:
                    fh, fw = frame.shape[:2]
                    scale = ref_h / fh
                    new_w = int(fw * scale) // 2 * 2
                    resized.append(cv2.resize(frame, (new_w, ref_h)))
                combined = np.hstack(resized)
            else:
                for frame in frames:
                    fh, fw = frame.shape[:2]
                    scale = ref_w / fw
                    new_h = int(fh * scale) // 2 * 2
                    resized.append(cv2.resize(frame, (ref_w, new_h)))
                combined = np.vstack(resized)
            writer.append_data(combined)

    for r in readers:
        r.close()
    return os.path.exists(output_path)


def save_images_to_video(images, output_video_path, fps=DEFAULT_MOTION_FPS, desc="Saving video"):
    """
    Save a sequence of images to a video file

    Args:
        images (list or generator): Sequence of images as numpy arrays (H, W) or (H, W, C)
        output_video_path (str): Path for output video
        fps (int): Frames per second (default: 24)
        desc (str): Description for progress bar
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Convert to list if it's a generator to get length
    if not isinstance(images, (list, tuple)):
        images = list(images)

    if not images:
        print("No images provided to save")
        return

    print(f"Saving {len(images)} images to video: {output_video_path}")

    # Create video writer
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for image in tqdm(images, desc=desc):
            # Convert tensor to numpy if needed
            if hasattr(image, "cpu"):
                image = image.cpu().numpy()

            # Squeeze extra dimensions
            if len(image.shape) > 2:
                image = image.squeeze()

            # Handle different image formats
            if len(image.shape) == 2:
                # Single channel (H, W) - convert to RGB for video
                if image.dtype == bool or image.max() <= 1.0:
                    # Binary mask or normalized values
                    image = (image * 255).astype(np.uint8)
                elif image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                # Convert grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                # Multi-channel (H, W, C)
                if image.dtype == bool or image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                elif image.dtype != np.uint8:
                    image = image.astype(np.uint8)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            writer.append_data(image)

    print(f"Video saved: {output_video_path}")


def compile_images_to_video(
    image_dir, output_video_path, fps=DEFAULT_MOTION_FPS, image_pattern="*.jpg"
):
    """
    Compile rendered images into a video file

    Args:
        image_dir (str): Directory containing rendered images
        output_video_path (str): Path for output video
        fps (int): Frames per second
    """
    image_files = sorted(glob(os.path.join(image_dir, image_pattern)))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Compiling {len(image_files)} images into video: {output_video_path}")

    # Create video writer
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for image_file in tqdm(image_files, desc="Compiling video"):
            image = imageio.imread(image_file)
            writer.append_data(image)

    print(f"Video saved: {output_video_path}")


def get_video_fps_and_frame_count(video_path, default_fps=DEFAULT_MOTION_FPS):
    """
    Get video FPS and total frame count using OpenCV.

    Args:
        video_path (str): Path to video file

    Returns:
        tuple: (fps, frame_count)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = normalize_motion_fps(cap.get(cv2.CAP_PROP_FPS), default=default_fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, frame_count


def resize_video(input_path, target_width, target_height, output_path=None):
    """Resize a video to target dimensions.

    When *output_path* is ``None`` the file is resized **in-place** (via a
    temporary file).  If already at the target size the file is returned
    unchanged.

    Args:
        input_path: Path to the video file.
        target_width: Target width in pixels.
        target_height: Target height in pixels.
        output_path: Optional separate output path.  Defaults to overwriting
            *input_path*.

    Returns:
        Path to the (possibly resized) video file.
    """
    reader = imageio.get_reader(input_path, format="ffmpeg")
    meta = reader.get_meta_data()
    first = reader.get_data(0)
    cur_h, cur_w = first.shape[:2]
    fps = normalize_motion_fps(meta.get("fps"))

    if cur_w == target_width and cur_h == target_height:
        reader.close()
        return input_path

    from PIL import Image as _PILImage

    dest = output_path or input_path
    tmp = f"{os.path.splitext(dest)[0]}_tmp{os.path.splitext(dest)[1]}"

    writer = imageio.get_writer(
        tmp,
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    for frame in reader:
        img = _PILImage.fromarray(frame).resize((target_width, target_height), _PILImage.LANCZOS)
        writer.append_data(np.array(img))
    reader.close()
    writer.close()

    os.replace(tmp, dest)
    return dest


def crop_video_from_start(input_video_path, output_video_path, start_seconds):
    """
    Crop video starting from start_seconds to the end using moviepy.

    Args:
        input_video_path (str): Path to input video
        output_video_path (str): Path to output cropped video
        start_seconds (float): Number of seconds to crop from the beginning
    """
    try:
        from moviepy.editor import VideoFileClip

        # Load video
        clip = VideoFileClip(input_video_path)

        # Crop from start_seconds to end
        cropped_clip = clip.subclip(start_seconds)

        # Write cropped video
        cropped_clip.write_videofile(
            output_video_path, codec="libx264", preset="slow", bitrate="5000k", audio=True
        )

        # Clean up
        clip.close()
        cropped_clip.close()

        print(f"Cropped video saved to: {output_video_path}")

    except Exception as e:
        print(f"Error cropping video {input_video_path}: {e}")
        raise


def trim_video(input_video_path, output_video_path, start_seconds):
    """
    Trim video starting from start_seconds to the end (alias for crop_video_from_start)

    Args:
        input_video_path (str): Path to input video
        output_video_path (str): Path to output trimmed video
        start_seconds (float): Number of seconds to trim from the beginning
    """
    crop_video_from_start(input_video_path, output_video_path, start_seconds)


def extract_frames_from_video(video_file, output_dir, frame_prefix="", image_format="jpg"):
    """
    Extract all frames from video and save as JPG images

    Args:
        video_file (str): Path to input video file
        output_dir (str): Directory to save extracted frames
        frame_prefix (str): Prefix for frame filenames (optional)

    Returns:
        int: Number of frames extracted
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        return 0

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting {total_frames} frames from {video_file}")
    print(f"Saving frames to: {output_dir}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Format frame filename with zero-padding
        if frame_prefix:
            frame_filename = f"{frame_prefix}{frame_count:06d}.{image_format}"
        else:
            frame_filename = f"{frame_count:06d}.{image_format}"

        frame_path = os.path.join(output_dir, frame_filename)

        # Save frame as JPG
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Successfully extracted {frame_count} frames")

    return frame_count


def extract_frames_from_cropped_video(video_file, output_dir, start_frame=0):
    """
    Extract frames from cropped video and save as JPG images.

    Args:
        video_file (str): Path to cropped video file
        output_dir (str): Directory to save extracted frames
        start_frame (int): Starting frame number for naming (usually frames skipped from original)

    Returns:
        int: Number of frames extracted
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        return 0

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting {total_frames} frames from cropped video")
    print(f"Saving frames to: {output_dir}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Frame filename with zero-padding (continuing from where original left off)
        frame_filename = f"{(start_frame + frame_count):06d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)

        # Save frame as JPG
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Successfully extracted {frame_count} frames")

    return frame_count


def download_video(url, output_dir, filename):
    """Download a video from a URL.

    Args:
        url: Video URL.
        output_dir: Directory to save into.
        filename: Output filename (`.mp4` appended if missing).

    Returns:
        Path to the downloaded file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not filename.endswith(".mp4"):
        filename += ".mp4"
    output_path = os.path.join(output_dir, filename)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return output_path


def is_valid_video(video_path):
    """Return True if *video_path* can be opened and has at least one frame."""
    try:
        reader = imageio.get_reader(video_path, format="ffmpeg")
        meta = reader.get_meta_data()
        reader.close()
        return meta.get("nframes", 0) != 0
    except Exception:
        return False


def extract_last_frame(video_path, output_path=None):
    """Extract the last frame from a video and save it as PNG.

    Args:
        video_path: Path to the video.
        output_path: Where to save; defaults to ``<video>_lastframe.png``.

    Returns:
        Path to the saved frame image.
    """
    from PIL import Image as _PILImage

    reader = imageio.get_reader(video_path, format="ffmpeg")
    last_frame = None
    for frame in reader:
        last_frame = frame
    reader.close()

    if last_frame is None:
        raise ValueError(f"No frames in {video_path}")

    if output_path is None:
        output_path = f"{os.path.splitext(video_path)[0]}_lastframe.png"

    _PILImage.fromarray(last_frame).save(output_path)
    return output_path


def concatenate_videos(video_paths, output_path):
    """Concatenate multiple videos into one file.

    Frames are resized to match the first video's dimensions when needed.

    Args:
        video_paths: Ordered list of video file paths.
        output_path: Destination for the concatenated video.

    Returns:
        *output_path*.
    """
    if not video_paths:
        raise ValueError("No video paths provided")

    from PIL import Image as _PILImage

    first_reader = imageio.get_reader(video_paths[0], format="ffmpeg")
    fps = normalize_motion_fps(first_reader.get_meta_data().get("fps"))
    first_frame = first_reader.get_data(0)
    height, width = first_frame.shape[:2]
    first_reader.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    for vp in video_paths:
        reader = imageio.get_reader(vp, format="ffmpeg")
        for frame in reader:
            if frame.shape[:2] != (height, width):
                frame = np.array(
                    _PILImage.fromarray(frame).resize((width, height), _PILImage.LANCZOS)
                )
            writer.append_data(frame)
        reader.close()

    writer.close()
    return output_path


def crop_visualization_images(vis_dir, frames_to_remove, output_vis_dir):
    """
    Crop visualization images by removing the first N frames.

    Args:
        vis_dir (str): Directory containing original visualization images
        frames_to_remove (int): Number of frames to remove from the beginning
        output_vis_dir (str): Directory to save cropped visualization images
    """
    if not os.path.exists(vis_dir):
        print(f"Warning: Visualization directory not found: {vis_dir}")
        return

    # Get all JPG files and sort them
    vis_files = sorted(glob(os.path.join(vis_dir, "*.jpg")))

    if not vis_files:
        print(f"No visualization images found in {vis_dir}")
        return

    print(f"Found {len(vis_files)} visualization images")

    if frames_to_remove >= len(vis_files):
        print(
            f"Warning: Trying to remove {frames_to_remove} images but only {len(vis_files)} available"
        )
        return

    # Create output directory
    os.makedirs(output_vis_dir, exist_ok=True)

    # Copy images starting from frames_to_remove
    cropped_files = vis_files[frames_to_remove:]

    for i, src_file in enumerate(cropped_files):
        # Create new filename with sequential numbering
        src_filename = os.path.basename(src_file)

        # If the original filename has a specific format, preserve the extension but renumber
        if src_filename.endswith(".jpg"):
            new_filename = f"{i:06d}.jpg"
        else:
            new_filename = f"{i:06d}_{src_filename}"

        dst_file = os.path.join(output_vis_dir, new_filename)

        # Copy the file
        shutil.copy2(src_file, dst_file)

    print(f"Copied {len(cropped_files)} visualization images to {output_vis_dir}")
