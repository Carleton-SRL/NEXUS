#!/usr/bin/env python3
"""
frames_to_avi.py

Scans a directory for image files with common prefixes and sequential
numeric suffixes (e.g., left_0000.png, left_0001.png, frame_0000.png),
groups them by prefix, and encodes each group into an AVI video.

Usage:
    python frames_to_avi.py /path/to/frames [--fps 30] [--output /path/to/output]
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import cv2

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Matches a prefix (ending with _ or -) followed by a numeric suffix.
# e.g.  "left_0001.png"  -> prefix="left_", index=1
#        "frame-042.jpg" -> prefix="frame-", index=42
FRAME_PATTERN = re.compile(
    r"^(?P<prefix>.+?[_\-])(?P<index>\d+)$"
)


def discover_groups(directory: Path) -> dict[str, list[tuple[int, Path]]]:
    """Return {prefix: [(index, filepath), ...]} for all matching images."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        match = FRAME_PATTERN.match(path.stem)
        if not match:
            continue

        prefix = match.group("prefix")
        index = int(match.group("index"))
        groups[prefix].append((index, path))

    # Sort each group by frame index
    for prefix in groups:
        groups[prefix].sort(key=lambda t: t[0])

    return dict(groups)


def encode_video(
    frames: list[tuple[int, Path]],
    output_path: Path,
    fps: float,
) -> None:
    """Write a list of (index, filepath) frames to an AVI file."""
    first_img = cv2.imread(str(frames[0][1]))
    if first_img is None:
        raise RuntimeError(f"Cannot read image: {frames[0][1]}")

    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for idx, path in frames:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  WARNING: skipping unreadable frame {path.name}")
            continue
        # Resize if dimensions don't match the first frame
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine sequentially numbered image frames into AVI videos."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the image frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the output video (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the AVI files (default: same as input).",
    )
    args = parser.parse_args()

    src = args.directory.resolve()
    if not src.is_dir():
        raise SystemExit(f"Error: '{src}' is not a directory.")

    dst = (args.output or src).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    groups = discover_groups(src)
    if not groups:
        print("No frame sequences found.")
        return

    for prefix, frames in groups.items():
        label = prefix.rstrip("_-")
        out_file = dst / f"{label}.avi"
        n = len(frames)
        first_idx, last_idx = frames[0][0], frames[-1][0]
        print(f"[{label}]  {n} frames  (indices {first_idx}–{last_idx})  ->  {out_file.name}")
        encode_video(frames, out_file, args.fps)

    print(f"\nDone — wrote {len(groups)} video(s) to {dst}")


if __name__ == "__main__":
    main()
