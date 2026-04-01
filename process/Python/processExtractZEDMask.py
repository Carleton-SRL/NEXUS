"""
SAM 2 Manual Segmentation Pipeline
------------------------------------
1. Displays the prompt frame so the user can click a polygon around the satellite
2. Converts the polygon to a binary mask prompt
3. Feeds the mask to SAM 2 video predictor to segment the entire sequence

Usage:
    python processExtractZEDMask.py --input <image_folder> [options]

Interactive polygon controls:
    Left-click  : add a point
    Right-click : remove the last point
    Enter/Space : confirm selection and run SAM 2
    ESC         : quit without processing

Requires:
    - PyTorch with ROCm (or CUDA)
    - SAM 2 (pip install -e . from the sam2 repo)
    - OpenCV (pip install opencv-python)

Sample Script Call:

    HSA_OVERRIDE_GFX_VERSION=11.0.2 python processExtractZEDMask.py --input "/home/alexandercrain/Dropbox/Graduate Documents/Doctor of Philosophy/Thesis Research/Datasets/SPOT/RAWs MK5/DXL-ROT-DATASET/DXL-ROT-NOM-MK5-ZED/"

    Using the above call will generate a folder called sam2_output by default, and this folder will be saved in the same folder as the target ZED camera footage.
"""


import cv2
import numpy as np
import argparse
import os
import re
import sys
from pathlib import Path

IMAGE_EXTS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# ─── Frame I/O ───────────────────────────────────────────────────────

def detect_prefixes(folder):
    """
    Scan folder for image files whose stems end in a run of digits
    (e.g. left_0000.png, aba_001.jpg, aer_0001.png).
    Returns a dict mapping prefix -> sorted list of file paths.
    """
    groups = {}
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lstrip(".").lower() not in IMAGE_EXTS:
            continue
        m = re.match(r'^(.*?)(\d+)$', path.stem)
        if not m:
            continue
        prefix = m.group(1)
        groups.setdefault(prefix, []).append(str(path))

    # Sort each group by the numeric suffix
    for prefix in groups:
        groups[prefix].sort(key=lambda p: int(re.search(r'(\d+)$', Path(p).stem).group(1)))

    return groups


def load_sorted_frames(folder, prefix=None):
    """
    Find image frames in folder whose filenames end in digits.
    If prefix is given, use only that prefix.
    If omitted and exactly one prefix exists, use it automatically.
    If multiple prefixes exist and none is specified, print them and return [].
    """
    groups = detect_prefixes(folder)

    if not groups:
        print("WARNING: No numbered image files found in folder.")
        return [], None

    if prefix is not None:
        if prefix not in groups:
            print(f"ERROR: Prefix '{prefix}' not found. Available: {list(groups.keys())}")
            return [], None
        paths = groups[prefix]
        print(f"Found {len(paths)} frames with prefix '{prefix}'")
        return paths, prefix

    if len(groups) == 1:
        chosen = next(iter(groups))
        paths = groups[chosen]
        print(f"Found {len(paths)} frames with prefix '{chosen}'")
        return paths, chosen

    # Multiple prefixes — ask the user to pick
    print("Multiple frame prefixes found in folder:")
    for p, files in groups.items():
        print(f"  '{p}'  ({len(files)} frames)")
    print("Use --prefix to select one.")
    return [], None


def prepare_video_dir(frame_paths, video_dir):
    """
    SAM 2 video predictor expects a folder of sequentially named JPEG frames.
    Symlink (or copy) our frames into that format.
    """
    os.makedirs(video_dir, exist_ok=True)
    for i, src in enumerate(frame_paths):
        dst = os.path.join(video_dir, f"{i:05d}.jpg")
        if os.path.exists(dst):
            os.remove(dst)

        # Read and re-save as JPEG to guarantee format
        img = cv2.imread(src)
        cv2.imwrite(dst, img)

    return video_dir


# ─── Interactive Polygon Selection ───────────────────────────────────

class PolygonSelector:
    """Interactive OpenCV window for drawing a polygon by clicking points."""

    POINT_RADIUS = 5
    POINT_COLOR  = (0, 255, 0)      # green dots
    LINE_COLOR   = (0, 255, 0)      # green lines
    FILL_COLOR   = (0, 255, 0)      # green fill (semi-transparent)
    FILL_ALPHA   = 0.25
    CLOSE_COLOR  = (0, 200, 255)    # orange closing edge

    def __init__(self, image):
        self.base = image.copy()
        self.points = []
        self.done = False
        self.cancelled = False

    def _render(self):
        canvas = self.base.copy()
        pts = self.points

        if len(pts) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], self.FILL_COLOR)
            cv2.addWeighted(overlay, self.FILL_ALPHA, canvas, 1 - self.FILL_ALPHA, 0, canvas)

        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], self.LINE_COLOR, 2)

        if len(pts) >= 2:
            # Draw closing edge in a different colour to hint at the loop
            cv2.line(canvas, pts[-1], pts[0], self.CLOSE_COLOR, 1)

        for p in pts:
            cv2.circle(canvas, p, self.POINT_RADIUS, self.POINT_COLOR, -1)

        n = len(pts)
        hint = (
            f"Points: {n}  |  Left-click: add  Right-click: undo  "
            f"Enter/Space: confirm  ESC: quit"
        )
        cv2.putText(canvas, hint, (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, hint, (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    def _mouse_cb(self, event, x, y, _flags, _param):
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.imshow("Select satellite region", self._render())
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
            cv2.imshow("Select satellite region", self._render())

    def run(self):
        """Block until the user confirms or cancels. Returns polygon points or None."""
        win = "Select satellite region"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._mouse_cb)
        cv2.imshow(win, self._render())

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):  # Enter or Space
                if len(self.points) < 3:
                    print("  Need at least 3 points to form a polygon. Keep clicking.")
                else:
                    self.done = True
                    break
            elif key == 27:  # ESC
                self.cancelled = True
                break

        cv2.destroyWindow(win)
        return None if self.cancelled else self.points


def get_manual_polygon_mask(frame_path):
    """
    Show an interactive window on frame_path so the user can draw a polygon.
    Returns (polygon_points, binary_mask) where mask dtype=uint8 with values 0/255,
    or (None, None) if the user cancelled.
    """
    img = cv2.imread(frame_path)
    if img is None:
        print(f"ERROR: Cannot read frame: {frame_path}")
        return None, None

    h, w = img.shape[:2]
    print("\nStep 1: Manual region selection")
    print("  A window will open showing the prompt frame.")
    print("  Left-click to place polygon points around the satellite.")
    print("  Right-click to remove the last point.")
    print("  Press Enter or Space when done, ESC to quit.\n")

    selector = PolygonSelector(img)
    points = selector.run()

    if points is None:
        print("  Selection cancelled.")
        return None, None

    print(f"  Polygon confirmed with {len(points)} points.")

    # Rasterise polygon to a binary mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return points, mask


# ─── SAM 2 Video Segmentation ───────────────────────────────────────

def run_sam2_segmentation(video_dir, polygon_mask, num_frames, checkpoint, config, prompt_frame=0):
    """
    Use SAM 2 video predictor to segment the object across all frames,
    seeded with a polygon mask prompt at prompt_frame.
    """
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  SAM 2 device: {device}")
    if device == "cpu":
        print("  WARNING: Running on CPU will be very slow.")

    # Build predictor
    predictor = build_sam2_video_predictor(config, checkpoint, device=device)

    # Initialize video state
    inference_state = predictor.init_state(video_path=video_dir)

    # Add mask prompt on prompt_frame
    prompt_frame_idx = min(prompt_frame, num_frames - 1)
    frame0 = cv2.imread(os.path.join(video_dir, f"{prompt_frame_idx:05d}.jpg"))
    h, w = frame0.shape[:2]
    mask_resized = cv2.resize(polygon_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_input = (mask_resized > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_input).to(device)

    _, obj_ids, mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=prompt_frame_idx,
        obj_id=1,
        mask=mask_tensor,
    )
    print(f"  Prompted on frame {prompt_frame_idx} with manual polygon mask")

    # Propagate through entire video
    print(f"  Propagating masks across {num_frames} frames...")
    masks_per_frame = {}

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        # mask_logits shape: (num_objects, 1, H, W)
        # Threshold at 0 to get binary mask
        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
        masks_per_frame[frame_idx] = mask.astype(np.uint8) * 255

        if frame_idx % 50 == 0:
            print(f"    Frame {frame_idx}/{num_frames-1}")

    # Reset state to free memory
    predictor.reset_state(inference_state)

    return masks_per_frame


# ─── Save Results ────────────────────────────────────────────────────

def save_results(frame_paths, masks, output_dir, prefix):
    """Save binary masks and masked RGB images."""
    mask_dir = os.path.join(output_dir, f"{prefix}_masks")
    rgb_dir = os.path.join(output_dir, f"{prefix}_masked")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    for frame_idx, mask in masks.items():
        frame_name = Path(frame_paths[frame_idx]).stem

        # Resize mask to match original frame if needed
        orig = cv2.imread(frame_paths[frame_idx])
        h, w = orig.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Save binary mask
        cv2.imwrite(os.path.join(mask_dir, f"{frame_name}_mask.png"), mask)

        # Save masked RGB
        masked = cv2.bitwise_and(orig, orig, mask=mask)
        cv2.imwrite(os.path.join(rgb_dir, f"{frame_name}_masked.png"), masked)

    print(f"  Saved {len(masks)} masks to: {mask_dir}")
    print(f"  Saved {len(masks)} masked frames to: {rgb_dir}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manual polygon → SAM 2 video segmentation pipeline"
    )
    parser.add_argument(
        "--input", type=str,
        default="/home/alexandercrain/Dropbox/Graduate Documents/"
                "Doctor of Philosophy/Thesis Research/Datasets/SPOT/"
                "RAWs MK5/DXL-ROT-DATASET/DXL-ROT-NOM-MK5-ZED",
        help="Folder containing left_XXXX / right_XXXX images"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output folder (default: <input>/sam2_output)"
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Frame filename prefix to process (e.g. 'left_', 'aba_'). "
             "Auto-detected when only one prefix exists in the folder."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to SAM 2 checkpoint .pt file"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to SAM 2 config .yaml file"
    )
    parser.add_argument(
        "--model_size", type=str, default="large",
        choices=["tiny", "small", "base_plus", "large"],
        help="SAM 2.1 model size (default: large). Ignored if --checkpoint is set."
    )
    parser.add_argument(
        "--prompt_frame", type=int, default=160,
        help="Frame index to display for polygon selection and SAM 2 prompt (default: 30)"
    )
    args = parser.parse_args()

    # ── Resolve checkpoint and config paths ──
    sam2_dirs = [
        os.path.expanduser("~/sam2"),
        os.path.expanduser("~/segment-anything-2"),
        os.path.join(os.path.dirname(__file__), "sam2"),
    ]

    size_map = {
        "tiny":      ("sam2.1_hiera_tiny.pt",      "configs/sam2.1/sam2.1_hiera_t.yaml"),
        "small":     ("sam2.1_hiera_small.pt",      "configs/sam2.1/sam2.1_hiera_s.yaml"),
        "base_plus": ("sam2.1_hiera_base_plus.pt",  "configs/sam2.1/sam2.1_hiera_b+.yaml"),
        "large":     ("sam2.1_hiera_large.pt",      "configs/sam2.1/sam2.1_hiera_l.yaml"),
    }

    ckpt_name, config_name = size_map[args.model_size]

    if args.checkpoint is None:
        for d in sam2_dirs:
            candidate = os.path.join(d, "checkpoints", ckpt_name)
            if os.path.isfile(candidate):
                args.checkpoint = candidate
                break
        if args.checkpoint is None:
            print(f"ERROR: Cannot find checkpoint '{ckpt_name}'.")
            print(f"  Searched: {sam2_dirs}")
            print(f"  Use --checkpoint to specify the path explicitly.")
            sys.exit(1)

    if args.config is None:
        for d in sam2_dirs:
            candidate = os.path.join(d, config_name)
            if os.path.isfile(candidate):
                args.config = candidate
                break
        if args.config is None:
            args.config = config_name
            print(f"  Using config name '{config_name}' (SAM 2 package resolution)")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")

    # ── Load frames ──
    output_dir = args.output or os.path.join(args.input, "sam2_output")
    frame_paths, prefix = load_sorted_frames(args.input, args.prefix)

    if not frame_paths:
        sys.exit(1)
    if len(frame_paths) < 2:
        print("Need at least 2 frames. Exiting.")
        sys.exit(1)

    # ── Step 1: Manual polygon selection ──
    prompt_frame_idx = min(args.prompt_frame, len(frame_paths) - 1)
    polygon_points, polygon_mask = get_manual_polygon_mask(frame_paths[prompt_frame_idx])

    if polygon_points is None:
        print("No region selected. Exiting.")
        sys.exit(0)

    # Save debug image showing the selected polygon overlaid on the prompt frame
    os.makedirs(output_dir, exist_ok=True)
    debug_img = cv2.imread(frame_paths[prompt_frame_idx])
    cv2.polylines(debug_img, [np.array(polygon_points, dtype=np.int32)],
                  isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(output_dir, "polygon_prompt_debug.png"), debug_img)
    cv2.imwrite(os.path.join(output_dir, "polygon_mask.png"), polygon_mask)
    print(f"  Debug images saved to: {output_dir}")

    # ── Step 2: Prepare frames for SAM 2 ──
    print("\nStep 2: Preparing frames for SAM 2...")
    video_dir = os.path.join(output_dir, f"_sam2_frames_{prefix}")
    prepare_video_dir(frame_paths, video_dir)

    # ── Step 3: Run SAM 2 ──
    print("\nStep 3: Running SAM 2 video segmentation...")
    masks = run_sam2_segmentation(
        video_dir, polygon_mask, len(frame_paths),
        args.checkpoint, args.config, prompt_frame_idx
    )

    # ── Step 4: Save results ──
    print("\nStep 4: Saving results...")
    save_results(frame_paths, masks, output_dir, prefix)

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
