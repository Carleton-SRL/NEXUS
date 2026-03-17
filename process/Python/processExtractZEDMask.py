"""
SAM 2 Motion Segmentation Pipeline
------------------------------------
1. Computes optical flow on frame 0->1 to auto-detect moving object
2. Converts motion region to a bounding box prompt
3. Feeds prompt to SAM 2 video predictor to segment entire sequence

Usage:
    python sam2_motion_segment.py --input <image_folder> [options]

Requires:
    - PyTorch with ROCm (or CUDA)
    - SAM 2 (pip install -e . from the sam2 repo)
    - OpenCV (pip install opencv-python)
"""

import cv2
import numpy as np
import argparse
import glob
import os
import sys
from pathlib import Path


# ─── Frame I/O ───────────────────────────────────────────────────────

def load_sorted_frames(folder, prefix):
    """Find all images matching prefix, sorted by frame number."""
    for ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
        pattern = os.path.join(folder, f"{prefix}_*.{ext}")
        paths = sorted(glob.glob(pattern))
        if paths:
            print(f"Found {len(paths)} '{prefix}' frames (.{ext})")
            return paths
    print(f"WARNING: No frames found for prefix '{prefix}'")
    return []


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


# ─── Optical Flow → Bounding Box ────────────────────────────────────

def detect_motion_bbox(frame_paths, prompt_frame=30, flow_threshold=1.5, morph_size=7):
    """
    Compute optical flow between prompt_frame and prompt_frame+1.
    Return bounding box [x1, y1, x2, y2] around the moving region.
    """
    idx = min(prompt_frame, len(frame_paths) - 2)
    img0 = cv2.imread(frame_paths[idx])
    img1 = cv2.imread(frame_paths[idx + 1])
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray0, gray1,
        flow=None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # Flow magnitude → binary mask
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = (mag > flow_threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if mask.sum() == 0:
        print(f"  ERROR: No motion detected at frames {idx}-{idx + 1}. Lower --flow_threshold or check data.")
        return None, mask

    # Find bounding box of the motion region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Merge all contours into one bounding rect
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Add padding (10% on each side, clamped to image bounds)
    H, W = mask.shape
    pad_x = int(w * 0.10)
    pad_y = int(h * 0.10)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    bbox = [x1, y1, x2, y2]
    print(f"  Motion bbox: [{x1}, {y1}, {x2}, {y2}] (w={x2-x1}, h={y2-y1})")
    return bbox, mask


# ─── SAM 2 Video Segmentation ───────────────────────────────────────

def run_sam2_segmentation(video_dir, bbox, flow_mask, num_frames, checkpoint, config, prompt_frame=0):
    """
    Use SAM 2 video predictor to segment the object across all frames,
    seeded with a flow mask prompt at prompt_frame.
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
    mask_resized = cv2.resize(flow_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_input = (mask_resized > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_input).to(device)

    _, obj_ids, mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=prompt_frame_idx,
        obj_id=1,
        mask=mask_tensor,
    )
    print(f"  Prompted on frame {prompt_frame_idx} with flow mask (bbox {bbox})")

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
        description="Optical flow → SAM 2 motion segmentation pipeline"
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
        "--camera", type=str, default="left", choices=["left", "right"],
        help="Which camera to process (default: left)"
    )
    parser.add_argument(
        "--flow_threshold", type=float, default=1.1,
        help="Optical flow magnitude threshold for motion detection (default: 1.5)"
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
        "--prompt_frame", type=int, default=30,
        help="Frame index to compute flow and place SAM 2 prompt (default: 30)"
    )
    args = parser.parse_args()

    # ── Resolve checkpoint and config paths ──
    # Try to find the sam2 repo relative to this script or in common locations
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
            # SAM 2 can also resolve configs by name if installed as a package
            args.config = config_name
            print(f"  Using config name '{config_name}' (SAM 2 package resolution)")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")

    # ── Load frames ──
    output_dir = args.output or os.path.join(args.input, "sam2_output")
    frame_paths = load_sorted_frames(args.input, args.camera)

    if len(frame_paths) < 2:
        print("Need at least 2 frames. Exiting.")
        sys.exit(1)

    # ── Step 1: Detect motion via optical flow ──
    print("\nStep 1: Detecting motion via optical flow...")
    bbox, flow_mask = detect_motion_bbox(frame_paths, args.prompt_frame, args.flow_threshold)

    if bbox is None:
        sys.exit(1)

    # Save the flow mask for visual verification
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(output_dir, "flow_detection_mask.png"),
        flow_mask
    )

    # Draw bbox on the prompt frame for verification
    debug_idx = min(args.prompt_frame, len(frame_paths) - 1)
    debug_img = cv2.imread(frame_paths[debug_idx])
    cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.imwrite(
        os.path.join(output_dir, "bbox_prompt_debug.png"),
        debug_img
    )
    print(f"  Debug images saved to: {output_dir}")

    # ── Step 2: Prepare frames for SAM 2 ──
    print("\nStep 2: Preparing frames for SAM 2...")
    video_dir = os.path.join(output_dir, f"_sam2_frames_{args.camera}")
    prepare_video_dir(frame_paths, video_dir)

    # ── Step 3: Run SAM 2 ──
    print("\nStep 3: Running SAM 2 video segmentation...")
    masks = run_sam2_segmentation(
        video_dir, bbox, flow_mask, len(frame_paths),
        args.checkpoint, args.config, args.prompt_frame
    )

    # ── Step 4: Save results ──
    print("\nStep 4: Saving results...")
    save_results(frame_paths, masks, output_dir, args.camera)

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()