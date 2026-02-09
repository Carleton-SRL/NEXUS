#!/usr/bin/env python3
import sys
from moviepy import VideoFileClip, clips_array

def stack_horizontal(input1, input2, output="output.mp4"):
    # Load clips
    clip1 = VideoFileClip(input1)
    clip2 = VideoFileClip(input2)

    # Resize clip2 to match clip1's height (optional but recommended for clean stacking)
    # If you skip this, the shorter video will be centered vertically by default.
    if clip1.h != clip2.h:
        print(f"Resizing {input2} to match height of {input1}...")
        clip2 = clip2.resized(height=clip1.h)

    # Use clips_array for automatic layout
    # [[clip1, clip2]] means one row with two columns
    final = clips_array([[clip1, clip2]])

    # Write output
    final.write_videofile(output)

    # Close clips to release system resources
    clip1.close()
    clip2.close()
    final.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python mergeVideos.py left.mp4 right.mp4 [output.mp4]")
        sys.exit(1)

    input1 = sys.argv[1]
    input2 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "output.mp4"

    stack_horizontal(input1, input2, output)

if __name__ == "__main__":
    main()
