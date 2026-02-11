#!/usr/bin/env python3
import sys
from moviepy import VideoFileClip, clips_array, TextClip, CompositeVideoClip

'''
python tools/mergeVideos.py     /home/alexandercrain/Videos/Research/SITS_DV.mp4     /home/alexandercrain/Videos/Research/SITS_MATLAB.avi     output_comparison_sits.mp4     "SITS - DV-Processing"     "SITS - Custom MATLAB"
'''

def stack_horizontal(input1, input2, output="output.mp4", left_text="Left Video", right_text="Right Video"):
    # Load clips
    clip1 = VideoFileClip(input1)
    clip2 = VideoFileClip(input2)

    # Resize clip2 to match clip1's height (optional but recommended for clean stacking)
    # If you skip this, the shorter video will be centered vertically by default.
    if clip1.h != clip2.h:
        print(f"Resizing {input2} to match height of {input1}...")
        clip2 = clip2.resized(height=clip1.h)

    # Create text clips with styling
    txt_clip1 = TextClip(
        text=left_text,
        font_size=24,
        color="black",
        font="/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
        method="caption",
        size=(int(clip1.w * 0.8), None),
        bg_color="white",
        stroke_color="black",
        stroke_width=2
    )
    
    txt_clip2 = TextClip(
        text=right_text,
        font_size=24,
        color="black",
        font="/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
        method="caption",
        size=(int(clip2.w * 0.8), None),
        bg_color="white",
        stroke_color="black",
        stroke_width=2
    )

    # Position text at bottom center of each video
    txt_clip1 = txt_clip1.with_position(lambda t: ("center", clip1.h - txt_clip1.h - 10)).with_duration(clip1.duration)
    txt_clip2 = txt_clip2.with_position(lambda t: ("center", clip2.h - txt_clip2.h - 10)).with_duration(clip2.duration)

    # Composite text onto videos
    clip1 = CompositeVideoClip([clip1, txt_clip1])
    clip2 = CompositeVideoClip([clip2, txt_clip2])

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
        print("Usage: python mergeVideos.py left.mp4 right.mp4 [output.mp4] [left_text] [right_text]")
        sys.exit(1)

    input1 = sys.argv[1]
    input2 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "output.mp4"
    left_text = sys.argv[4] if len(sys.argv) > 4 else "Left Video"
    right_text = sys.argv[5] if len(sys.argv) > 5 else "Right Video"

    stack_horizontal(input1, input2, output, left_text, right_text)

if __name__ == "__main__":
    main()
