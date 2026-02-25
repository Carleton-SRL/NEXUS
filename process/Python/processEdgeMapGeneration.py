import dv_processing as dv
import cv2
from pathlib import Path
import datetime
import os

# --- 1. Configuration ---

# Path to the dataset directory. The script will find the first .aedat4 file inside.
dataset_path = Path('/home/alexandercrain/Dropbox/Graduate Documents/Doctor of Philosophy/Thesis Research/Datasets/SPOT/AEDAT4/ROT_NOM')

output_path = Path('/home/alexandercrain/Videos/Research')

# Desired output frame rate for the video
output_fps = 60

# --- 2. File and Directory Setup ---

# Find the first aedat4 file in the directory
try:
    aedat4_file = next(dataset_path.glob('*.aedat4'))
    print(f"Found and loading aedat4 file: {aedat4_file}")
except StopIteration:
    print(f"Error: No .aedat4 file found in '{dataset_path}'")
    exit()

# Create a unique output directory based on the input filename and current time
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_parent_dir = output_path / f"{aedat4_file.stem}_output_{timestamp}"

# Create subdirectories for the video and the individual frames
output_video_dir = output_parent_dir / "video"
output_frames_dir = output_parent_dir / "frames"
output_video_dir.mkdir(parents=True, exist_ok=True)
output_frames_dir.mkdir(parents=True, exist_ok=True)

# --- 3. Initialize DV Processing and Video Writer ---

# Open the recording file
try:
    recording = dv.io.MonoCameraRecording(str(aedat4_file))
except RuntimeError as e:
    print(f"Error opening file: {e}")
    exit()

# Make sure we have an event stream available
if not recording.isEventStreamAvailable():
    print("Error: Event stream is not available in the recording.")
    exit()

# Get resolution from the event stream for configuring components
resolution = recording.getEventResolution()
print(f"Stream resolution: {resolution[0]}x{resolution[1]}")

# GENERAL ACCUMULATOR
'''
accumulator = dv.Accumulator(resolution)
accumulator.setMinPotential(0.0)
accumulator.setMaxPotential(1.0)
accumulator.setNeutralPotential(0.5)
accumulator.setEventContribution(0.15)
accumulator.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
accumulator.setDecayParam(1e+6)
accumulator.setIgnorePolarity(True)
accumulator.setSynchronousDecay(False)
output_name = "simple_accumulator.mp4"
'''

# EDGE MAP ACCUMULATOR

accumulator = dv.EdgeMapAccumulator(resolution)
accumulator.setNeutralPotential(0.5)
accumulator.setEventContribution(0.15)
accumulator.setIgnorePolarity(False)
output_name = "edge_map_accumulator.mp4"


# TIME SURFACE ACCUMULATOR

'''
accumulator = dv.TimeSurface(resolution)
output_name = "time_surface_map.mp4"
'''

# SPEED INVARIENT TIME SURFACE
'''
accumulator = dv.SpeedInvariantTimeSurface(resolution)
output_name = "speed_invarient_time_surface_map.mp4"
'''

# Create a filter chain to reduce noise
#filter_chain = dv.EventFilterChain()
#filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(resolution, 1.0))

# Set up the OpenCV VideoWriter to save the MP4
video_path = output_video_dir / output_name
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
video_writer = cv2.VideoWriter(str(video_path), fourcc, output_fps, (resolution[0], resolution[1]), isColor=False)
print(f"Will save video to: {video_path}")
print(f"Will save frames to: {output_frames_dir}")

# --- 4. Processing Logic ---

# Global frame counter for naming exported image files
frame_counter = 0

def process_events_callback(events: dv.EventStore):
    """
    This function is called by the slicer for each time interval.
    It filters events, passes them to the accumulator, generates a frame,
    and saves the frame to both the video and an image file.
    """
    global frame_counter

    # Pass data to the filter chain
    #filter_chain.accept(events)
    # Get the filtered events
    #filtered_events = filter_chain.generateEvents()

    # Pass the filtered data into the accumulator
    if events is not None and not events.isEmpty():
        accumulator.accept(events)

        # Generate the edge map frame
        frame = accumulator.generateFrame()

        # Write the frame to the MP4 video file
        video_writer.write(frame.image)

        # Export the frame as a PNG image
        frame_filename = output_frames_dir / f"frame_{frame_counter:05d}.png"
        cv2.imwrite(str(frame_filename), frame.image)

        frame_counter += 1

# Initiate the event stream slicer
slicer = dv.EventStreamSlicer()
# Calculate slicing interval from the desired FPS
slicing_interval = datetime.timedelta(seconds=0.33)
slicer.doEveryTimeInterval(slicing_interval, process_events_callback)

# --- 5. Main Processing Loop ---

# Get the first batch of events
events = recording.getNextEventBatch()

print("\nProcessing events...")
# Loop while events are available
while events is not None:
    # Pass the events to the slicer, which will trigger the callback periodically
    slicer.accept(events)
    # Get the next batch of events from the file
    events = recording.getNextEventBatch()

# Process any remaining events in the slicer
#slicer.doWithoutTrigger()

print(f"\nProcessing complete.")
print(f"Total frames generated: {frame_counter}")

# --- 6. Finalize ---

# Release the video writer to finalize the video file
video_writer.release()
print("Video file finalized and saved.")
