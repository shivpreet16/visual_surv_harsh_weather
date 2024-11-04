from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def split_video(input_path, output_folder):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the video file
    video = VideoFileClip(input_path)
    
    # Define segment duration in seconds (1 minute = 60 seconds)
    segment_duration = 60
    total_duration = int(video.duration)
    
    # Loop over the video duration in 1-minute increments
    for start_time in range(0, total_duration, segment_duration):
        end_time = min(start_time + segment_duration, total_duration)
        
        # Create a subclip for the segment
        segment = video.subclip(start_time, end_time)
        
        # Define the output path for each segment
        output_path = f"{output_folder}/segment_{start_time // 60 + 1}.mp4"
        
        # Write the segment to the output path
        segment.write_videofile(output_path, codec="libx264")
    
    # Close the video file
    video.close()
    print("Video split into 1-minute segments successfully.")

# Example usage
input_video_path = "afternoon-rain-small.mp4"
output_directory = f"output_segments_{input_video_path.split('.')[0]}"
split_video(input_video_path, output_directory)
