import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from utils import make_dataset, edge_compute  # Ensure utils.py has these functions
import argparse
from tqdm import tqdm
import time

# Set number of times to derain/dehaze
parser = argparse.ArgumentParser()
parser.add_argument('--numDerain', default=1, type=int, help='Number of times to derain')
parser.add_argument('--numDehaze', default=1, type=int, help='Number of times to dehaze')
parser.add_argument('--input_video', default='highway1.webm', help='Input video path')
args = parser.parse_args()

# Set paths for input and output
input_video = args.input_video
output_video = 'output_processed_video.mp4'
model_dehaze_path = 'models/wacv_gcanet_dehaze.pth'
model_derain_path = 'models/wacv_gcanet_derain.pth'
use_cuda = torch.cuda.is_available()

# Load GCANet model class
from GCANet import GCANet

# Load both models with their respective weights
model_dehaze = GCANet(in_c=4, out_c=3, only_residual=True)
model_derain = GCANet(in_c=4, out_c=3, only_residual=False)

# Load models onto GPU if available
if use_cuda:
    model_dehaze.cuda()
    model_derain.cuda()
else:
    model_dehaze.float()
    model_derain.float()

model_dehaze.load_state_dict(torch.load(model_dehaze_path, map_location='cuda' if use_cuda else 'cpu'))
model_derain.load_state_dict(torch.load(model_derain_path, map_location='cuda' if use_cuda else 'cpu'))
model_dehaze.eval()
model_derain.eval()

# Function to extract frames from video
def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_dir = f"{os.path.splitext(os.path.basename(video_path))[0]}_frames"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    cap.release()
    return frame_paths, frame_dir

# Function to process each frame with both deraining and dehazing
def process_frame(img_path):
    img = Image.open(img_path).convert('RGB')
    im_w, im_h = img.size
    if im_w % 4 != 0 or im_h % 4 != 0:
        img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4))) 

    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
    if use_cuda:
        img_data = img_data.cuda()

    # Apply deraining
    for _ in range(args.numDerain):
        edge_data = edge_compute(img_data)
        in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128
        in_data = in_data.cuda() if use_cuda else in_data.float()
        
        with torch.no_grad():
            pred = model_derain(Variable(in_data))
        
        img_data = (pred.data[0].cpu() + img_data).clamp(0, 255) if model_derain.only_residual else pred.data[0].cpu().clamp(0, 255)

    # Apply dehazing
    for _ in range(args.numDehaze):
        edge_data = edge_compute(img_data)
        in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128
        in_data = in_data.cuda() if use_cuda else in_data.float()
        
        with torch.no_grad():
            pred = model_dehaze(Variable(in_data))
        
        img_data = (pred.data[0].cpu() + img_data).clamp(0, 255) if model_dehaze.only_residual else pred.data[0].cpu().clamp(0, 255)

    # Convert final output to image format
    return img_data.numpy().astype(np.uint8).transpose(1, 2, 0)


# Function to recompile frames into a video
def frames_to_video(frame_dir, output_video_path, fps=30):
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frame_files:
        print("No frames to compile.")
        return

    frame_sample = cv2.imread(frame_files[0])
    height, width, _ = frame_sample.shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in tqdm(frame_files, desc="Compiling video"):
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"Processed video saved as {output_video_path}")

# Main processing pipeline
print("Extracting frames from video...")
frame_paths, frame_dir = video_to_frames(input_video)

processed_frame_dir = os.path.join(frame_dir, "processed")
if not os.path.exists(processed_frame_dir):
    os.makedirs(processed_frame_dir)

# Process each frame and save it
print("Processing frames with deraining and dehazing...")
start_time = time.time()
for frame_path in tqdm(frame_paths, desc="Processing frames"):
    processed_img = process_frame(frame_path)
    processed_img_path = os.path.join(processed_frame_dir, os.path.basename(frame_path))
    Image.fromarray(processed_img).save(processed_img_path)

# Display estimated time for processing
end_time = time.time()
processing_time = end_time - start_time
avg_time_per_frame = processing_time / len(frame_paths)
estimated_total_time = avg_time_per_frame * len(frame_paths)
print(f"Average time per frame: {avg_time_per_frame:.2f} seconds")
print(f"Estimated total processing time: {estimated_total_time / 60:.2f} minutes")

# Recompile processed frames into a video
frames_to_video(processed_frame_dir, output_video)
