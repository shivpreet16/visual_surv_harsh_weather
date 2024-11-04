import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from collections import defaultdict
from ultralytics import YOLO
import argparse
import time
from tqdm import tqdm

# Set number of times to derain/dehaze and other configurations
parser = argparse.ArgumentParser()
parser.add_argument('--numDerain', default=1, type=int, help='Number of times to derain')
parser.add_argument('--numDehaze', default=1, type=int, help='Number of times to dehaze')
parser.add_argument('--input_video', default='highway1.webm', help='Input video path')
parser.add_argument('--output_video', default='output_processed_tracking.mp4', help='Output video path')
args = parser.parse_args()

# Set paths for models and other resources
model_dehaze_path = 'models/wacv_gcanet_dehaze.pth'
model_derain_path = 'models/wacv_gcanet_derain.pth'
use_cuda = torch.cuda.is_available()

# Load GCANet model class
from GCANet import GCANet

# Load both dehazing and deraining models
model_dehaze = GCANet(in_c=4, out_c=3, only_residual=True)
model_derain = GCANet(in_c=4, out_c=3, only_residual=False)
model_dehaze.load_state_dict(torch.load(model_dehaze_path, map_location='cuda' if use_cuda else 'cpu'))
model_derain.load_state_dict(torch.load(model_derain_path, map_location='cuda' if use_cuda else 'cpu'))
if use_cuda:
    model_dehaze.cuda()
    model_derain.cuda()
model_dehaze.eval()
model_derain.eval()

# Load YOLO model for tracking
yolo_model = YOLO("yolo11l.pt")

# Initialize video capture and set up output writer
cap = cv2.VideoCapture(args.input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: [])

# Helper function for deweathering
def process_frame(img):
    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
    if use_cuda:
        img_data = img_data.cuda()

    # Deraining
    for _ in range(args.numDerain):
        edge_data = edge_compute(img_data)
        in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128
        in_data = in_data.cuda() if use_cuda else in_data.float()
        with torch.no_grad():
            pred = model_derain(Variable(in_data))
        img_data = (pred.data[0].cpu() + img_data).clamp(0, 255) if model_derain.only_residual else pred.data[0].cpu().clamp(0, 255)

    # Dehazing
    for _ in range(args.numDehaze):
        edge_data = edge_compute(img_data)
        in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128
        in_data = in_data.cuda() if use_cuda else in_data.float()
        with torch.no_grad():
            pred = model_dehaze(Variable(in_data))
        img_data = (pred.data[0].cpu() + img_data).clamp(0, 255) if model_dehaze.only_residual else pred.data[0].cpu().clamp(0, 255)

    return img_data.numpy().astype(np.uint8).transpose(1, 2, 0)

# Loop through the video frames with progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames, desc="Extracting frames") as extraction_pbar:
    with tqdm(total=total_frames, desc="Processing frames") as processing_pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update frame extraction progress bar
            extraction_pbar.update(1)

            # Derain and dehaze the frame
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            processed_frame = process_frame(frame_image)
            
            # Convert processed frame back to BGR format for OpenCV display and further processing
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

            # Run YOLO tracking on the processed frame
            results = yolo_model.track(source=processed_frame_bgr, persist=True)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                annotated_frame = results[0].plot()
                
                # Update and draw tracking lines
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=10)
            
                # Write annotated frame to the output video
                output_writer.write(annotated_frame)
            
            # Update frame processing progress bar
            processing_pbar.update(1)

# Release resources
cap.release()
output_writer.release()
print(f"Processed video saved as {args.output_video}")
cv2.destroyAllWindows()
