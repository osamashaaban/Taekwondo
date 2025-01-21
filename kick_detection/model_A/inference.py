import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from resnet_lstm import ResNetLSTM  # Import the new model
from config import Config
from ultralytics import YOLO
from collections import defaultdict
import sys

# Append the ByteTrack path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ByteTrack/yolox/tracker')))
from byte_tracker import BYTETracker

class Args:
    track_thresh = 0.5
    track_buffer = 50
    match_thresh = 0.8
    aspect_ratio_thresh = 10.0
    min_box_area = 1.0
    mot20 = False

args = Args()

# Load YOLO model for player detection
def load_yolo_model(weights_path):
    model = YOLO(weights_path, verbose=False)  # Set verbose=False to suppress logs
    return model

# Load action classification model
def load_action_model(checkpoint_path):
    model = ResNetLSTM(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)  # Use the new model
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint)  # Load model state
    model.eval()
    return model

# Preprocess frames for action classification
def preprocess_frames(frames):
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frames = [transform(Image.fromarray(frame)) for frame in frames]
    frames = torch.stack(frames)  # Shape: [sequence_length, channels, height, width]
    return frames.unsqueeze(0).to(Config.DEVICE)  # Add batch dimension

# Main inference function
def main():
    # Load YOLO model
    yolo_model = load_yolo_model("D:/pose estimation/runs/Taekwondo_dataset_detection/taekwondo_train_Y10s5/weights/best.pt")
    
    # Load action classification model
    action_model = load_action_model(r"D:\pose estimation\datasets\punch_kick_dataset\models\model_A\checkpoints\train_2_20250115_232049\last_checkpoint.pth")
    
    # Set video path (0 for webcam or provide a video file path)
    video_path = r"D:\pose estimation\videos\korea.webm"  # Replace with your video path or 0 for webcam
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define output video path
    output_video_path = "D:/pose estimation/datasets/punch_kick_dataset/models/model_B/output_inference/output_video.mp4"
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize ByteTrack tracker
    tracker = BYTETracker(args)  # Create a ByteTrack tracker instance
    
    # Initialize frame buffers for each player
    frame_buffers = defaultdict(list)  # Dictionary to store frame buffers for each tracked player
    
    # Initialize kick counters
    red_kick_count = 0
    blue_kick_count = 0
    
    # Initialize kick detection counters
    red_kick_detections = 0
    blue_kick_detections = 0
    
    # Initialize frame counter
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update frame counter
        frame_counter += 1
        
        # Skip frames outside the range 3000-3600
        if frame_counter < 3100:
            continue
        if frame_counter > 3600:
            break
        
        # Perform player detection
        results = yolo_model(frame)
        
        # Access detections from the results
        detections = results[0].boxes.data.cpu().numpy()  # Extract detections from the first (and only) result
        
        # Prepare detections for ByteTrack
        output_results = np.zeros((len(detections), 6))
        output_results[:, :4] = detections[:, :4]  # Bounding boxes
        output_results[:, 4] = detections[:, 4]  # Confidence scores
        output_results[:, 5] = detections[:, 5]  # Class IDs
        
        # Update tracker with detections
        tracked_objects = tracker.update(output_results, (frame_width, frame_height), (frame_width, frame_height))
        
        # Process each tracked player
        for track in tracked_objects:
            track_id = int(track.track_id)
            bbox = track.tlbr  # Get bounding box in (x1, y1, x2, y2) format
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get the class ID from the original detections
            class_id = int(track.class_id)  # Class ID (0 for blue, 1 for red)
            
            # Crop player from frame
            player_frame = frame[y1:y2, x1:x2]
            
            # Initialize frame buffer for new players
            if track_id not in frame_buffers:
                frame_buffers[track_id] = []
            
            # Add frame to buffer with skip rate of n frames
            if frame_counter % 1 == 0:  # Process every frame
                if len(frame_buffers[track_id]) < Config.SEQUENCE_LENGTH:
                    frame_buffers[track_id].append(player_frame)
                else:
                    frame_buffers[track_id].pop(0)
                    frame_buffers[track_id].append(player_frame)
            
            # If buffer is full, classify action
            if len(frame_buffers[track_id]) == Config.SEQUENCE_LENGTH:
                frames = frame_buffers[track_id]
                input_tensor = preprocess_frames(frames)
                with torch.no_grad():
                    output = action_model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    action = Config.CLASSES[predicted.item()]  # Map output to class name
                    print(f"[action] : {predicted.item()}")
                
                # Update kick detection counters
                if action == "kick":
                    if class_id == 0:  # Blue player
                        blue_kick_detections += 1
                    elif class_id == 1:  # Red player
                        red_kick_detections += 1
                    
                    # Increment kick count every 15 detections
                    if class_id == 0 and blue_kick_detections % 10 == 0:
                        blue_kick_count += 1
                    elif class_id == 1 and red_kick_detections % 10 == 0:
                        red_kick_count += 1
                
                # Draw bounding box and action label for both "kick" and "normal"
                color = (255, 0, 0) if class_id == 0 else (0, 0, 255)  # Blue for blue_player, Red for red_player
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{action}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display kick counts on the frame
        cv2.putText(frame, f"Blue Kicks: {blue_kick_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f"Red Kicks: {red_kick_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow('Taekwondo Action Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()