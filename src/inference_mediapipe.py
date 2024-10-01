import mediapipe as mp  # Add Mediapipe for pose estimation
import numpy as np
import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import copy
import time
import sys

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

# Initialize Mediapipe Pose Estimation
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class TaekwondoPlayer():
    def __init__(self, model_path):
        self.detector = YOLO(model_path)  # Player detection model
        self.tracker = BYTETracker(args, frame_rate=30.0)
        self.classes = self.detector.names
        self.pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect_players_and_poses(self, img):
        # Detect players using YOLO
        results = self.detector(img)
        players_boxes = results[0].boxes
        boxes, scores, class_ids, poses = [], [], [], []

        for box in players_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            class_id = int(box.cls[0])

            # Extract the ROI (Region of Interest) of the player for pose estimation
            player_roi = img[y1:y2, x1:x2]

            # Run pose estimation on the player's ROI
            player_rgb = cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB)
            results_pose = self.pose_model.process(player_rgb)

            if results_pose.pose_landmarks:
                # Save the keypoints of the pose
                poses.append(results_pose.pose_landmarks)

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)

        return np.array(boxes), np.array(scores), np.array(class_ids), poses

class VideoProcessor():
    def __init__(self, video_path, output_path, player_detector):
        self.video_path = video_path
        self.output_path = output_path
        self.player_detection = player_detector
        self.trackers = copy.deepcopy(BYTETracker(args))
        self.track_labels = defaultdict(lambda: None)
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error: could not open video file {video_path}!")

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (640, 640))

    def process_video(self):
        frame_count = 0
        connections = mp_pose.POSE_CONNECTIONS  # Use Mediapipe pose connections

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame_count += 1
            frame_resized = cv2.resize(frame, (640, 640))
            inference_start_time = time.time()

            # Detect players and their poses
            boxes, scores, class_ids, poses = self.player_detection.detect_players_and_poses(frame_resized)
            total_time = time.time() - inference_start_time
            print(f"Detected {len(boxes)} players with poses.")

            # Create a dictionary to store player detections by class_id
            player_dict = defaultdict(list)

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                class_id = class_ids[i]
                score = scores[i]
                pose = poses[i] if len(poses) > i else None

                player_dict[class_id].append((i, score, (x1, y1, x2, y2), pose))

            for class_id, detections in player_dict.items():
                # Sort detections by score (confidence) in descending order
                detections.sort(key=lambda x: x[1], reverse=True)

                for idx, (detection_idx, score, (x1, y1, x2, y2), pose) in enumerate(detections):
                    color = (255, 0, 0) if class_id == 0 else (0, 0, 255)  # Red for class_id 0, Blue otherwise

                    # Assign labels based on confidence ranking
                    if idx == 0:
                        label = f'{self.player_detection.classes[class_id]}, {score:.2f}'  # Main label
                    else:
                        label = f'{self.player_detection.classes[class_id]}_2, {score:.2f}'  # Secondary label

                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw pose keypoints and connections
                    if pose:
                        mp_drawing.draw_landmarks(
                            frame_resized[y1:y2, x1:x2], 
                            pose, 
                            connections, 
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )

            # Display FPS and inference time
            fps_text = f"FPS: {self.fps:.2f}"
            inference_time_text = f"Inference Time: {total_time:.2f}s"
            cv2.putText(frame_resized, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame_resized, inference_time_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Write and display the frame
            self.out.write(frame_resized)
            cv2.imshow('Video Processing', frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_count == 1500:
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path = "Path to video"
    model_path = "Train_weights_Taekwondo_dataset_detection/taekwondo_train_Y10s5\weights\best.pt"
    output_path = "Path to save video/Taekwondo_Players.avi"

    player_detect = TaekwondoPlayer(model_path)
    video_processor = VideoProcessor(video_path, output_path, player_detect)
    video_processor.process_video()
