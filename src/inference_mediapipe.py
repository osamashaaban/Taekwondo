import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import supervision as sv
from collections import defaultdict
import time

# Initialize Mediapipe Pose Estimation
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class TaekwondoPlayer:
    def __init__(self, model_path):
        self.detector = YOLO(model_path)  # Player detection model
        self.pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect_players_and_poses(self, img):
        # Detect players using YOLO
        results = self.detector(img)
        # Using 'from_ultralytics' to convert detections to 'supervision' format
        detections = sv.Detections.from_ultralytics(results[0])
        
        poses = []
        for box in detections.xyxy:
            x1, y1, x2, y2 = map(int, box)
            
            # Extract the ROI (Region of Interest) for pose estimation
            player_roi = img[y1:y2, x1:x2]
            player_rgb = cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB)
            results_pose = self.pose_model.process(player_rgb)
            poses.append(results_pose.pose_landmarks if results_pose.pose_landmarks else None)
        
        return detections, poses

class VideoProcessor:
    def __init__(self, video_path, output_path, player_detector):
        self.video_path = video_path
        self.output_path = output_path
        self.player_detection = player_detector
        self.tracker = sv.ByteTrack()
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error: could not open video file {video_path}!")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def process_video(self):
        frame_count = 0
        connections = mp_pose.POSE_CONNECTIONS

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame_count += 1
            frame_resized = cv2.resize(frame, (640, 640))
            inference_start_time = time.time()

            # Detect players and their poses
            detections, poses = self.player_detection.detect_players_and_poses(frame_resized)

            # Track detections
            tracked_detections = self.tracker.update_with_detections(detections)

            # Annotate each tracked detection
            for i, bbox in enumerate(tracked_detections.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                score = tracked_detections.confidence[i]
                class_id = tracked_detections.class_id[i]

                color = (255, 0, 0) if class_id == 0 else (0, 0, 255)
                label = f'{self.player_detection.detector.names[class_id]}, {score:.2f}'

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if poses[i]:
                    mp_drawing.draw_landmarks(
                        frame_resized[y1:y2, x1:x2],
                        poses[i],
                        connections,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )

            # Display FPS and inference time
            total_time = time.time() - inference_start_time
            fps_text = f"FPS: {self.fps:.2f}"
            inference_time_text = f"Inference Time: {total_time:.2f}s"
            cv2.putText(frame_resized, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame_resized, inference_time_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Write the processed frame
            self.out.write(frame_resized)
            cv2.imshow('Video Processing', frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"D:\pose estimation\fight.mp4"
    model_path = r"D:\pose estimation\runs\Taekwondo_dataset_detection\taekwondo_train_Y10s5\weights\best.pt"
    output_path = r"D:\pose estimation\Taekwondo\Taekwondo_Players.avi"

    player_detect = TaekwondoPlayer(model_path)
    video_processor = VideoProcessor(video_path, output_path, player_detect)
    video_processor.process_video()
