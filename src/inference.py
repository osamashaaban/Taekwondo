from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import copy
import time
import sys
import cv2
import os

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

class TaekwondoPlayer():
    def __init__(self, model_path, pose_model_path):
        self.detector = YOLO(model_path)  # Player detection model
        self.pose_model = YOLO(pose_model_path)  # Pose estimation model
        self.tracker = BYTETracker(args, frame_rate=30.0)
        self.classes = self.detector.names

    def detect_players_and_poses(self, img):
        # Detect players
        results = self.detector(img)
        players_boxes = results[0].boxes
        boxes, scores, class_ids, poses = [], [], [], []

        # Loop through detected players to get their poses
        for box in players_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            class_id = int(box.cls[0])

            # Extract the ROI (Region of Interest) of the player for pose estimation
            player_roi = img[y1:y2, x1:x2]

            # Run the pose estimation on the player's ROI
            pose_results = self.pose_model(player_roi)
            if pose_results:
                keypoints = pose_results[0].keypoints.cpu().numpy()  # Get keypoints of the pose
                poses.append(keypoints)

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

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (640, 640))

    def process_video(self):
        frame_count = 0
        connections = [
                        (0, 1),  # Neck -> Shoulder (left)
                        (1, 2),  # Shoulder (left) -> Elbow (left)
                        (2, 3),  # Elbow (left) -> Wrist (left)
                        (0, 4),  # Neck -> Shoulder (right)
                        (4, 5),  # Shoulder (right) -> Elbow (right)
                        (5, 6),  # Elbow (right) -> Wrist (right)
                        (0, 7),  # Neck -> Hip (left)
                        (7, 8),  # Hip (left) -> Knee (left)
                        (8, 9),  # Knee (left) -> Ankle (left)
                        (0, 10), # Neck -> Hip (right)
                        (10, 11),# Hip (right) -> Knee (right)
                        (11, 12) # Knee (right) -> Ankle (right)
                    ]
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

            # Draw bounding boxes and poses
            # Inside the video processing loop
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                class_id = class_ids[i]
                score = scores[i]
                pose = poses[i] if len(poses) > i else None  # Get the pose if available

                # Draw player bounding box
                color = (255, 0, 0) if class_id == 0 else (0, 0, 255)
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_resized, f'{self.player_detection.classes[class_id]}, {score:.2f}', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw the pose (keypoints and connections)
                if pose is not None:
                    keypoints = pose.xy[0] if isinstance(pose.xy, np.ndarray) else pose.xy  # Extract keypoint coordinates

                    # Draw keypoints
                    for point in keypoints:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(frame_resized, (x1 + x, y1 + y), 3, (0, 255, 0), -1)  # Draw keypoint

                    # Draw connections between keypoints
                    for start, end in connections:
                        if start < len(keypoints) and end < len(keypoints):
                            x_start, y_start = int(keypoints[start][0]), int(keypoints[start][1])
                            x_end, y_end = int(keypoints[end][0]), int(keypoints[end][1])
                            cv2.line(frame_resized, (x1 + x_start, y1 + y_start), (x1 + x_end, y1 + y_end), (0, 255, 0), 2)  # Draw line

                    # Write the frame to the output video
                    self.out.write(frame_resized)



            # Display FPS and Inference time
            fps_text = f"FPS: {self.fps:.2f}"
            inference_time_text = f"Inference Time: {total_time:.2f}s"
            cv2.putText(frame_resized, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame_resized, inference_time_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Write the frame to the output video
            self.out.write(frame_resized)

            # Display the frame
            cv2.imshow('Video Processing', frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_count == 1500:
                break
        
        self.cap.release()
        self.out.release()  # Release the video writer
        cv2.destroyAllWindows()

if __name__ == "__main__":

    video_path = r"D:\pose estimation\fight.mp4"
    model_path = r"D:\pose estimation\runs\Taekwondo_dataset_detection\taekwondo_train_Y10s5\weights\best.pt"
    pose_model_path = r"D:\pose estimation\yolov8m-pose.pt"  # Pose estimation model path
    output_path = r"D:\pose estimation\Taekwondo_Players.avi"

    player_detect = TaekwondoPlayer(model_path, pose_model_path)

    video_processor = VideoProcessor(video_path, output_path, player_detect)
    video_processor.process_video()
