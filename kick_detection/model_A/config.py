# config.py
"""

yolo_model = load_yolo_model("D:/pose estimation/runs/Taekwondo_dataset_detection/taekwondo_train_Y10s5/weights/best.pt")
    
# Load action classification model
action_model = load_action_model(r"D:\pose estimation\datasets\punch_kick_dataset\models\model_B\checkpoints\train_10_20250111_190017\best_checkpoint.pth")
    
# Set video path (0 for webcam or provide a video file path)
video_path = r"D:\pose estimation\videos\Philippines vs Vietnam ｜ Taekwondo M -68kg Semifinal ｜ 2019 SEA Games.webm"  # Replace with your video path or 0 for webcam
    
"""
import torch
from torchvision import transforms

class Config:
    # Dataset paths
    ROOT_DIR = 'D:/pose estimation/datasets/punch_kick_dataset/train_dataset'
    CLASSES = ['kick', 'normal']
    SEQUENCE_LENGTH = 25

    # Training parameters
    BATCH_SIZE = 6
    EPOCHS = 100
    LEARNING_RATE = 1e-5
    TRAIN_VAL_SPLIT = 0.7  # 70% training, 30% validation
    WEIGHT_DECAY = 1e-2
    # Model parameters
    NUM_CLASSES = len(CLASSES)

    # Image dimensions  
    IMAGE_HEIGHT = 450  
    IMAGE_WIDTH = 200 

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data transforms
    TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),  # Resize to maintain aspect ratio (height, width)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Checkpoint paths
    CHECKPOINT_DIR = 'checkpoints'
