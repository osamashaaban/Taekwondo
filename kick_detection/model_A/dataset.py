# dataset.py

import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
from config import Config

class ActionRecognitionDataset(Dataset):
    def __init__(self, root_dir, classes, sequence_length=Config.SEQUENCE_LENGTH, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.sequence_length = sequence_length
        self.transform = transform
        self.data = []

        for label, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            for sequence in os.listdir(class_dir):
                sequence_path = os.path.join(class_dir, sequence)
                frame_paths = [os.path.join(sequence_path, f) for f in os.listdir(sequence_path)]
                frame_paths.sort()  # Ensure frames are in order
                if len(frame_paths) == sequence_length:
                    self.data.append((frame_paths, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_paths, label = self.data[idx]
        frames = []
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        # Stack frames to shape [sequence_length, channels, height, width]
        frames = torch.stack(frames)  # Shape: [sequence_length, channels, height, width]
        return frames, label

def get_train_val_datasets():
    dataset = ActionRecognitionDataset(Config.ROOT_DIR, Config.CLASSES, Config.SEQUENCE_LENGTH, Config.TRANSFORM)
    train_size = int(Config.TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset