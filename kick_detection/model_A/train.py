import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from dataset import get_train_val_datasets
from resnet_lstm import ResNetLSTM
from config import Config
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

def main():
    # Create checkpoint directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Get the maximum train index from existing directories
    existing_dirs = [dir for dir in os.listdir(Config.CHECKPOINT_DIR) if os.path.isdir(os.path.join(Config.CHECKPOINT_DIR, dir)) and '_' in dir]
    current_train_idx = np.max([int(dir.split('_')[1]) for dir in existing_dirs]) if existing_dirs else 0

    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the new train directory name with index and datetime
    new_train_path = os.path.join(Config.CHECKPOINT_DIR, f"train_{current_train_idx + 1}_{current_datetime}")
    os.makedirs(new_train_path, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=new_train_path)

    # Get datasets and dataloaders
    train_dataset, val_dataset = get_train_val_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    # Model, loss, and optimizer
    model = ResNetLSTM(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize a new optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler
    scaler = GradScaler()  # For mixed precision training

    # Load checkpoint
    checkpoint_path = r"D:\pose estimation\datasets\punch_kick_dataset\models\model_A\checkpoints\train_5_20250116_124519\last_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint contains metadata (epoch, optimizer state, etc.)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        best_val_loss = checkpoint['best_val_loss']
    else:
        # Checkpoint only contains the model's state dictionary
        model.load_state_dict(checkpoint)
        start_epoch = 50  # Manually set the starting epoch
        best_val_loss = float('inf')  # Reset best validation loss

    print(f"Loaded checkpoint. Resuming training from epoch {start_epoch}.")

    # Training and validation loop
    for epoch in range(start_epoch, Config.EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Wrap the training dataloader with tqdm
        train_loop = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{Config.EPOCHS}] Training")
        for frames, labels in train_loop:
            frames, labels = frames.to(Config.DEVICE), labels.to(Config.DEVICE)

            # Forward pass with mixed precision
            optimizer.zero_grad()
            with autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)

            # Backward pass and optimization with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            running_loss += loss.item() * frames.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update tqdm progress bar with current loss and accuracy
            train_loop.set_postfix({
                "Loss": running_loss / total,
                "Accuracy": 100 * correct / total
            })

        train_loss = running_loss / len(train_dataset)
        train_accuracy = 100 * correct / total

        # Log training metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        # Wrap the validation dataloader with tqdm
        val_loop = tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{Config.EPOCHS}] Validation")
        with torch.no_grad():
            for frames, labels in val_loop:
                frames, labels = frames.to(Config.DEVICE), labels.to(Config.DEVICE)

                with autocast():
                    outputs = model(frames)
                    loss = criterion(outputs, labels)

                # Metrics
                val_running_loss += loss.item() * frames.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # Update tqdm progress bar with current loss and accuracy
                val_loop.set_postfix({
                    "Loss": val_running_loss / val_total,
                    "Accuracy": 100 * val_correct / val_total
                })

        val_loss = val_running_loss / len(val_dataset)
        val_accuracy = 100 * val_correct / val_total

        # Log validation metrics to TensorBoard
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save the last checkpoint
        last_checkpoint_path = os.path.join(new_train_path, "last_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, last_checkpoint_path)

        # Save the best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(new_train_path, "best_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_checkpoint_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # Step the scheduler
        scheduler.step()

    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()