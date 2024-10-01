import datetime
from ultralytics import YOLO

def main():
    # Start time tracking
    start_time = datetime.datetime.now()
    
    # Print start time
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load the YOLO model with the correct path
    model = YOLO(r"yolov10s.pt")  # Do not use the 'pretrained' argument here

    # Explicitly set the model to avoid using any additional pretrained weights (if applicable)
    # Depending on your use case, you may need to modify the model structure here

    # Train the model with updated parameters
    model.train(
        data=r"D:\pose estimation\images\Taekwondo_DATA\taekwondo.yaml",
        project=r'D:\pose estimation\runs\Taekwondo_dataset_detection',
        name='taekwondo_train_Y10s',
        epochs=100,
        imgsz=480,
        batch=32,
        device='0',  # Use GPU device 0
        workers=16,  # Number of data loading workers (adjust based on your CPU cores)
        lr0=0.001,
        lrf=0.0001,
        amp=False,  # Disable Automatic Mixed Precision (AMP) to avoid downloading YOLOv8
    )
    
    # End time tracking
    end_time = datetime.datetime.now()
    
    # Calculate training duration
    duration = end_time - start_time
    
    # Print end time and duration
    print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training duration: {duration}")

if __name__ == "__main__":
    main()
