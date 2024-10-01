
# Taekwondo

This is the first phase of applying video analysis and player performance analysis on red and blue players.

## Environment Setup

To get started with this project, follow the steps below to create and activate a Conda environment, and install the necessary packages.

### Step 1: Create Conda Environment
```bash
conda create --name taekwondo python=3.9
```

### Step 2: Activate the Environment
```bash
conda activate taekwondo
```

### Step 3: Install Packages

You can install the required packages from `requirements.txt` or manually using the following steps:

```bash
# Install YOLOv8 and required dependencies
pip install ultralytics

# Install PyTorch and CUDA for GPU acceleration
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install OpenCV for video processing
pip install opencv-python

# Install Requests for HTTP requests
pip install requests

# Install MKL and NumPy for optimized math routines
conda install -c conda-forge mkl
conda install -c conda-forge numpy

# Install DeepFace for facial analysis (if needed)
pip install deepface

# Update all installed Conda packages
conda update --all

# Install TensorFlow Keras
pip install tf-keras

# Install Loguru for logging
pip install loguru

# Install LAP for tracking acceleration
conda install -c conda-forge lap

# Install Cython and Bounding Box utilities
pip install cython_bbox

# Install OmegaConf for configuration management
pip install omegaconf
```

### Step 4: Clone ByteTrack Repository

Inside the `src` directory, clone the ByteTrack repository:

```bash
cd src
git clone https://github.com/ifzhang/ByteTrack.git
```

---

### 5. Run inference:

To run inference:

1. Navigate to `/src/inference_mediapipe.py`.
2. Add the required paths: `video_path`, `checkpoint_path`, and `output_video_path`.
3. Run the following commands in the terminal:

```
conda activate taekwondo
python inference_mediapipe.py
```



By following the steps above, you will have the necessary environment and dependencies set up for the Taekwondo project.
