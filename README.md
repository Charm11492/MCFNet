# 
## Beyond Conventional Vision: 
RGB-Event Fusion for Robust Object Detection in Dynamic Traffic Scenarios
## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Visualization](#visualization)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)

## 🛠 Installation

### Requirements

```bash
# Create conda environment
conda create -n MCFNet python=3.10
conda activate MCFNet

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch >= 2.1.1
- CUDA >= 11.8
- mamba-ssm
- opencv-python
- loguru
- einops
- grad-cam
- thop

### Mamba Installation

```bash
# Install causal-conv1d and mamba-ssm
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

## 📊 Dataset Preparation
download DSEC: (https://dsec.ifi.uzh.ch/)
download PKU-DAVIS-SOD (https://www.pkuml.org/research/pku-davis-sod-dataset.html)
### Event Representation Options

This project supports multiple event representations:

1. **IWE (Image of Warped Events)** - ECM generated
2. **Voxel Grid** --pretreatment.py
3. **Event Timestamps** --pretreatment.py 
4. **Event Frames**  --pretreatment.py

### DSEC Dataset Structure

```
datasets/
├── train_8class_960*1280.json
├── test_8class_960*1280.json
└── DSEC/
    ├── train/
    │   └── images/
    │       ├── zurich_city_09_a/
    │       ├── interlaken_00_b/
    │       └── ...
    ├── iwe/  # IWE or other event representations
    │   ├── zurich_city_09_a/
    │   ├── interlaken_00_b/
    │   └── ...
    ├── voxel_flow/  # Voxel representations
    │   ├── zurich_city_09_a/
    │   ├── interlaken_00_b/
    │   └── ...
    └── events/  # Raw event data
        ├── zurich_city_09_a/
        ├── interlaken_00_b/
        └── ...
```

### Data Configuration

Update paths in `yolox_DSEC.py`:

```
self.data_dir = '/path/to/your/datasets'
self.root_event_dir = '/path/to/DSEC/iwe'
self.root_img_dir = '/path/to/DSEC/train/images'
```
## 📁 Project Structure

```
commtr/
├── datasets/
│       ├── test_8class_960_1280.json   #data annotation
│       ├── ......
├── joint/                    # Joint training pipeline(ECM+CMM)
├──── tools/                  
│       ├── train.py          # training
│       ├── config/           # Flow model configurations
│─────yoolox/ 
│       ├─── models/          # Detection model (EDUM+CMM)
├── tools/
│   ├── eval.py               # Evaluation script
│   ├── train.py              # Training script
├── yolox/
│   ├── models/               # Model architectures
│   ├── data/                 # Data loading utilities
│   └── utils/                # Utility functions
├── exps/
│   └── example/yolox_voc/
│       └── yolox_DSEC.py     # DSEC experiment config
└── requirements.txt
```
## 🏋️ Training

### Stage 1: Joint Training(optional)
To use IWE event representations, joint training in sub dataset must be performed first. In `~/join/exps/example/yolox_DSEC.py`, use 'train_sub_8class.json' to train, and 'test_sub_8class.json'to eval.
However, this process is more complicated, and we also recommend using pretreatment.py or other publicly available methods to generate other event representations as network inputs, such as voxels, timestamps, etc.
```bash
cd joint/
python -m yolox.tools.train.py -n  yolox-s -c yolox_s.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```

#### Stage 2: Generate IWE Representations (optional)

After joint training, generate IWE representations, 

```bash
python -m yolox.tools.eval -n  yolox-s -c [**.pth] -b 1 -d 1 --conf 0.001 --fp16 --fuse
```

#### Stage 3: Train the main detection model:
In `~/exps/example/yolox_DSEC.py`, use IWE and RGB Image as inputs of network, and use 'datasets/train_8class_960_1280.json or datasets/train_2class_960_1280.json' to train, and 'datasets/test_8class_960_1280.json or datasets/test_2class_960_1280.json'to eval.
```bash
cd ..
python3 -m yolox.tools.train \ -n yolox-m \ -d 2 \ -b 4 \ -f exps/example/yolox_voc/yolox_DSEC.py \ -c /path/to/pretrained/yolox_m.pth \ -o
```

### Training Parameters

- `-n`: Model name (yolox-s, yolox-m, yolox-l, yolox-x)
- `-d`: Number of GPUs
- `-b`: Batch size
- `-f`: Experiment configuration file
- `-c`: Pretrained checkpoint path
- `-o`: Use multiple GPUs

### Custom Configuration

Modify `yolox_DSEC.py` for custom settings:

```python
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 8  # Number of object classes
        self.depth = 0.67     # Model depth multiplier
        self.width = 0.75     # Model width multiplier
        self.warmup_epochs = 1
        
        # Data augmentation
        self.mosaic_prob = 0.9
        self.mixup_prob = 0.5
        self.hsv_prob = 0
        self.flip_prob = 0
```

## 📈 Evaluation

### Standard Evaluation
The weight file we provide for verification is 'val.pth'
```bash
python3 -m yolox.tools.eval \-n yolox-m \ -b 2 \-d 1 \-f exps/example/yolox_voc/yolox_DSEC.py \-c /path/to/checkpoint.pth \--conf 0.001 \--fp16 \--fuse
```

### Evaluation Parameters

- `--conf`: Confidence threshold
- `--nms`: NMS threshold
- `--fp16`: Use mixed precision
- `--fuse`: Fuse conv and batch norm layers


## 📝 Citation


## 🙏 Acknowledgments

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for the base detection framework
- [Mamba](https://github.com/state-spaces/mamba) for the state space model implementation
- [DSEC Dataset](https://dsec.ifi.uzh.ch/) for providing the event camera dataset
- [PKU-DAVIS-SOD] (https://www.pkuml.org/research/pku-davis-sod-dataset.html) for providing the event camera dataset
- The event-based vision community for inspiration and support