# Robust Object Detection Under Adverse Conditions

This project builds a highly robust object detection system designed to perform under challenging real-world conditions such as fog, snow, glare, nighttime, and occlusion. It is inspired by the perception challenges in autonomous vehicles like Waymo and Zoox.

## Project Structure
- `configs/`: Training configs (YAML/JSON).
- `data/`: Data loaders, augmentation transforms, and utilities.
- `models/`: Wrapper or custom model code.
- `train/`: Training scripts and loss function definitions.
- `evaluate/`: Evaluation metrics and visualization tools.
- `scripts/`: Data downloading, formatting, and conversion utilities.
- `notebooks/`: EDA and experiment notebooks.
- `outputs/`: Saved checkpoints, logs, and result images.

## Quick Start
```bash
git clone <your-repo-url>
cd robust_object_detection
pip install -r requirements.txt
python main.py --mode train --config configs/default_config.yaml
```
