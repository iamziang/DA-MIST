# DA-MIST: Domain Adaptive Multiple Instance Self-Training for Intraoperative Adverse Event Detection

This repository contains the code, dataset, and visualization tools for the paper **"Domain Adaptive Multiple Instance Self-Training for Intraoperative Adverse Event Detection"**.

---

## 📂 Dataset

We release a large-scale endoscopic video dataset covering seven types of intraoperative adverse events (iAEs) across heterogeneous surgical domains.

- **Download**:
  - **Source domain (Cholec80)**: [Google Drive Link](https://drive.google.com/drive/folders/1uD6xBg4Iq8ypyDN8DbM1OgFVSX5QXGwL?usp=sharing)
  - **Target domain (dViAEs)**: [Google Drive Link](https://drive.google.com/drive/folders/10gkrhLgWkh5zdhjeVvc7rglEc6Aco2dl?usp=sharing)

- **Contents**:
  - **Source domain**: Cholec80 is re-annotated for iAEs detection from laparoscopic cholecystectomy videos.
  - **Target domain**: dViAEs comprises robot-assisted colorectal and HPB surgery videos.

- **Preprocessing**:
  1. Download all videos into the `data/` directory.
  2. Extract video frames by running:
     ```bash
     python extract_frame.py --input_dir videos/ --output_dir frames/
     ```
  3. Extract I3D feature sequences for model training:
     ```bash
     python i3d_extract.py --input_dir frames/ --output_dir features/
     ```
     
## 🛠️ Training

We adopt a two-stage training strategy to address intraoperative adverse events (iAEs) detection.

- **Stage 1: Pretraining with Multiple Instance Learning (source domain)**
  ```bash
  python stage1_main.py
  ```
- **Generate pseudo-labels for both the source and target domains. Pseudo-labels save in the `pseudo_label/` directory.
  ```bash
  python pseudo_generation.py
  ```
- **Stage 2: Self-Training for Domain Adaptation (mixed domains)**
  ```bash
  python stage2_main.py
  ```

## 📈 Visualization

We provide a visualization script to demonstrate anomaly detection results on demo videos.

- **Step 1: Download the provided pretrained weights**
  
  Please download the provided trained model weights and pretrained I3D extractor weights. Place them into the `weights/` and `extractors/weights/` directories, respectively.

- **Step 2: Run visualization on demo videos**
  
  Execute the following script to visualize anomaly scores and detection results:
  ```bash
  python visualization.py --input demo/videos/BLEED&BLUR.mp4 --output demo/results/VIS_BLEED&BLUR.mp4
  ```

