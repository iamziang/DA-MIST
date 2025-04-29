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

