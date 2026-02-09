<div align="center">

<h1 align="center"><strong>EasyTune: Efficient Step-Aware Fine-Tuning for Diffusion-Based Motion Generation</strong></h1>
  <p align="center">
   <a href='https://xiaofeng-tan.github.io/' target='_blank'>Xiaofeng Tan*<sup>1</sup></a>&emsp;
   Wanjiang Weng*<sup>1</sup>&emsp;
   Haodong Lei<sup>1</sup>&emsp;
   Hongsong Wang<sup>&dagger;1</sup>&emsp;
    <br>
    <sup>1</sup>PALM Lab, Southeast University&emsp;
    <br>
    *Equal contribution &emsp;
    <sup>&dagger;</sup>Corresponding author
    <br>
    For any questions, please contact Xiaofeng Tan (xiaofengtan@seu.edu.cn) or Wanjiang Weng (wjweng@seu.edu.cn).

  </p>
</p>

<p align="center">
  <a href="https://iclr.cc/virtual/2026/poster/">
    <img src="https://img.shields.io/badge/ICLR-2026-9065CA" alt="ICLR 2026">
  </a>
  <a href="https://arxiv.org/abs/">
    <img src="https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow" alt="Paper PDF on arXiv">
  </a>
  <a href="https://xiaofeng-tan.github.io/projects/EasyTune/">
    <img src="https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green" alt="Project Page">
  </a>
</p>

</div>

> **TL;DR:** We propose **EasyTune**, a reinforcement fine-tuning framework for diffusion models that decouples recursive dependencies and enables (1) dense and effective optimization, (2) memory-efficient training, and (3) fine-grained alignment.

This repository offers the official implementation for the paper. 

## News

- **[2025/12]** Training and evaluation code has been released!
- **[2025/12]** Pre-trained model weights have been released!
- **[2025/01]** **EasyTune** has been officially accepted by *ICLR 2026*!

## Plan

- [x] Release paper.
- [ ] Release training code.
- [ ] Release evaluation code.
- [ ] Release pre-trained model weights.
- [x] Release environment guidance.

## Model Zoo

<table>
  <tr>
    <th>Model</th>
    <th>Dataset</th>
    <th>Download</th>
    <th>Description</th>
  </tr>
  <tr>
    <td rowspan="2"> Fine-tuned Step-Aware Preference Model (SPM)</td>
    <td>HumanML3D</td>
    <td>
      <a href="https://1drv.ms/">OneDrive</a> /
      <a href="https://pan.baidu.com/">BaiduNetDisk</a>
    </td>
    <td>Text-motion retrieval reward model</td>
  </tr>
  <tr>
    <td>KIT-ML</td>
    <td>
      <a href="https://1drv.ms/">OneDrive</a> /
      <a href="https://pan.baidu.com/">BaiduNetDisk</a>
    </td>
    <td>Text-motion retrieval reward model</td>
  </tr>
  <tr>
    <td rowspan="2">Fine-tuned MLD</td>
    <td>HumanML3D</td>
    <td>
      <a href="https://1drv.ms/">OneDrive</a> /
      <a href="https://pan.baidu.com/">BaiduNetDisk</a>
    </td>
    <td>MLD fine-tuned with EasyTune</td>
  </tr>
  
</table>

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n easytune python=3.10 -y
conda activate easytune
```

### 2. Install Dependencies

```bash
# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Prepare Pre-trained Models

Download and place the following models in the `deps/` directory:

```
deps/
├── sentence-t5-large/       # Sentence-T5 for text encoding
├── clip-vit-large-patch14/  # CLIP model
├── distilbert-base-uncased/
├── glove/                   # GloVe word embeddings
├── smpl/                    # SMPL model
└── t2m/                     # Text-to-Motion evaluation model
```

### 4. Prepare Dataset

Download the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) or [KIT-ML](https://motion-annotation.humanoids.kit.edu/dataset/) dataset and place in the `datasets/` directory:

```
datasets/
└── humanml3d/
    ├── new_joint_vecs/
    ├── new_joints/
    ├── texts/
    └── ...
```

## Training

### Step 1: Train Step-Aware Preference Model (SPM)

The SPM learns a step-aware text-motion alignment reward signal used to guide fine-tuning.

```bash
# Train SPM on HumanML3D
bash run.sh 0 spm

# Or run directly with custom parameters
CUDA_VISIBLE_DEVICES=0 python -m train_spm \
    --cfg configs/spm_h3d.yaml \
    --NoiseThr 0.5 \
    --maxT 1000 \
    --step_aware M1T0
```

**Key Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--NoiseThr` | Noise threshold (>1 = pure motion only) | 0.5 |
| `--maxT` | Maximum diffusion timestep | 1000 |
| `--step_aware` | Step-aware mode: `M0T0`, `M1T0`, `M0T1`, `M1T1` | `M0T0` |
| `--CLThr` | Contrastive learning threshold | 0.9 |
| `--CLTemp` | Contrastive learning temperature | 0.1 |
| `--finetune` | Checkpoint to fine-tune from (optional) | `None` |

Checkpoints are saved to `checkpoints/spm/`.

### Step 2: Fine-tune MLD with EasyTune

Fine-tune the pre-trained MLD motion generation model using the trained SPM as a reward model.

```bash
# Fine-tune with baseline methods (ReFL, DRaFT, AlignProp, DRTune)
bash run.sh 0 mld

# Or run directly
CUDA_VISIBLE_DEVICES=0 python -m ft_mld \
    --cfg configs/ft_mld_t2m.yaml \
    --spm_path your_spm_checkpoint.pth \
    --ft_type ReFL \
    --ft_m 20 \
    --ft_lambda_reward 1.0

# Fine-tune with EasyTune (our method)
CUDA_VISIBLE_DEVICES=0 python -m ft_mld_chain \
    --cfg configs/ft_mld_t2m.yaml \
    --spm_path your_spm_checkpoint.pth \
    --ft_type EZtune \
    --ft_k 10 \
    --ft_lambda_reward 1.0
```

**Key Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--spm_path` | SPM checkpoint filename (in `checkpoints/spm/`) | Required |
| `--ft_type` | Fine-tuning method: `ReFL`, `DRaFT`, `DRTune`, `AlignProp`, `EZtune` | Required |
| `--ft_m` | Number of denoising steps for ReFL/DRTune | 1 |
| `--ft_k` | Number of gradient-enabled steps for EZtune/DRaFT | 1 |
| `--ft_lambda_reward` | Reward loss weight | 1.0 |

## Evaluation

### Evaluate Fine-tuned Model

```bash
bash run.sh 0 eval

# Or run directly
CUDA_VISIBLE_DEVICES=0 python -m test \
    --cfg configs/ft_mld_t2m.yaml \
    --mld_path /path/to/finetuned_checkpoint.ckpt
```

### Evaluate TMR Retrieval

```bash
bash run.sh 0 tmr
```

## Project Structure

```
EasyTune/
├── configs/                    # Configuration files
│   ├── ft_mld_t2m.yaml        # Fine-tuning config (HumanML3D)
│   ├── ft_mld_kit.yaml        # Fine-tuning config (KIT-ML)
│   ├── spm_h3d.yaml           # SPM training config (HumanML3D)
│   ├── spm_kit.yaml           # SPM training config (KIT-ML)
│   ├── mld_t2m.yaml           # MLD base config
│   └── modules/               # Module-specific configs
├── GradGuidance/               # Core EasyTune module
│   ├── spm.py                  # Step-Aware Preference Model
│   ├── finetune_config.py      # Fine-tuning strategy configs
│   ├── utils.py                # Utility functions
│   └── opt/                    # Optimized transformer components
│       ├── attention.py
│       ├── embeddings.py
│       └── position_encoding.py
├── mld/                        # MLD base model
│   ├── config.py               # Configuration parsing
│   ├── data/                   # Data loading and processing
│   ├── models/                 # Model architectures
│   ├── utils/                  # Utility functions
│   └── transforms/             # Data transformations
├── train_spm.py                # SPM training script
├── train_mld.py                # MLD pre-training script
├── ft_mld.py                   # Baseline fine-tuning script
├── ft_mld_chain.py             # EasyTune fine-tuning script
├── eval_tmr.py                 # TMR retrieval evaluation
├── test.py                     # Model evaluation entry point
├── run.sh                      # Unified run script
└── requirements.txt            # Python dependencies
```

## Acknowledgements

This codebase builds upon the following excellent projects:

- [MLD](https://github.com/ChenFengYe/motion-latent-diffusion) - Motion Latent Diffusion
- [MotionLCM](https://github.com/Dai-Wenxun/MotionLCM) - Motion Latent Consistency Model
- [Sentence-T5](https://huggingface.co/sentence-transformers/sentence-t5-large) - Text Encoder

We thank the authors for their contributions to the community.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{tan2026easytune,
  title={EasyTune: Efficient Step-Aware Fine-Tuning for Diffusion-Based Motion Generation},
  author={Xiaofeng Tan and Wanjiang Weng and Haodong Lei and Hongsong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
