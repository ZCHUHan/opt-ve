# Monkey Verifier

## Requirements
- CUDA 11.8.0
- Ubuntu 20.04

## Installation

### 1. Create Conda Environment

```bash
conda create -n robomonkey python=3.10 -y
conda activate robomonkey
```

### 2. Set Up Verifier & LLaVA

```bash
git clone https://github.com/robomonkey-vla/monkey-verifier
cd monkey-verifier/llava_setup
git clone https://github.com/robomonkey-vla/LLaVA
cd LLaVA
git apply < ../fix_llava_padding.patch
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install ninja
```

### 3. Install Dependencies

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed==0.9.3
pip install peft==0.4.0
pip install transformers==4.31.0
pip install bitsandbytes==0.41.0
pip install datasets
pip install wandb
pip install "numpy<2"
```

### 4. Download Model Weights

```bash
cd ~/monkey-verifier/model_dir
git clone https://huggingface.co/zhiqings/LLaVA-RLHF-7b-v1.5-224
```

### 5. Download Data

```bash
cd ~/monkey-verifier/data_dir
git clone https://huggingface.co/datasets/robomonkey-vla/bridge_data_v2
unzip bridge_v2_images.zip

cd ~/monkey-verifier/data_dir
git clone https://huggingface.co/datasets/robomonkey-vla/action_preference_bridge
```

## Training

Run the training script:

```bash
cd ~/monkey-verifier/RLHF
bash scripts/7b-v1.5-224/full_bridge_finetune.sh
```

## Acknowledgements

Built on [LLaVA-RLHF](https://llava-rlhf.github.io/) by Sun et al.
