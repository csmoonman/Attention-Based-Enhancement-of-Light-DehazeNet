# Attention-Based-Enhancement-of-Light-DehazeNet
Re-implemented Light-DehazeNet (TIP 2021) and integrated a CBAM attention mechanism with residual connections.

# Attention-Enhanced Light-DehazeNet

[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

This repository contains a PyTorch implementation of an enhanced **Light-DehazeNet**, based on the original paper: ["Light-DehazeNet: A Novel Lightweight CNN Architecture for Single Image Dehazing" (TIP 2021)](https://ieeexplore.ieee.org/abstract/document/9562276).

This project successfully replicates the original architecture and introduces a significant enhancement by integrating the **Convolutional Block Attention Module (CBAM)**. This addition allows the network to adaptively focus on critical haze-relevant features, leading to superior dehazing performance in terms of both quantitative metrics and visual quality.

## Key Enhancements

- **CBAM Integration:** Seamlessly integrated the CBAM attention mechanism into the multi-level feature fusion path to enhance the model's feature representation capabilities.
- **Residual Connections:** Introduced residual connections within the CBAM module to prevent gradient vanishing, ensuring stable training and effective performance gains.
- **Optimized Architecture:** The final model (`lightdehazeNetCBAMv2.py`) is fine-tuned to balance performance and efficiency.

## Performance

The enhanced model demonstrates significant improvements over the original baseline on the benchmark dataset.

- **Peak Signal-to-Noise Ratio (PSNR):** **+1.0 dB**
- **Structural Similarity (SSIM):** **+0.052**

### Visual Results

Here is a comparison of the results on a sample image from the dataset.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/csmoonman/Attention-Based-Enhancement-of-Light-DehazeNet
    cd Attention-Based-Enhancement-of-Light-DehazeNet
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    This project requires Python 3.7+ and the following packages. You can install them using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Training

To train the model from scratch, use the `run_experimentCBAM.py` script.

```bash
python run_experimentCBAM.py \
    --train_hazy ./data/training_images \
    --train_original ./data/original_images \
    --epochs 60 \
    --learning_rate 0.0001
```
- Trained model weights will be saved in the `trained_weights/` directory.
- Visual snapshots from the validation set will be saved in `training_data_captures/`.

### Inference

To perform dehazing on a single image or a directory of images, use the `muliple_test_inferenceCBAMv2.py` script.

1.  **Place your trained model weights** (e.g., `trained_LDNetCBAM.pth`) in the `trained_weights/` directory.
2.  **Update the model path** in `inferenceCBAMv2.py` to point to your weights file.
3.  **Run the inference script:**

    ```bash
    # To test a directory of hazy images
    python muliple_test_inferenceCBAMv2.py --test_directory ./path/to/your/hazy_images

    # To evaluate with ground truth images and get PSNR/SSIM scores
    python muliple_test_inferenceCBAMv2.py \
        --test_directory ./path/to/your/hazy_images \
        --gt_directory ./path/to/your/ground_truth_images
    ```
- Dehazed images will be saved in the `vis_results/` directory.

## Acknowledgments

- This work is built upon the foundation of the original [Light-DehazeNet](https://github.com/H-deep/Light-DehazeNet) paper and implementation.
- The attention mechanism is inspired by the [CBAM](https://github.com/luuuyi/CBAM.PyTorch) paper.

##  License

This project is licensed under the MIT License. See the `LICENSE` file for details.
