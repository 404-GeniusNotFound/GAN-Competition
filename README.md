# GAN Challenge: Learning to Generate "Noise" 🎨🤖

![Status](https://img.shields.io/badge/Status-Completed-success) ![Rank](https://img.shields.io/badge/Private_LB-5th_%2F_110-gold) ![Rank](https://img.shields.io/badge/Public_LB-7th_%2F_110-silver) ![Metric](https://img.shields.io/badge/Metric-FID_Score-blue) ![Framework](https://img.shields.io/badge/PyTorch-GAN-red)

## 🏆 Achievements
**Finished 5th on the Private Leaderboard** and **7th on the Public Leaderboard** out of **110 participants** (550 total submissions).

* **Competition:** [GAN Competition (GAN Challenge)](https://kaggle.com/competitions/test-gan-competition)
* **Host:** Nitin Kumar Jha
* **Participant:** Sathvik V (Roll No: 22f2001468)

## 📖 Overview
This project was developed for the GAN Challenge competition. The objective was to train a Generative Adversarial Network (GAN) to generate realistic **32×32 images**.

**The Twist:** Unlike standard datasets (like faces or animals), the training data consisted of **58,578 images of abstract noise patterns** (see samples below). My goal was to make a model that could learn the specific statistical distribution of this "chaos" and generate new, indistinguishable noise samples.

## 🖼️ The Data: "Pure Noise"
The dataset was provided as a collection of zipped JSONL shards. Each record contained:
* **Base64 Encoded Image:** Required decoding from text to bytes.
* **Metadata:** EXIF rotation, inversion flags (`invert: true/false`), and alpha masks.
* **Visuals:** The "real" images looked like white noise, static, or random pixel distributions.

![Real Data Samples](image_7bd547.png)
*Figure 1: Samples from the training set. The model had to learn to replicate these specific noise textures.*

## 🛠️ Methodology

### 1. Robust Data Pipeline
I built a custom `ImageShardDataset` to handle the complex raw data:
* **Decoding:** Parsed JSONL files and decoded Base64 strings on the fly.
* **Standardization:** Applied EXIF rotations, handled 16-bit grayscale conversions, and cropped images based on alpha masks.
* **Normalization:** Resized all inputs to **32x32** and normalized pixel values to `[-1, 1]` for stable GAN training.

### 2. Architecture: WGAN-GP
Standard DCGANs often suffer from mode collapse. To generate high-quality noise distributions, I implemented a **Wasserstein GAN with Gradient Penalty (WGAN-GP)**.

* **Generator:**
    * Input: Random latent vector ($z=100$).
    * Layers: `ConvTranspose2d` for upsampling.
    * **Innovation:** Replaced `BatchNorm2d` with **`InstanceNorm2d`**, which proved more stable for this specific texture-generation task.
    * Activation: `ReLU` (hidden) and `Tanh` (output).

* **Discriminator (Critic):**
    * Architecture: Strided `Conv2d` layers for downsampling.
    * **No Sigmoid:** The critic outputs a raw "realness" score (Wasserstein distance), not a probability.
    * **Gradient Penalty:** Enforced the Lipschitz constraint to stabilize training and prevent vanishing gradients.

### 3. Training & Optimization
* **Optimizer:** Adam (`lr=0.0001`, `beta1=0.5`, `beta2=0.9`).
* **Critic Updates:** Trained the critic **5 times** for every 1 generator step to ensure the Wasserstein distance estimate was accurate.
* **Loss Function:** Wasserstein Loss + Gradient Penalty ($\lambda=10$).

### 4. Evaluation (FID Score)
The competition used **Fréchet Inception Distance (FID)** to evaluate performance.
1.  Generated **1,000** new images using the trained Generator.
2.  Passed them through a pre-trained **Inception-V3** model (pool3 layer).
3.  Extracted **2048-dimensional feature vectors** for submission.
4.  Minimizing the FID score meant my generated "noise" was statistically indistinguishable from the real "noise."

## 🚀 How to Run
1.  Clone the repository.
2.  Install dependencies: `torch`, `torchvision`, `Pillow`, `pandas`, `numpy`.
3.  Download the competition shards into an `input/` folder.
4.  Run the notebook `22f2001468-gan-challenge.ipynb`.

```bash
# Example: Generate samples after training
# This will create submission.csv with Inception features
python generate_submission.py
