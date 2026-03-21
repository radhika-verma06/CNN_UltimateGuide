# 🔬 CNN Interpretability Lab

A premium, interactive AI interpretability tool designed to peel back the curtain on Convolutional Neural Networks. Built with **Python**, **Streamlit**, and **PyTorch**, this project demonstrates how a CNN "sees" and processes Fashion-MNIST images.

![Project Preview Placeholder](https://via.placeholder.com/800x400.png?text=CNN+Interpretability+Lab+Dashboard)

## 🌟 Key Features

### 1. **Dual-Lens Perspective**
- **Lab Mode (Educational)**: Detailed deep-dive into a 2-layer CNN trained on Fashion-MNIST. Perfect for learning how filters detect edges and textures.
- **Pro Mode (Generalist)**: Leverages a pretrained **MobileNet-V3** to classify and explain 1,000+ real-world object categories from any uploaded image.

### 2. **Explainable AI (XAI)**

### 2. **Live Feature Extraction**
- **Activation Maps**: Real-time visualization of all 16 filters in Conv-1 and 32 filters in Conv-2.
- **Hooked Architecture**: Captures intermediate activations during the forward pass.

### 3. **Interactive Convolution Lab**
- **Math Walkthrough**: Slide a 3x3 kernel over your input image and see the dot-product calculations update live.
- **ReLU Visualization**: Understand how activation functions suppress noise and highlight features.

### 4. **Cinematic UI & Experience**
- **Premium Aesthetics**: High-end typography (Syne), noise textures, and glassmorphic elements.
- **Visual Pipeline**: A horizontal track visualizing the data flow across all layers.
- **Interactive Math**: A high-fidelity "Filter Explorer" with real-time dot product and ReLU calculation.

---

## 🛠 Tech Stack
- **Backend:** PyTorch, Torchvision
- **Frontend:** Streamlit, Custom CSS
- **Visuals:** Matplotlib, Plotly, OpenCV
- **Data:** Fashion-MNIST

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/cnn-interpretability-lab.git
   cd cnn-interpretability-lab
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (optional):**
   *(The project includes a pre-trained weights script)*
   ```bash
   python train_model.py
   ```

4. **Launch the Lab:**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure
```text
cnn_lab/
├── app.py              # Main Streamlit entrance
├── model.py            # CNN Architecture & Preprocessing
├── interpretability.py # Grad-CAM & Saliency logic
├── visualization.py    # Plotting & CSS helpers
├── train_model.py      # Training script
├── requirements.txt    # Dependencies
└── fashion_mnist_cnn.pth # Saved weights
```

## 🧠 Future Improvements
- [ ] Support for CIFAR-10 (Color images)
- [ ] Real-time Webcam inference
- [ ] Lime / SHAP integration for feature attribution comparison
- [ ] Exportable PDF reports for model analysis

---

*Developed by Radhika // 2026*
