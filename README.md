# 🔬 CNN Interpretability Lab

An interactive CNN learning experience that explains how convolutional networks process images, from input tensor to softmax output. This repo includes:
- a **live static web walkthrough** (`cnn_lab_ultimate.html`) for students
- a **Python + Streamlit + PyTorch** lab stack for deeper experimentation

## 🌐 Live Student Version (No Setup Required)

- **GitHub Pages App:** https://radhika-verma06.github.io/CNN_UltimateGuide/cnn_lab_ultimate.html
- Students can open this link directly in any browser (no clone, no install).

![CNN Lab Showcase](preview.png)
> *Note: Upload a preview.png of the lab to your repo for this image to render on GitHub.*

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

### Run the Interactive HTML Version Locally

```bash
python3 -m http.server 8000
```

Then open:

`http://localhost:8000/cnn_lab_ultimate.html`

---

## 📂 Project Structure
```text
cnn_lab/
├── cnn_lab_ultimate.html # Final interactive student-facing CNN walkthrough
├── cnn_explainer_v3.html # Earlier v3 visual version
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


## 💡 What I Learned Building This

- **Visualizing Black Boxes**: Bridging the gap between raw tensor math and intuitive visualizations requires careful UI mapping (e.g., building a live `requestAnimationFrame` loop to animate flatten operations).
- **Interactive Educational Tools**: Moving from static charts to an editable convolution kernel significantly deepens user engagement.
- **Client-Side CNN Simulation**: Writing a mini-inference engine purely in Javascript (performing convolution, pooling, and dense passes) was a great exercise in understanding tensor operations from the ground up without relying on PyTorch/TensorFlow.

---

**Tags:** `machine-learning`, `cnn`, `interactive`, `visualization`, `vanilla-js`, `education`, `portfolio`
