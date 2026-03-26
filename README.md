# CNN Ultimate Guide 🧠

An interactive, visual guide to **Convolutional Neural Networks** — covering architecture, training, generalization, and regularization. Built as a single self-contained HTML file with zero dependencies.

## ✨ Features

### CNN Architecture (Interactive)
- **Input Image** — toggle between sample patterns (shoe, bag, shirt) displayed as 28×28 pixel grids
- **Convolution** — animated kernel scan with switchable filters (Edge, Blur, Sharpen) and live dot-product readout
- **Max Pooling** — step-through animation showing 2×2 window selection
- **Flatten** — animated 2D-to-1D vector transformation
- **Dense Layers** — neural network forward-pass visualization
- **Softmax Output** — prediction bars linked to the selected input

### Training Loop
- Step-by-step pipeline: Forward Pass → Loss → Backprop → Weight Update → Repeat
- Live weight update animation
- Progressive epoch training with animated loss/accuracy chart
- Toggle to show train vs. validation curves

### Bias vs Variance
- Interactive 3-mode switcher (Underfit / Good Fit / Overfit) with live chart updates
- **Dartboard analogy** — 4 SVG boards visually demonstrating bias-variance combinations
- Each board has a "Re-throw" button for re-randomization

### Dropout Regularization
- 24-neuron grid with Training / Inference mode toggle
- Adjustable dropout rate slider (0–70%)
- Visual comparison: with vs. without dropout

### Extras
- 6-question interactive quiz with instant feedback
- FAQ accordion with key CNN concepts
- Animated stat counters (parameters, epochs, accuracy, loss)
- Fixed navbar with section links + scroll progress bar
- Section reveal-on-scroll animations
- Fully responsive (desktop + mobile)

## 🚀 Quick Start

Just open the file in any browser:

```
[open cnn_lab_ultimate.html](https://radhika-verma06.github.io/CNN_UltimateGuide/cnn_lab_ultimate.html)
```

No server, no install, no dependencies needed.

## 🛠 Tech

- **Zero dependencies** — pure HTML + CSS + JavaScript
- Single file (~80KB)
- CSS custom properties, glassmorphism cards, gradient palette
- Canvas + SVG for all visualizations
- Intersection Observer for scroll animations

## 📚 Topics Covered

| Concept | Section |
|---|---|
| Convolution & Filters | Conv |
| Feature Maps & ReLU | Conv |
| Max Pooling | Pool |
| Flattening | Flatten |
| Fully Connected Layers | Dense |
| Softmax & Probabilities | Output |
| Training Loop & Epochs | Training |
| Loss & Backpropagation | Training |
| Bias vs Variance | Bias/Var |
| Overfitting & Underfitting | Bias/Var |
| Dropout Regularization | Dropout |

## Tags

`CNN` `Deep Learning` `Neural Networks` `Interactive` `Visualization` `Education` `Training` `Bias-Variance` `Dropout` `Regularization` `HTML` `CSS` `JavaScript` `No Dependencies`

---

*Built as an interactive deep learning concept visualizer.*
