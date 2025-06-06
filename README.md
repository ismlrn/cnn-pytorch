# CNN with PyTorch for FashionMNIST

This is a simple Convolutional Neural Network (CNN) built with **PyTorch** to classify images from the **FashionMNIST** dataset.

The model achieves around **90% accuracy** on the test set after just 5 training epochs.

---

## Model Architecture
  - `Input` 1x28x28 grayscale image
  - `Conv2D` (1→16), ReLU, MaxPool
  - `Conv2D` (16→32), ReLU, MaxPool
  - `Flatten`
  - `Linear` (32×7×7 → 128), ReLU
  - `Linear` (128 → 10 logits)
