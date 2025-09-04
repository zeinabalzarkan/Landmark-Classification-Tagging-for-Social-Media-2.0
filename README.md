# 🏞️ Landmark Classification & Tagging for Social Media 2.0

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)
![Jupyter](https://img.shields.io/badge/notebooks-Jupyter-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

This repository demonstrates a **deep learning workflow** for automatic **landmark recognition and tagging**, designed to enhance photo-sharing platforms by identifying geographic locations when GPS metadata is missing.

---

## 🚀 Project Overview

A complete pipeline to:

1. **Preprocess Image Data** for model training.  
2. **Train Models** using both a CNN from scratch and Transfer Learning.  
3. **Classify Landmarks** from new/unseen images.  
4. **Deploy the Model** for interactive use in a simple application.  

---

## 📂 Repository Structure

```
.
├── app.ipynb                   # Interactive demo app for landmark predictions
├── cnn_from_scratch.ipynb      # Notebook for custom CNN architecture & training
├── transfer_learning.ipynb     # Notebook using pre-trained model (e.g., VGG16/DenseNet)
├── src/                        # Helper modules & unit tests
│   └── ...
└── README.md                   # This document
```

---

## 🛠️ Getting Started

### Prerequisites:
- Python 3.7+  
- [Jupyter Notebook](https://jupyter.org/install)  
- PyTorch with GPU support (optional but recommended)  

### Installation

```bash
# Clone the repository
git clone https://github.com/zeinabalzarkan/Landmark-Classification-Tagging-for-Social-Media-2.0.git
cd Landmark-Classification-Tagging-for-Social-Media-2.0/

# (Optional) create and activate conda environment
conda create --name landmark python=3.7
conda activate landmark

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Launch the notebooks

```bash
jupyter notebook cnn_from_scratch.ipynb
jupyter notebook transfer_learning.ipynb
jupyter notebook app.ipynb
```

### Example Workflow

1. Open **`app.ipynb`**.  
2. Upload a landmark image (e.g., Eiffel Tower).  
3. The model will return top predicted landmark tags with confidence scores.  

---

## ✅ Example Results

- **CNN from Scratch**  
  Achieved ~52% accuracy on validation set.  

- **Transfer Learning (VGG16/DenseNet)**  
  Achieved ~66% accuracy with fine-tuning.  

*(Adjust with your actual results if different.)*

---

## 📖 References

- [Google Landmarks Dataset v2](https://arxiv.org/abs/2004.01804)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
- [Udacity Machine Learning Fundamentals](https://www.udacity.com/)  

---

## 👩‍💻 Author

Created by **Zeinab AlZarkan** as part of the Udacity AWS Machine Learning Fundamentals Scholarship program.  
