# Landmark Classification & Tagging for Social Media 2.0

##  Project Overview

This project builds upon the Udacity Deep Learning Nanodegree initiative to automatically determine the geographic location of a photo by identifying **landmarks depicted in the image**. It addresses the common issue of missing EXIF metadata by using computer vision and deep learning techniques to classify landmarks and infer image location.

Key aims include:
- Improving user experience for photo-sharing platforms when GPS metadata is not available.
- Demonstrating a full machine learning pipeline: from data preprocessing to model deployment.

##  Project Structure

The repository is organized as follows:

```
├── app.ipynb                   # Interactive demo/app that uses the best-performing model
├── cnn_from_scratch.ipynb      # Notebook for building and evaluating CNN from scratch
├── transfer_learning.ipynb     # Notebook demonstrating a transfer learning approach
└── src/                        # Core Python modules & unit tests
    └── …                       
```

- **`cnn_from_scratch.ipynb`**  
  Loads and preprocesses the dataset, defines and trains a custom CNN architecture, and visualizes decision-making around model design.

- **`transfer_learning.ipynb`**  
  Utilizes pre-trained CNNs (e.g., ResNet variants) to fine-tune for landmark classification. Includes comparative analysis and model selection rationale.

- **`app.ipynb`**  
  A user interface for testing the model: upload an image and get predicted landmark tags. Demonstrates real-world applicability of your model.

- **`src/` folder**  
  Contains utility functions and corresponding test cases to modularize dataset handling, model architectures, training, and evaluation processes.

##  Getting Started

### Prerequisites

- Python 3.7+  
- Recommended: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for managing Python environments.

### Installation

```bash
# Clone the repository
git clone https://github.com/zeinabalzarkan/Landmark-Classification-Tagging-for-Social-Media-2.0.git
cd Landmark-Classification-Tagging-for-Social-Media-2.0/

# (Optional) create and activate conda environment
conda create --name landmark python=3.7
conda activate landmark

# Install required dependencies
pip install -r requirements.txt
```

### Verify GPU Setup (optional)

If you’re using an NVIDIA GPU and have PyTorch configured:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

If it returns `False`, confirm your GPU and drivers are installed correctly, or switch to a CPU-only environment.

### Launching Notebooks

Run one of the following to start:

```bash
jupyter notebook cnn_from_scratch.ipynb
jupyter notebook transfer_learning.ipynb
jupyter notebook app.ipynb
```

### Running Python Modules (optional)

If `src/` includes executable modules or scripts:

```bash
python -m src.<module_name>
```

Replace `<module_name>` with the actual script name.

##  Dataset

The project uses a curated subset of the **Google Landmarks Dataset v2 (GLDv2)**, offering thousands of labeled landmark images drawn from a wide global distribution ([arxiv.org](https://arxiv.org/abs/2004.01804?utm_source=chatgpt.com)). 

If the dataset isn't included in the repo, please download it from the official source and place it in the appropriate directory as described in the notebooks.

##  Results & Performance

- **CNN from Scratch**: Custom-designed CNN achieved [**~52% accuracy**] (adjust with actual number if different).
- **Transfer Learning**: Fine-tuning a pre-trained model (e.g., ResNet50) improved performance to [**~66% accuracy**].

_(You can include a comparison table or performance visuals here.)_

##  Usage

1. Open **`app.ipynb`** in Jupyter.
2. Upload an image with a recognizable landmark.
3. The model will output the top predicted landmark tags along with confidence scores.
4. (Optional) Visualize predictions or integrate into a web/front-end application.

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Implement changes and add tests where applicable.
4. Submit a pull request once your changes are ready.

##  License & Acknowledgments

- This work was completed as part of the **Udacity Machine Learning Fundamentals (AWS AI & ML Scholarship)** program.  
- Inspired by similar implementations in Udacity’s Nanodegree community.  
- The dataset used is the **Google Landmarks Dataset v2**, licensed as described in its documentation.

