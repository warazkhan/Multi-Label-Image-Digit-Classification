# Multi-Label Image-Based Digit Classification

This project aims to develop a deep learning model for **multi-label digit classification** using images containing multiple overlapping digits. The methodology involves **image preprocessing**, **feature extraction**, **CNN model training**, and **evaluation**.

## 📂 Dataset Overview
Each image in the dataset contains **three MNIST digits**, requiring the model to detect and classify multiple digits within a single image. The digits may be **overlapping** or partially **obscured**, increasing classification difficulty.

---

## 🛠️ Project Workflow

### 1. **Data Loading & Preprocessing**
- **Resize**: Images are resized to a uniform dimension (28x28).
- **Grayscale Conversion**: Convert images to grayscale for consistency.
- **Gaussian Blurring**: Applied to smooth images and reduce noise.
- **Thresholding**: Convert the image into binary to simplify digit detection.
- **Morphological Operations**: Erosion and dilation help separate overlapping digits.
- **Region of Interest (ROI) Segmentation**: Extracts individual digits from the image.
- **Normalization**: Pixel values are normalized for better model performance.

### 2. **Model Development**
- **CNN Architecture**: A convolutional neural network (CNN) is used, with the ability to adapt and experiment with different architectures to improve accuracy.
- **Data Augmentation**: Used to introduce variations and enhance model generalization.

### 3. **Evaluation & Hyperparameter Tuning**
- **Cross-Validation**: n-fold cross-validation ensures robust model evaluation.
- **Hyperparameter Tuning**: Grid search is used to optimize hyperparameters for the best performance.
- **Early Stopping**: Prevents overfitting by monitoring performance on the validation set.

### 4. **Performance Metrics**
- **Accuracy**: Measures the overall performance.
- **Precision, Recall, and F1-Score**: Additional metrics used to evaluate model performance, especially in multi-label scenarios.

---

## 📈 Model Performance Summary

Two different model configurations were tested, as shown in the following data:

| **Metric**              | **Model 0 (Epoch 11)**          | **Model 1 (Epoch 65)**          |
|-------------------------|---------------------------------|---------------------------------|
| **Train Loss**           | 0.000692                        | 0.03086                         |
| **Train Accuracy**       | 99.98%                          | 99.08%                          |
| **Train Precision**      | 99.98%                          | 99.18%                          |
| **Train Recall**         | 99.98%                          | 99.02%                          |
| **Validation Loss**      | 0.00049                         | 0.0153                          |
| **Validation Accuracy**  | 99.99%                          | 99.49%                          |
| **Validation Precision** | 99.99%                          | 99.54%                          |
| **Validation Recall**    | 99.99%                          | 99.47%                          |
| **Test Loss**            | 0.00027                         | 0.00111                         |
| **Test Accuracy**        | 99.99%                          | 99.97%                          |
| **Test Precision**       | 99.99%                          | 99.97%                          |
| **Test Recall**          | 99.99%                          | 99.97%                          |
| **Test F1 Score**        | 99.99%                          | 99.97%                          |

---

### Key Observations:
- The model performed exceptionally well with **very high accuracy**, **precision**, **recall**, and **F1-score** across training, validation, and testing datasets.
- **Model 0 (Epoch 11)** showed slightly better performance in terms of **train and validation accuracy**, **loss**, and **precision/recall**, compared to **Model 1 (Epoch 65)**.

---

## 🔧 Installation

To set up the environment, install the required dependencies:

```bash
pip install h5py scikeras opencv-python keras tensorflow numpy pandas matplotlib seaborn scikit-learn
