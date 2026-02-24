# 🖼️ image-classification-6-categories

This project builds a **Convolutional Neural Network (CNN)** to classify images into **six distinct categories**.  
It demonstrates a complete **deep learning pipeline**, including data extraction, preprocessing, model training, evaluation, and prediction.

---

## 📌 Project Overview

- **Problem Type:** Multi-class image classification  
- **Domain:** Computer Vision / Deep Learning  
- **Number of Classes:** 6  
- **Model Used:** Convolutional Neural Network (CNN)  
- **Framework:** TensorFlow / Keras  

---

## 📂 Dataset

- **Source:** Provided ZIP dataset  
- **Metadata File:** `images.csv`  
- **Content:** Images mapped to labels across six categories  

### Dataset Workflow
- Extracted ZIP dataset
- Verified folder structure
- Displayed sample images from each category

---

## 🔧 Data Preprocessing

- Resized images to **128 × 128**
- Normalized pixel values to **[0,1]**
- Converted labels to numerical format
- Split dataset into:
  - **Training:** 70%
  - **Validation:** 15%
  - **Testing:** 15%

---

## 🧠 Model Architecture

The CNN model includes:

- Convolution layer (ReLU)
- MaxPooling layer
- Convolution layer
- MaxPooling layer
- Flatten layer
- Dense hidden layer
- Softmax output layer

```python
Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Softmax
````

---

## 📈 Model Training

* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam
* **Batch Size:** 32
* **Epochs:** 10

Training and validation accuracy/loss were visualized to monitor learning.

---

## 📊 Model Evaluation

* **Test Accuracy:** **77.26%**
* Generated confusion matrix
* Displayed sample predictions
* Evaluated classification performance across categories

---

## 🔍 Model Prediction

The trained model predicts image categories using softmax probabilities.

### Example Workflow

* Load image
* Resize & normalize
* Run inference
* Output predicted label

---

## 💾 Model Saving & Loading

The trained model was saved for reuse:

```python
model.save("image_classification_model.h5")
```

Then reloaded for inference on new images.

---

## 🛠️ Tech Stack

| Tool                 | Purpose              |
| -------------------- | -------------------- |
| Python               | Programming          |
| TensorFlow / Keras   | Deep learning        |
| OpenCV               | Image preprocessing  |
| NumPy                | Numerical operations |
| Pandas               | Data handling        |
| Matplotlib / Seaborn | Visualization        |
| Scikit-learn         | Evaluation metrics   |

---

## 🚀 How to Run

```bash
git clone https://github.com/SyedHussain23/image-classification-6-categories
cd image-classification-6-categories
pip install -r requirements.txt
jupyter notebook image-classification-6-categories.ipynb
```

---

## 🔮 Future Improvements

* Transfer learning (VGG16, ResNet, EfficientNet)
* Hyperparameter tuning
* Advanced augmentation
* Larger dataset
* Class imbalance handling
* Deeper CNN architectures

---

## 👨‍💻 Author

**Syed Hussain Abdul Hakeem**

* LinkedIn: [https://www.linkedin.com/in/syed-hussain-abdul-hakeem](https://www.linkedin.com/in/syed-hussain-abdul-hakeem)
* GitHub: [https://github.com/SyedHussain23](https://github.com/SyedHussain23)

---

## 📄 License

This project is open-source and available under the MIT License.

---

⭐ **If you found this project useful, consider giving it a star!**
