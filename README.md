# 🩺 Diabetic Retinopathy Detection

An AI-powered web application that detects the stage of Diabetic Retinopathy from retinal fundus images using a custom Deep Learning CNN model.

🔗 **Live Demo**: [huggingface.co/spaces/chashvith/bio-proj](https://huggingface.co/spaces/chashvith/bio-proj)

---

## - What is Diabetic Retinopathy?

Diabetic Retinopathy is a diabetes complication that affects the eyes. It's caused by damage to blood vessels in the retina and is one of the leading causes of blindness worldwide. Early detection is critical for preventing vision loss.

---

## - Model Architecture

Built a **custom CNN from scratch** using TensorFlow/Keras:

```
Input (224x224x3)
    → Conv2D(32, 3x3, ReLU)
    → MaxPooling2D(2x2)
    → Conv2D(64, 3x3, ReLU)
    → MaxPooling2D(2x2)
    → Flatten
    → Dense(128, ReLU)
    → Dense(5, Softmax)
```

### Training Details
| Parameter | Value |
|-----------|-------|
| Input Size | 224 x 224 |
| Batch Size | 32 |
| Epochs | 3 |
| Train/Val Split | 80% / 20% |
| Preprocessing | Gaussian filtered images, rescaled to [0,1] |
| Loss | Categorical Crossentropy |
| Output | 5 classes (Softmax) |

---

## - Classes

| Class | Description |
|-------|-------------|
| No_DR | No Diabetic Retinopathy |
| Mild | Mild NPDR |
| Moderate | Moderate NPDR |
| Severe | Severe NPDR |
| Proliferate_DR | Proliferative DR (most advanced) |

---

## - Dataset

- **Source**: [Kaggle - Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- **Preprocessing**: Gaussian filtered images
- **Model hosted on**: HuggingFace Spaces (287 MB)

---

## - Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pillow
- NumPy
- HuggingFace Spaces (deployment)

---

## - Run Locally

```bash
# Clone the repo
git clone https://github.com/chashvith/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

# Install dependencies
pip install -r requirements.txt

# Download the model from HuggingFace and place it in the same folder
# Then run
streamlit run app.py
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified ophthalmologist for eye health concerns.

---

## 👨‍💻 Author

- HuggingFace: [huggingface.co/chashvith](https://huggingface.co/chashvith)
- GitHub: [github.com/chashvith](https://github.com/chashvith)
