# 🌸 Flower Classification using Transfer Learning

This project classifies images of flowers into 5 categories using a pre-trained MobileNetV2 model and transfer learning.

![Flower Classification](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

## 🔗 GitHub Repository
[Flower Classification Project](https://github.com/HimelDx1/flower-cloassification)

---

## 📁 Dataset

- **Source**: [TensorFlow Flower Photos Dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
- **Classes**:
  - Daisy
  - Dandelion
  - Roses
  - Sunflowers
  - Tulips
- **Size**: 3,670 total images
- **Split**: 80% training, 20% validation

Dataset was loaded using `image_dataset_from_directory()` after extraction.

---

## 🧠 Model Architecture

We used **MobileNetV2** as a base model for transfer learning.

```python
base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')
])
````

---

## 🧪 Real-Life Testing

After training, you can upload your own flower image to test the model:

```python
from google.colab import files
uploaded = files.upload()
# Predict class after resizing and normalizing the image
```

---

## 📊 Results

* ✅ Training Accuracy: \~96%
* ✅ Validation Accuracy: \~91%
* ✅ Real-life image test: Works well with uploaded flower images
* ⏱ Training time: < 3 minutes on Google Colab (free GPU)

---

## 🚀 How to Run

1. Open the notebook in Google Colab or your Jupyter environment.
2. Run all cells to train the model.
3. Upload your own flower image to test.

---

## 📜 License

This project is for academic and educational use. Dataset is publicly available under the TensorFlow datasets license.

---

```

---


