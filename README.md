# Contrastive Medical MNIST — project_1

A simple **contrastive learning** pipeline using a ResNet18 encoder on the Medical-MNIST dataset.  
The notebook `minst.ipynb` creates two augmented views per image, trains an encoder with **NT-Xent loss**, and visualizes embeddings with **t-SNE**.

---

## 📂 Contents
- `minst.ipynb` — main notebook: dataset, encoder, training, visualization  
- dataset link : [dataset](https://www.kaggle.com/datasets/gennadiimanzhos/medical-mnist-train-test-val)

---

## ⚡ Key components
- **Dataset:** `ContrastiveMedicalMNISTData` returns `(x1, x2, label)` with augmentations.  
- **Encoder:** ResNet18 backbone (pretrained), projection head → 64-d normalized embeddings.  
- **Loss:** NT-Xent (contrastive loss).  
- **Visualization:** t-SNE of embeddings for selected classes.  

---

## 🛠 Requirements
- Python 3.8+  
- torch, torchvision, matplotlib, scikit-learn, pandas  
- Optional: CUDA for GPU acceleration  

Install dependencies:
```bash
pip install torch torchvision matplotlib scikit-learn pandas
