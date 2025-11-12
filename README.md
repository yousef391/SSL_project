# 🏥 Medical MNIST: Self-Supervised Learning & Transfer Learning (v2)

A comprehensive **self-supervised learning (SSL)** project demonstrating contrastive learning, transfer learning, and efficiency analysis on medical imaging data. This project showcases how to learn meaningful representations from unlabeled data and apply them to downstream classification tasks.

---

## 🎯 Project Goals

1. **Learn without labels**: Train a powerful encoder using only unlabeled medical images through contrastive learning
2. **Transfer to classification**: Fine-tune the learned features on a small labeled subset for medical image classification
3. **Prove SSL effectiveness**: Compare SSL-trained encoder vs. randomly initialized encoder
4. **Demonstrate efficiency**: Show that SSL works well even with minimal labeled data (1-10% of dataset)
5. **Analyze learning dynamics**: Track parameter trajectories during training to understand the learning process

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **SSL Encoder Accuracy** | 99.14% |
| **Random Encoder Accuracy** | 16.96% |
| **Improvement** | **+82.17%** (5.84x better) |
| **Best Data Efficiency** | 97.24% accuracy with only **1% of data** (471 images) |
| **Full Data Performance** | 99.14% accuracy with 100% of data |

---

## 🚀 What's New in v2

### ✨ Enhanced Features

1. **Transfer Learning Pipeline**
   - Fine-tune SSL encoder on small labeled subsets (10% of data)
   - Freeze encoder weights, train only classifier head
   - Achieve 99.14% accuracy with minimal labeled data

2. **Comprehensive Comparison**
   - SSL encoder vs. Random encoder baseline
   - Quantifies the benefit of self-supervised pretraining
   - Demonstrates 5.84x improvement over random initialization

3. **Data Efficiency Analysis**
   - Test performance with 1%, 5%, 10%, 25%, 50%, and 100% of training data
   - Shows SSL embeddings work well even with minimal data
   - Enables cost-effective medical image classification

4. **Parameter Trajectory Analysis**
   - Track parameter evolution during SSL training
   - Analyze convergence patterns
   - Foundation for dataset distillation techniques

5. **Professional Visualization Dashboard**
   - 6-panel comprehensive comparison visualization
   - Clean, organized presentation of all results
   - Summary statistics table

---

## 📂 Project Structure

```
project_1/
├── minst_v2.ipynb          # Main notebook (v2 with enhancements)
├── minst.ipynb             # Original notebook (basic contrastive learning)
├── README.md               # This file
└── Dataset/                # Medical MNIST dataset
    ├── train/              # 47,163 training images
    ├── test/               # 5,896 test images
    └── val/                # 5,895 validation images
```

---

## 🔬 Technical Approach

### Phase 1: Self-Supervised Learning (No Labels)

**Contrastive Learning (SimCLR-style)**
- Create two augmented views of each image
- Train encoder to maximize similarity between views of the same image
- Minimize similarity between views of different images
- Use NT-Xent (Normalized Temperature-scaled Cross Entropy) loss

**Architecture:**
```
Input Image (64×64×1)
    ↓
ResNet18 Backbone (pretrained on ImageNet)
    ↓
512-dimensional features
    ↓
Projection Head: 512 → 128 → 64
    ↓
Normalized 64-dimensional embeddings
```

**Training:**
- Dataset: 47,163 unlabeled medical images
- Epochs: 2-5 epochs
- Batch size: 32
- Optimizer: Adam (lr=1e-3)
- Loss: NT-Xent with temperature=0.5

### Phase 2: Transfer Learning (Small Labeled Subset)

**Fine-tuning Strategy:**
1. Freeze SSL encoder (pretrained on unlabeled data)
2. Add classifier head: 64 → 128 → 6 classes
3. Train only classifier on 10% labeled data (4,716 images)
4. Evaluate on test set

**Results:**
- Achieves 99.14% accuracy with only 10% labeled data
- Demonstrates strong transfer learning capability

### Phase 3: Analysis & Evaluation

**1. SSL vs Random Encoder Comparison**
- Train random encoder (no pretraining) on same labeled subset
- Compare final accuracies
- **Result**: SSL encoder is 5.84x better (99.14% vs 16.96%)

**2. Data Efficiency Analysis**
- Test with different data sizes: 1%, 5%, 10%, 25%, 50%, 100%
- **Key Finding**: SSL works well even with 1% of data (97.24% accuracy)

**3. Parameter Trajectory Tracking**
- Save parameter snapshots after each SSL training epoch
- Compute L2 norms and parameter changes
- Visualize learning dynamics
- **Insight**: Parameters evolve gradually, enabling trajectory matching

---

## 📈 Results & Insights

### 1. Transfer Learning Performance

| Model | Test Accuracy | Improvement |
|-------|--------------|-------------|
| **SSL Encoder (Fine-tuned)** | **99.14%** | Baseline |
| Random Encoder (Fine-tuned) | 16.96% | -82.17% |

**Key Insight**: Self-supervised pretraining provides a massive advantage, even when fine-tuning on the same amount of labeled data.

### 2. Data Efficiency

| Training Data Size | Images | Test Accuracy |
|-------------------|--------|---------------|
| 1% | 471 | **97.24%** |
| 5% | 2,358 | **99.14%** |
| 10% | 4,716 | **99.14%** |
| 25% | 11,791 | **99.14%** |
| 50% | 23,582 | **99.14%** |
| 100% | 47,163 | **99.14%** |

**Key Insight**: SSL embeddings are so powerful that even 1% of labeled data (471 images) achieves 97.24% accuracy. This demonstrates extreme data efficiency.

### 3. Parameter Trajectory Analysis

- **Parameter Norm Evolution**: Gradually increases from 97.65 to 165.56 over 5 epochs
- **Parameter Changes**: Shows convergence pattern (changes decrease over time)
- **Application**: These trajectories can be used for dataset distillation via trajectory matching

---

## 🛠️ Requirements

### Python Packages

```bash
pip install torch torchvision matplotlib scikit-learn pandas tqdm numpy
```

### System Requirements

- **Python**: 3.8+
- **PyTorch**: 1.12+ (with CUDA support recommended)
- **GPU**: Optional but recommended for faster training
- **RAM**: 8GB+ recommended
- **Storage**: ~500MB for dataset

### Dataset

Download the Medical MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/gennadiimanzhos/medical-mnist-train-test-val) and place it in the `Dataset/` directory.

**Dataset Structure:**
- 6 medical image classes: AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT
- Total: ~59,000 images (47K train, 6K test, 6K val)

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
cd project_1

# Install dependencies
pip install torch torchvision matplotlib scikit-learn pandas tqdm numpy
```

### 2. Prepare Dataset

```bash
# Download dataset from Kaggle
# Place in: project_1/Dataset/
# Structure should be:
#   Dataset/
#     ├── train/
#     │   ├── AbdomenCT/
#     │   ├── BreastMRI/
#     │   └── ...
#     ├── test/
#     └── val/
```

### 3. Run the Notebook

```bash
# Open minst_v2.ipynb in Jupyter Notebook or Google Colab
# Run all cells sequentially
```

### 4. Expected Runtime

- **SSL Training**: ~7-15 minutes (2-5 epochs, depends on GPU)
- **Fine-tuning**: ~2-3 minutes (5 epochs on 10% data)
- **Evaluation**: ~1-2 minutes
- **Total**: ~10-20 minutes

---

## 📖 Notebook Walkthrough

### Part 1: Basic Contrastive Learning (Cells 0-12)

1. **Dataset Setup**: Load Medical MNIST with augmentations
2. **Encoder Architecture**: ResNet18 + projection head
3. **NT-Xent Loss**: Contrastive learning objective
4. **Training Loop**: Train encoder on unlabeled data
5. **Visualization**: t-SNE embeddings visualization

### Part 2: Enhancements (Cells 13-31)

1. **Self-Contained Initialization**: Fresh start for enhancements
2. **SSL Training**: Train encoder from scratch (2 epochs)
3. **Fine-tuning**: Transfer learning on 10% labeled data
4. **Comparison**: SSL vs Random encoder
5. **Efficiency Analysis**: Test with different data sizes
6. **Trajectory Tracking**: Parameter evolution analysis
7. **Visualization Dashboard**: Comprehensive 6-panel comparison

---

## 🎓 Key Concepts Explained

### What is Contrastive Learning?

Contrastive learning learns representations by comparing images:
- **Positive pairs**: Two augmented views of the same image → should be similar
- **Negative pairs**: Views from different images → should be different
- **Goal**: Learn embeddings where similar images are close, different images are far

### What is Transfer Learning?

Transfer learning uses knowledge from one task to help with another:
1. **Pretraining**: Learn general features on large unlabeled dataset (SSL)
2. **Fine-tuning**: Adapt to specific task with small labeled dataset
3. **Benefit**: Achieve good performance with minimal labeled data

### Why is SSL Better than Random?

- **Random initialization**: Model starts from scratch, learns everything from labeled data
- **SSL initialization**: Model already learned useful features from unlabeled data
- **Result**: SSL needs less labeled data and achieves better performance

---

## 🔬 Scientific Contributions

1. **Demonstrates SSL effectiveness** on medical imaging with limited labels
2. **Quantifies data efficiency** showing 97%+ accuracy with only 1% labeled data
3. **Provides trajectory analysis** foundation for dataset distillation research
4. **Shows practical transfer learning** pipeline for medical image classification

---

## 📊 Visualization Dashboard

The notebook includes a comprehensive 6-panel visualization dashboard showing:

1. **SSL vs Random Encoder Comparison** (Bar chart)
2. **Data Efficiency Curve** (Line plot: accuracy vs data size)
3. **Parameter Norm Evolution** (Learning dynamics)
4. **Parameter Trajectory Changes** (Convergence analysis)
5. **SSL Training Loss** (Training progress)
6. **Summary Statistics Table** (All key metrics)

---

## 🎯 Use Cases

This project is useful for:

1. **Medical Image Classification**: Learn from unlabeled medical scans
2. **Limited Label Scenarios**: When annotation is expensive/time-consuming
3. **Transfer Learning Research**: Understanding SSL benefits
4. **Dataset Distillation**: Foundation for trajectory matching techniques
5. **Educational Purposes**: Learning SSL and transfer learning concepts

---

## 🔮 Future Enhancements

Potential improvements:

- [ ] Implement DINO-based dataset distillation
- [ ] Add more SSL methods (Barlow Twins, SwAV)
- [ ] Experiment with different encoder architectures
- [ ] Add more medical imaging datasets
- [ ] Implement online evaluation during training
- [ ] Add model checkpointing and saving

---

## 📚 References

- **SimCLR**: [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
- **Medical MNIST Dataset**: [Kaggle](https://www.kaggle.com/datasets/gennadiimanzhos/medical-mnist-train-test-val)
- **Transfer Learning**: Standard practice in deep learning
- **Dataset Distillation**: [Dataset Distillation by Matching Training Trajectories](https://arxiv.org/abs/2203.11932)

---

## 👤 Author & License

**Project**: Medical MNIST Self-Supervised Learning (v2)  
**Purpose**: Educational and research demonstration  
**License**: Open for educational use

---

## 📝 Notes

- The notebook is self-contained and can run independently
- All enhancements are initialized from scratch (no dependencies on previous cells)
- Results may vary slightly due to random initialization
- GPU recommended for faster training (CPU works but slower)

---

## 🎉 Summary

This project demonstrates a complete **self-supervised learning pipeline** for medical image analysis:

1. ✅ Learn powerful features from **47,163 unlabeled images**
2. ✅ Achieve **99.14% accuracy** with only **10% labeled data**
3. ✅ Outperform random initialization by **5.84x**
4. ✅ Work effectively with as little as **1% of data** (97.24% accuracy)
5. ✅ Provide comprehensive analysis and visualization

**Key Takeaway**: Self-supervised learning enables effective medical image classification with minimal labeled data, making it practical for real-world scenarios where annotation is expensive.

---

**Happy Learning! 🚀**
