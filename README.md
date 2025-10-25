# Contrastive Medical MNIST — project_1

Brief: Implementation of a simple contrastive learning pipeline using a ResNet18 encoder on the Medical-MNIST dataset. The notebook `minst.ipynb` defines a dataset that returns two augmented views per image, trains an encoder with NT-Xent loss, and visualizes embeddings with t-SNE.

## Contents
- minst.ipynb — main notebook: dataset, encoder, training loop, visualization
- (optional) saved models / outputs

## Key components
- ContrastiveMedicalMNISTData: custom Dataset that wraps torchvision.ImageFolder and returns (x1, x2, label) with augmentations.
- Encoder: ResNet18 backbone (pretrained on ImageNet), projection head -> 64-d normalized embeddings.
- Loss: NT-Xent (normalized temperature-scaled cross entropy).
- Visualization: t-SNE of learned embeddings for selected classes.

## Requirements
- Python 3.8+
- torch
- torchvision
- matplotlib
- scikit-learn
- pandas
- (optional) CUDA for GPU acceleration

Install (example):
pip install torch torchvision matplotlib scikit-learn pandas

Or create venv on Windows:
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install torch torchvision matplotlib scikit-learn pandas

## Dataset
This notebook expects the Medical-MNIST folder layout compatible with torchvision.ImageFolder. Default path used in the notebook:
`/kaggle/input/medical-mnist-train-test-val/train`

Adjust `path` in the notebook to point to your local dataset root (e.g. `C:\data\medical-mnist\train`) before running.

## How to run
1. Open `minst.ipynb` in Jupyter / VS Code.
2. Update `path` variable to your dataset location.
3. Run cells sequentially:
   - Imports and dataset class
   - Transform definitions
   - Dataset instantiation and preview (`show_pairs`)
   - Encoder definition
   - NT-Xent loss
   - DataLoader and training loop
   - t-SNE visualization

Example: run the notebook cells or execute in an interactive environment.

## Training notes
- Default epoch count: 5, batch_size: 32, lr: 1e-3. Adjust as needed.
- Uses data augmentations (RandomResizedCrop, Flip, Rotation, ColorJitter).
- If images are single-channel, the encoder repeats channel to 3 before the backbone.

Save model example (add to notebook after training):
```python
torch.save(encoder.state_dict(), "encoder_contrastive.pth")
```

Load model example:
```python
encoder.load_state_dict(torch.load("encoder_contrastive.pth", map_location=device))
encoder.eval()
```

## Visualization
- The notebook extracts embeddings for a few samples per class and uses t-SNE to plot 2D embedding clusters.
- Adjust `perplexity`, `num_per_class`, and class selection for clearer plots.

## Tips & troubleshooting
- If dataset fails to load: confirm directory structure `root/class_x/img.png`.
- If using GPU: confirm `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
- For large datasets, increase `num_workers` in DataLoader (on Windows use caution).

## License & attribution
Project code: MIT-style (add preferred license file if needed).  
Dataset: follow the original Medical-MNIST dataset license/terms.

## Contact
For questions, open an issue in the repo or edit the notebook with more specific requirements.