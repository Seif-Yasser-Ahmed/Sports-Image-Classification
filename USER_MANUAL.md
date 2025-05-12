# User Manual for Sports Image Classification

This user manual describes how to set up, run, and use the Sports Image Classification repository.

---

## 1. Prerequisites

* Python 3.8 or higher
* pip (Python package manager)
* Git
* (Optional) GPU with CUDA drivers for accelerated inference

## 2. Directory Structure

Below is the main structure of the repository:

```
Sports-Image-Classification/
├── data/               # Dataset CSVs and image folders (to be created) it should appear like this `data\sidharkal-sports-image-classification\dataset`
│   ├── train.csv       # Training labels and IDs
│   ├── test.csv        # Test image IDs
│   ├── train/          # Folder for training images
│   └── test/           # Folder for test images
├── models/             # Saved model checkpoints (.pth or .pt)
├── src/                  # Source code (training, evaluation, utilities)
│   ├── architectures/    # Architectures of all models
│   ├── gradcam/          # GradCam Results for last ran model
│   ├── utils/            # Utils directory for common files used
│   ├── notebooks/        # Secondary notebooks
│   ├── inference.py      # Inference of any model
│   └── main.ipynb        # Main Notebook
├── logs/                 # TensorBoard logs and training curves
│   ├── tensorboard logs/ # Tensorboard logs directory 
│   ├── plots/            # Plots of training/ val losse/acuracies curves
│   └── Confusion Matrix/ # For the last ran model
├── requirements.txt    # Python dependencies
├── README.md           # Project overview
└── USER_MANUAL.md      # This user manual
```

---

## 3. Installing Dependencies

1. Clone the repository and enter the directory:

   ```bash
   git clone https://github.com/Seif-Yasser-Ahmed/Sports-Image-Classification.git
   cd Sports-Image-Classification
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows
   ```

3. Install Python packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Downloading and Preparing the Dataset

The dataset is hosted on Kaggle. It contains images for seven sports categories.

1. Install the Kaggle CLI (if not already present):

   ```bash
   pip install kaggle
   ```

2. Configure Kaggle API credentials:

   * Place your `kaggle.json` file (downloaded from your Kaggle account) in `~/.kaggle/` (Linux/macOS) or `%USERPROFILE%\.kaggle\` (Windows).
   * Ensure permissions are correct: `chmod 600 ~/.kaggle/kaggle.json`.

3. Download and unzip the dataset into the `data/` folder:

   ```bash
   kaggle datasets download -d sidharkal/sports-image-classification -p data/ --unzip
   ```

4. Verify the folder structure under `data/`:

   ```bash
   data/sidharkal-sports-image-classification/dataset
   ├── train.csv
   ├── test.csv
   ├── train/      # contains image files
   └── test/       # contains image files
   ```

---

Instead you can download from kaggle from [here](https://www.kaggle.com/datasets/sidharkal/sports-image-classification)

## 5. Running Inference

Use `src/inference.py` to classify new images or batches:

```bash
python src/inference.py \
    --model-path models/best_model.pth \
    --input-dir data/test/ \
    --output-csv predictions.csv \
    --batch-size 16
```

### Inference Script Options

* `--model-run-name` (str): Run name of the trained model file


After running, `gradcam/` will contain gradcam results for each layer, and `logs/conf_matrix/` will contain the confusion matrix for the selected model.

---

## 6. Additional Scripts

* **Utilities**: Helper functions for data loading and transformations are in `src/utils.py`.
* **Initialization**: Helper functions for model and data initializations and transformations are in `src/initialize.py`.
* **Optuna**: Helper functions for optuna for hyperparameter optimization are in `src/optuna.py`.

---

## 7. Tips and Troubleshooting

* Ensure that `data/` paths are correct when invoking scripts.
* For GPU inference/training, verify that CUDA is installed and `torch.cuda.is_available()` returns `True`.
* If out-of-memory errors occur, reduce `batch_size`.
* Use TensorBoard to monitor training logs: `tensorboard --logdir logs/`.

---

## 8. Contact and Support

For questions or issues, open an issue on GitHub or reach out to:

Seif Yasser – [seiffyasserr@gmail.com](mailto:seiffyasserr@gmail.com)

Thank you for using the Sports Image Classification repository! Happy classifying.
