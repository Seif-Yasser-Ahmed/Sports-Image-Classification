# Sports Image Classification

## Table of Contents

* [About](#about)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Training](#training)
* [Evaluation](#evaluation)
* [Pretrained Models](#pretrained-models)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



## About

This repository contains all the code and resources for training and evaluating convolutional neural network models on the Sports Image Classification dataset. The dataset comprises images from seven sports categories:

* Cricket
* Wrestling
* Tennis
* Badminton
* Soccer
* Swimming
* Karate

The goal is to build a robust model that can accurately classify images into these categories.

## Dataset

The dataset is organized into training and testing splits. Each image is associated with a class label in `train.csv`. To obtain the data, you can use the Kaggle CLI:

### Download Dataset
#### Windows
```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\download_dataset.ps1 -Slug sidharkal/sports-image-classification
```

#### Linux
```bash
./download_dataset.sh sidharkal/sports-image-classification
```

#### Kaggle

```bash
pip install kaggle  # if you haven’t already
kaggle datasets download -d sidharkal/sports-image-classification -p data/ --unzip
```

After downloading, your `data/` directory should look like:

```
data/
├── train.csv
├── test.csv
├── train/      # training images
└── test/       # test images
```


## Project Structure

```
├── data/                 # Dataset CSVs and image folders
├── models/               # Saved model checkpoints
│   ├── graphs/           # Models Computational Graphs
│   └── checkpoints/
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
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
├── USER_MANUAL.md      # This user manual
└── README.md             # Project overview and instructions
```

## Installation

1. **Clone the repository**

```bash
git clone [https://github.com/Seif-Yasser-Ahmed/Sports-Image-Classification.git](https://github.com/Seif-Yasser-Ahmed/Sports-Image-Classification.git)
cd Sports-Image-Classification
```


2. **Install dependencies**

```bash
pip install -r requirements.txt
````

## Usage

### Download the Dataset

Use the Kaggle CLI as shown above to download and unzip the dataset into the `data/` folder.


### Evaluating a Model

After training, evaluate performance on the test set:

```bash
python src/inference.py \
    --model_run_name 
```

This will print classification accuracy, a confusion matrix, Save GradCam, and other metrics.

## Pretrained Models

The `models/` directory contains pretrained checkpoints. Download the latest checkpoint or train from scratch as above.

## Results

* **Top-1 Accuracy**: 95.2%
* **Confusion Matrix**: See `logs/` for visualizations after running `inference.py`


### Tensorboard
in the root dir
```bash
tensorboard --logdir=logs
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
