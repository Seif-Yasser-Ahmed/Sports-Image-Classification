import os
import random
import sys
import argparse
import torch
import importlib
import torch.nn as nn
from PIL import Image
# from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from utils.custom import CustomCNN
from utils.utils import compute_gradcam, plot_gradcam_for_all_conv_layers, load_trained_model, initialize_data_loaders, evaluate_model_with_confusion, transforms
from utils.dataset import SportsDataset
# Define default paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', 'models', 'checkpoints'))
# C:\Users\Seif Yasser\Desktop\DL\Project\Sports-Image-Classification\data\sidharkal-sports-image-classification\dataset\train
DATASET_PATH = '../data/sidharkal-sports-image-classification/dataset'
TEST_CSV_PATH = '../data/sidharkal-sports-image-classification/dataset/test.csv'
TRAIN_CSV_PATH = '../data/sidharkal-sports-image-classification/dataset/train.csv'
# Mapping from model key to class names
CLASS_MAP = {
    'custom': 'CustomCNN',
    'alex': 'AlexNet',
    'resnet18': 'ResNet18',
    'resnet34': 'ResNet34',
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'resnet152': 'ResNet152',
    'googlenet': 'GoogLeNet',
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',
}


def perform_gradcam(model, train_csv_path, dataset_path, device):
    train_dataset = SportsDataset(
        csv_file=train_csv_path, file_path=dataset_path, split='train', transform=transforms)
    train_idx = random.randint(0, len(train_dataset) - 1)
    train_img, train_label = train_dataset[train_idx]
    # input tensor must have batch dim
    input_tensor = train_img.unsqueeze(0).to(device)
    # Convert tensor image (C, H, W) to numpy (H, W, C) for display and clip to [0, 1]
    orig_train_img = np.transpose(train_img.cpu().numpy(), (1, 2, 0))
    orig_train_img = np.clip(orig_train_img, 0, 1)

    print("GradCAM for each conv layer on a random Train image:")

    plot_gradcam_for_all_conv_layers(
        model, input_tensor, orig_train_img, train_label, device)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using a pretrained model from a run name.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "run_name",
        type=str,
        help="The run name of the trained model (e.g. modelCustom_Naive_adam_0.01_None_32_crossentropy_wd0_epoch_100)."
    )
    args = parser.parse_args()

    run_name = args.run_name
    print("--- Loading Checkpoint ---")
    if run_name.startswith("Custom"):
        checkpoint_filename = 'model' + run_name + '_epoch_100.pth'
    else:
        checkpoint_filename = 'model' + run_name + '_epoch_7.pth'
    checkpoint_path = os.path.join(DEFAULT_CHECKPOINT_DIR, checkpoint_filename)
    if not os.path.isfile(checkpoint_path):
        print(
            f"Error: Checkpoint '{checkpoint_filename}' not found in {DEFAULT_CHECKPOINT_DIR}", file=sys.stderr)
        sys.exit(1)

    # print("--- Loading Checkpoint ---")
    # Extract model key from run_name
    # if not run_name.startswith("model"):
    #     print("Error: Run name does not start with 'model'", file=sys.stderr)
    #     sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("--- Inference Configurations ---")
    print(f"Model Name: {run_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("-----------------------------------")

    class_model = run_name.split('_')[0]
    # model = CLASS_MAP.get(model_key)

    print("--- Loading Model ---")
    # Use a hard-coded input image path (modify as needed)
    from utils.custom import CustomCNN
    if class_model == 'Custom':
        class_model = CustomCNN
        class_model = class_model(num_classes=7)

    elif class_model == 'ResNet50':
        from torchvision.models import resnet50
        class_model = resnet50

    elif class_model == 'AlexNet':
        from torchvision.models import alexnet
        class_model = alexnet

    elif class_model == 'VGG16':
        from torchvision.models import vgg16
        class_model = vgg16

    elif class_model == 'VGG19':
        from torchvision.models import vgg19
        class_model = vgg19

    elif class_model == 'GoogLeNet':
        from torchvision.models import googlenet
        class_model = googlenet

    elif class_model == 'ResNet18':
        from torchvision.models import resnet18
        class_model = resnet18

    elif class_model == 'ResNet34':
        from torchvision.models import resnet34
        class_model = resnet34

    elif class_model == 'ResNet101':
        from torchvision.models import resnet101
        class_model = resnet101

    elif class_model == 'ResNet152':
        from torchvision.models import resnet152
        class_model = resnet152

    else:
        # Dynamically import the model class from the module
        try:
            module = importlib.import_module(
                f"models.{class_model.lower()}")
            class_model = getattr(module, CLASS_MAP[class_model])
        except ImportError as e:
            print(f"Error importing model: {e}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(class_model, CustomCNN):
        if class_model.__name__ == 'alexnet':
            class_model = class_model(pretrained=False)
            in_features = class_model.classifier[-1].in_features
            class_model.classifier[-1] = nn.Linear(in_features, 7)
        elif 'resnet' in class_model.__name__.lower():
            class_model = class_model(pretrained=False)
            in_features = class_model.fc.in_features
            class_model.fc = nn.Linear(in_features, 7)
        elif class_model.__name__ == 'googlenet':
            class_model = class_model(pretrained=False)
            in_features = class_model.fc.in_features
            class_model.fc = nn.Linear(in_features, 7)
        else:
            class_model = class_model(pretrained=False)
            in_features = class_model.classifier[-1].in_features
            class_model.classifier[-1] = nn.Linear(in_features, 7)

    model = load_trained_model(class_model, checkpoint_path, device)
    if model is None:
        sys.exit(1)
    print("--- Model Loaded Successfully ---")

    print("-----------------------------------")

    # default_input = os.path.join(SCRIPT_DIR, "sample.jpg")
    print("--- Test Set Evaluation ---")
    # (val_loader was created previously in initialize_all; if not, we recreate it)
    _, test_loader = initialize_data_loaders(
        TRAIN_CSV_PATH, TEST_CSV_PATH, DATASET_PATH, transforms, batch_size=32)
    # train_loader, val_loader = initialize_data_loaders(train_csv_path, test_csv_path, dataset_path, transforms,batch_size=batch_size)
    # print(test_loader)
    test_acc, test_cm = evaluate_model_with_confusion(
        model, test_loader, device)

    train_dataset = SportsDataset(
        csv_file=TRAIN_CSV_PATH, file_path=DATASET_PATH, split='train', transform=transforms)
    train_idx = random.randint(0, len(train_dataset) - 1)
    train_img, train_label = train_dataset[train_idx]
    # input tensor must have batch dim
    input_tensor = train_img.unsqueeze(0).to(device)
    # Convert tensor image (C, H, W) to numpy (H, W, C) for display and clip to [0, 1]
    orig_train_img = np.transpose(train_img.cpu().numpy(), (1, 2, 0))
    orig_train_img = np.clip(orig_train_img, 0, 1)

    print("--- Saving GradCAM for each conv layer on a random Train image ---")
    perform_gradcam(model.to('cpu'), train_csv_path=TRAIN_CSV_PATH,
                    dataset_path=DATASET_PATH, device='cpu')
    print("--- Inference Results ---")
    # print(output)


if __name__ == "__main__":
    main()
