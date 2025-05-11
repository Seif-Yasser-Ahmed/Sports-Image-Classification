import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
import os
import torch.nn.functional as F
from utils.dataset import SportsDataset
from torch.utils.data import DataLoader

from torchvision import transforms

transforms = transforms.Compose([

    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0),
        ratio=(0.75, 1.3333)
    ),

    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    # transforms.Resize((224, 224)),  # remove if RandomResizedCrop already gives 224×224
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def calc_ouput_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def initialize_data_loaders(train_csv_path, test_csv_path, dataset_path, transforms, batch_size=32):
    train_dataset = SportsDataset(
        csv_file=train_csv_path, file_path=dataset_path, split='train', transform=transforms)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = SportsDataset(
        csv_file=test_csv_path, file_path=dataset_path, split='test', transform=transforms)

    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def show_tensor_images(image_tensor):
    # Detach tensor and ensure it's on CPU, then take the first image if batch exists
    img = image_tensor.detach().cpu()

    # If the image has a single channel, treat it as grayscale
    if img.dim() == 2 or img.shape[0] == 1:
        plt.imshow(img.squeeze(), cmap='gray')
    else:
        # If image is normalized (using the usual ImageNet means and stds), de-normalize it
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)
        # Convert tensor (C, H, W) → (H, W, C) for RGB plotting
        plt.imshow(img.permute(1, 2, 0))

    plt.axis("off")
    plt.show()


def evaluate_model_with_confusion(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    print("Evaluating model...")
    # print(data_loader)
    with torch.no_grad():
        for images, labels in data_loader:
            # print(images.shape)
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * (np.array(all_preds) ==
                      np.array(all_labels)).sum() / len(all_labels)
    print(f'Accuracy: {accuracy:.2f}%')
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Save the confusion matrix plot in ../logs/conf_matrix
    # os.makedirs('../logs/conf_matrix', exist_ok=True)
    plot_path = os.path.join('../logs/conf_matrix', 'conf_matrix.png')
    plt.savefig(plot_path)
    plt.show()
    return accuracy, cm


def compute_gradcam(model, input_tensor, target_layer, target_class=None):
    features, gradients = {}, {}

    def forward_hook(module, inp, output):
        features['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    model.zero_grad()
    output.backward(gradient=one_hot)

    f_handle.remove()
    b_handle.remove()

    fmap = features['value']  # feature map
    grads_val = gradients['value']
    weights = grads_val.mean(dim=(2, 3), keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, fmap, target_class


def plot_gradcam_for_all_conv_layers(model, input_tensor, _orig_img, actual_label, _device):
    # Mapping dictionary for classes
    label_map = {0: 'Badminton', 1: 'Cricket', 2: 'Karate',
                    3: 'Soccer', 4: 'Swimming', 5: 'Tennis', 6: 'Wrestling'}

    # Gather all convolutional layers from the model
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    print(f"Found {len(conv_layers)} conv layers")

    # Unnormalize input_tensor (assumed normalized with ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    # Create directory to save gradcam images
    os.makedirs('gradcam', exist_ok=True)

    # For each conv layer, compute gradcam and plot original image, overlay, and averaged feature map
    for idx, (name, layer) in enumerate(conv_layers, 1):
        cam, fmap, pred_class = compute_gradcam(model, input_tensor, layer)

        # Map the predicted and actual label indexes to class names
        predicted_label = label_map.get(pred_class, pred_class)
        actual_label_mapped = label_map.get(
            int(actual_label), actual_label)

        # Average the feature map across channels for visualization
        fmap_avg = fmap.mean(dim=1, keepdim=True).squeeze().cpu().numpy()
        fmap_avg = np.atleast_2d(fmap_avg)
        fmap_avg = (fmap_avg - fmap_avg.min()) / \
            (fmap_avg.max() - fmap_avg.min() + 1e-8)

        # Create a new figure for each layer
        plt.figure(figsize=(15, 5))

        # Original Image
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(inp)
        ax1.set_title(f'Layer: {name}')
        ax1.axis("off")

        # GradCAM Overlay
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(inp)
        ax2.imshow(cam, cmap='jet', alpha=0.5)
        ax2.set_title(
            f'Predicted: {predicted_label} | Actual: {actual_label_mapped}')
        ax2.axis("off")

        # Averaged Feature Map
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(fmap_avg, cmap='viridis')
        ax3.set_title("Averaged Feature Map")
        ax3.axis("off")

        plt.tight_layout()
        # Save figure with layer number in filename
        filename = os.path.join('gradcam', f'gradcam_layer_{idx}.png')
        plt.savefig(filename)
        plt.close()


def load_trained_model(model, model_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location=device)
    # Check if the checkpoint is a dictionary with extra keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict=state_dict)
    model.to(device)
    print("Pretrained weights loaded into ResNet34 successfully.")
    return model
