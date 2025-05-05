import argparse
import os
import sys
import torch
import importlib
from PIL import Image
from torchvision import transforms

# Define default paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', 'models', 'checkpoints')
)

# Mapping from model argument name to class names
CLASS_MAP = {
    'custom': 'CustomCNN',
    'alex': 'AlexNet',
    'resnet18': 'ResNet18',
    'resnet50': 'ResNet50',
    'googlenet': 'GoogLeNet',
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',
}


def load_model(model_name, checkpoint_path, device='cpu'):
    """
    Dynamically imports the model class and loads the checkpoint.

    Args:
        model_name (str): Name of the model architecture to load.
        checkpoint_path (str): Full path to the model checkpoint file.
        device (str): Device to load the model onto ('cpu', 'cuda').

    Returns:
        model: The loaded PyTorch model. None if loading fails.
    """
    print(f"Attempting to load model '{model_name}' from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(
            f"Error: Checkpoint file not found at {checkpoint_path}", file=sys.stderr)
        return None

    try:
        # Dynamically import the architecture module
        module = importlib.import_module(f"architectures.{model_name}")
        class_name = CLASS_MAP.get(model_name)
        if class_name is None:
            print(
                f"Error: No class mapping found for model '{model_name}'", file=sys.stderr)
            return None
        ModelClass = getattr(module, class_name)
        model = ModelClass()

        # Load checkpoint
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model '{model_name}' loaded successfully onto {device}.")
        return model

    except Exception as e:
        print(f"Error loading model {model_name}: {e}", file=sys.stderr)
        return None


def preprocess_input(input_path):
    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"Error reading image: {e}", file=sys.stderr)
        return None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    tensor = transform(img)
    tensor = tensor.unsqueeze(0)  # add batch dimension
    return tensor
    # pass


def perform_inference(model, input_data, device):
    with torch.no_grad():
        # Ensure the input data is on the correct device
        input_tensor = input_data.to(device) if hasattr(
            input_data, "to") else input_data
        # Run the inference
        output = model(input_tensor)
    return output
    # pass


def postprocess_output(model, image_tensor, target_layers=None, target_class=None):
    from utils.grad_cam import run_grad_cam

    print(
        f"\033[32mRunning Grad-CAM on model: {model.__class__.__name__}\033[0m")
    print(f"Target class: {target_class}")
    run_grad_cam(model, image_tensor.to('cuda'), target_class=target_class,
                 target_layers=target_layers)
    # pass


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_name',
        type=str,
        choices=list(CLASS_MAP.keys()),
        default='custom',
        help='Model architecture to use.'
    )
    parser.add_argument(
        '--training_method',
        type=str,
        choices=['naive', 'transfer'],
        default='naive',
        help='Training method used.'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'SGD', 'RMSProp'],
        default='adam',
        help='Optimizer used during training.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        choices=[0.01, 0.001],
        default=0.01,
        help='Learning rate used during training.'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['None', 'step', 'cosine'],
        default='None',
        help='LR scheduler used during training.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        choices=[16, 32, 64],
        default=32,
        help='Batch size used during training.'
    )
    parser.add_argument(
        '--loss_fn',
        type=str,
        default='crossentropy',
        help='Loss function used during training.'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay used during training.'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to the input data (e.g., image file, text file, csv).'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'gpu'],
        help='Device to run inference on (cpu or cuda/gpu).'
    )

    args = parser.parse_args()

    # Build composite checkpoint name string
    string = "seif"
    # string.capitalize()
    if args.weight_decay == 0.0:
        args.weight_decay = '0'
    composite_name = f"model{str(args.model_name).capitalize()}_{str(args.training_method).capitalize()}_{str(args.optimizer)}_{str(args.lr).capitalize()}_{str(args.scheduler).capitalize()}_{args.batch_size}_{args.loss_fn}_wd{args.weight_decay}_epoch_100"

    # Search for checkpoint file
    checkpoint_filename = composite_name + '.pth'
    checkpoint_path = os.path.join(DEFAULT_CHECKPOINT_DIR, checkpoint_filename)
    if not os.path.isfile(checkpoint_path):
        print(
            f"Error: Checkpoint '{checkpoint_filename}' not found in {DEFAULT_CHECKPOINT_DIR}", file=sys.stderr)
        sys.exit(1)

    # Resolve device alias
    device = 'cuda' if args.device in ['cuda', 'gpu'] else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to CPU.", file=sys.stderr)
        device = 'cpu'

    print(f"--- Inference Configurations ---")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Input Path: {args.input_path}")
    print("-----------------------------------")

    # Run inference pipeline
    model = load_model(args.model_name, checkpoint_path, device)
    if model is None:
        sys.exit(1)

    input_data = preprocess_input(args.input_path)
    if input_data is None:
        sys.exit(1)

    if args.model_name == 'custom':
        target_layers = [
            model.conv1,
            model.conv2,
            model.conv3,

            model.conv4
        ]
    output = perform_inference(model, input_data, device)
    if output is None:
        sys.exit(1)

    results = postprocess_output(
        model, input_data, target_layers=target_layers)

    print("--- Inference Results ---")
    print(results)


if __name__ == "__main__":
    main()
