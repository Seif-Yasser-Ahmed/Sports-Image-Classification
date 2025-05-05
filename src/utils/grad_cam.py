import torch
# import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Modified generate_grad_cam with custom target_layer


def generate_grad_cam(model, input_tensor, target_class=None, target_layer=None):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Default: use the last conv layer
    if target_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    forward_handle.remove()
    backward_handle.remove()

    grads = gradients[0][0].cpu().data.numpy()  # [C, H, W]
    acts = activations[0][0].cpu().data.numpy()  # [C, H, W]

    weights = np.mean(grads, axis=(1, 2))  # global average pooling
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
    cam = np.maximum(cam, 0)

    cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() != 0 else cam

    return cam

# Show Grad-CAM heatmap on image


def show_grad_cam_on_image(img_tensor, cam, title=None):
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img - img.min()
    img = img / img.max() if img.max() != 0 else img
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = cam / 255.0
    overlay = 0.5 * img + 0.5 * cam
    if not os.path.exists("gradcam"):
        os.makedirs("gradcam")
    filename = f"gradcam/{title}.png" if title else "gradcam/gradcam.png"
    plt.imsave(filename, overlay)
    # plt.close()


def run_grad_cam(model, input_tensor, target_class=None, target_layers=None):
    for i, layer in enumerate(target_layers, 1):
        cam = generate_grad_cam(model, input_tensor,
                                target_class=None, target_layer=layer)
        show_grad_cam_on_image(input_tensor[0], cam, title=f'Layer{i}')
