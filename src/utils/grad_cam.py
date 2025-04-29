import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
EXAMPLE USAGE: 

```cam = generate_grad_cam(model, input_tensor)
show_grad_cam_on_image(input_tensor[0], cam)```

"""


def get_last_conv_layer(model):
    # Example for ResNet
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def generate_grad_cam(model, input_tensor, target_class=None):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hook into last conv layer
    last_conv = get_last_conv_layer(model)
    forward_handle = last_conv.register_forward_hook(forward_hook)
    backward_handle = last_conv.register_backward_hook(backward_hook)

    # Forward
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    loss = output[0, target_class]
    loss.backward()

    # Clean up
    forward_handle.remove()
    backward_handle.remove()

    grads = gradients[0][0].cpu().data.numpy()
    acts = activations[0][0].cpu().data.numpy()

    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
    cam = cam - cam.min()
    cam = cam / cam.max()

    return cam


def show_grad_cam_on_image(img_tensor, cam):
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img - img.min()
    img = img / img.max()
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = cam / 255.0
    overlay = 0.5 * img + 0.5 * cam
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()
