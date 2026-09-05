"""Grad-CAM visualizations for model interpretability."""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def _find_target_layer(model):
    """Find the last convolutional layer in DenseNet121."""
    # DenseNet features are in model.net.features
    return model.net.features


def generate_cam(model, input_tensor, target_idx=None):
    """Generate Grad-CAM heatmap for a single input."""
    target_layer = _find_target_layer(model)

    # Register forward and backward hooks
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    h_activ = target_layer.register_forward_hook(forward_hook)
    h_back = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    model.eval()
    logits = model(input_tensor)
    if target_idx is None:
        target_idx = torch.argmax(logits).item()
    loss = logits[:, target_idx]

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Compute Grad-CAM
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

    return heatmap, target_idx


def overlay_cam_on_image(img_path, heatmap, alpha: float = 0.4):
    """Overlay Grad-CAM heatmap on original X-ray image."""
    img = np.array(Image.open(img_path).convert("RGB"))
    img = cv2.resize(img, (224, 224))

    # Color map for heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Superimpose
    superimposed = heatmap_color * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return superimposed, heatmap_resized


def visualize_prediction(model, image_path, class_names=CLASSES,
                         top_k: int = 3, device="cpu"):
    """Visualize prediction with Grad-CAM overlays."""
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Top K classes
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_classes = [(class_names[i], probs[i]) for i in top_idx]

    # Generate CAM for top class
    heatmap, target_idx = generate_cam(model, img_tensor, target_idx=top_idx[0])

    # Overlay on image
    overlay, heatmap_resized = overlay_cam_on_image(
        image_path, heatmap, alpha=0.35)

    # Plot
    fig, axes = plt.subplots(1, top_k + 1, figsize=(5 * (top_k + 1), 5))

    # Original image with CAM
    axes[0].imshow(overlay)
    axes[0].set_title(f"Grad-CAM: {class_names[target_idx]}")
    axes[0].axis("off")

    # Probability bar chart
    y_pos = range(top_k)
    probs_top = [probs[i] for i in top_idx]
    axes[1].barh(y_pos, probs_top, align="center")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([class_names[i] for i in top_idx])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Probability")

    # Show heatmap
    axes[2].imshow(heatmap_resized, cmap="jet")
    axes[2].set_title("Grad-CAM Heatmap")
    axes[2].axis("off")

    plt.tight_layout()
    return fig, top_classes