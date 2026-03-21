import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # Weight the activations by the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU and normalization
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx

def apply_heatmap(image_np, heatmap):
    """
    image_np: (H, W) or (H, W, 3) 0-1 float
    heatmap: (H_small, W_small) 0-1 float
    """
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        image_np = (image_np * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose
    superimposed = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
    return superimposed

def get_saliency_map(model, input_tensor):
    input_tensor.requires_grad_()
    output = model(input_tensor)
    
    score, _ = torch.max(output, dim=1)
    score.backward()
    
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency
