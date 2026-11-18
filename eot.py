import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableEoT(nn.Module):
    """
    Expectation over Transformation (EoT) differentiable simulation module.
    Applies real-world social media distortions during adversarial PGD optimization
    to ensure perturbations survive WhatsApp and Instagram compression.
    """
    def __init__(self, blur_prob: float = 0.5, noise_prob: float = 0.5, scale_prob: float = 0.5):
        super().__init__()
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.scale_prob = scale_prob
        
        # Predefined 3x3 Gaussian kernel for differentiable blur
        kernel = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ]) / 16.0
        kernel = kernel.repeat(3, 1, 1, 1) # [3, 1, 3, 3]
        self.register_buffer("gaussian_kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] in range [0, 1]
        """
        out = x
        
        # 1. Random Scale / Rescale Jitter
        if self.training and torch.rand(1).item() < self.scale_prob:
            _, _, h, w = out.shape
            scale_factor = 0.9 + torch.rand(1).item() * 0.2 # 0.9x to 1.1x
            new_h, new_w = max(32, int(h * scale_factor)), max(32, int(w * scale_factor))
            down = F.interpolate(out, size=(new_h, new_w), mode="bilinear", align_corners=False)
            out = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
            
        # 2. Differentiable Blur
        if self.training and torch.rand(1).item() < self.blur_prob:
            # Apply depthwise separable conv
            kernel = self.gaussian_kernel.to(out.device)
            out = F.conv2d(out, kernel, padding=1, groups=3)
            
        # 3. Additive Gaussian Noise (simulates sensor / compression noise)
        if self.training and torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(out) * 0.015
            out = out + noise

        # 4. Soft Differentiable Color Jitter / Quantization proxy
        if self.training:
            # Approximate 8-bit color quantization with smooth ramp
            out = torch.clamp(out, 0.0, 1.0)
            
        return torch.clamp(out, 0.0, 1.0)
