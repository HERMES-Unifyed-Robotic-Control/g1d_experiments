"""Frozen native Meta DINOv3 ViT-B/16, with the C01 patch-token interface."""
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import numpy as np


class NativeDINOv3(torch.nn.Module):
    def __init__(self, checkpoint, source, device='cuda:0', image_size=224):
        super().__init__()
        source = Path(source).resolve()
        if not (source / 'dinov3/hub/backbones.py').is_file():
            raise ValueError('Provide the official facebookresearch/dinov3 source directory')
        sys.path.insert(0, str(source))
        from dinov3.hub.backbones import dinov3_vitb16
        self.backbone = dinov3_vitb16(pretrained=False)
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.backbone.load_state_dict(weights, strict=True)
        self.backbone.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.backbone.to(device=self.device, dtype=self.dtype)
        self.image_size = image_size
        self.hidden, self.patch_size = 768, 16

    @torch.inference_mode()
    def encode_numpy(self, images, batch_size=16):
        outputs = []
        for lo in range(0, len(images), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(images[lo:lo+batch_size])).permute(0, 3, 1, 2)
            x = x.to(device=self.device, dtype=torch.float32) / 255.
            if x.shape[-2:] != (self.image_size, self.image_size):
                x = F.interpolate(x, (self.image_size, self.image_size), mode='bilinear', align_corners=False)
            mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
            std = x.new_tensor([.229, .224, .225])[None, :, None, None]
            tokens = self.backbone.forward_features(((x-mean)/std).to(self.dtype))['x_norm_patchtokens']
            if tokens.shape[1:] != (196, 768):
                raise ValueError(f'Unexpected DINO patches: {tokens.shape}')
            outputs.append(tokens.float().cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, 196, 768), np.float32)
