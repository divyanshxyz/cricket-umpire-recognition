"""Phase 6 Visual Feature Extractor (FUSE-02).

Extracts 576-D appearance embedding vectors from image crops using
a pretrained MobileNetV3-Small backbone with an Identity classifier.
Provides 10% context-padded bounding-box cropping with automatic
fallback to center-crops on degenerate/missing boxes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class VisualFeatureExtractor:
    """Extracts 576-D visual embeddings from BGR image crops using MobileNetV3-Small."""

    def __init__(self, weights: str = "DEFAULT", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = models.mobilenet_v3_small(weights=weights)
        self.model.classifier = nn.Identity()
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def _center_crop(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        return image[y0 : y0 + side, x0 : x0 + side]

    def crop_image(
        self,
        image_bgr: np.ndarray,
        bbox: Optional[Union[Sequence[float], Sequence[int]]] = None,
        pad_fraction: float = 0.10,
    ) -> np.ndarray:
        """Crop image with context padding, falling back to center-crop if bbox is missing/degenerate."""
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        if bbox is None or len(bbox) < 4:
            return self._center_crop(image_bgr)

        x0, y0, x1, y1 = bbox[:4]
        bw = x1 - x0
        bh = y1 - y0

        if bw <= 0 or bh <= 0:
            return self._center_crop(image_bgr)

        pad_x = pad_fraction * bw
        pad_y = pad_fraction * bh

        cx0 = max(0, int(round(x0 - pad_x)))
        cy0 = max(0, int(round(y0 - pad_y)))
        cx1 = min(w, int(round(x1 + pad_x)))
        cy1 = min(h, int(round(y1 + pad_y)))

        if cx1 <= cx0 or cy1 <= cy0:
            return self._center_crop(image_bgr)

        crop = image_bgr[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            return self._center_crop(image_bgr)
        return crop

    def extract_crop(
        self,
        image_bgr: np.ndarray,
        bbox: Optional[Union[Sequence[float], Sequence[int]]] = None,
    ) -> np.ndarray:
        """Extract a 576-D feature vector from a single image and optional bbox."""
        crop_bgr = self.crop_image(image_bgr, bbox=bbox)
        if crop_bgr.size == 0:
            return np.zeros(576, dtype=np.float32)

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_batch(
        self,
        items: Sequence[Union[np.ndarray, Image.Image, Tuple[np.ndarray, Optional[Sequence]]]],
    ) -> np.ndarray:
        """Extract 576-D feature vectors for a batch of images or (image, bbox) pairs.

        Returns an (N, 576) float32 ndarray.
        """
        if not items:
            return np.empty((0, 576), dtype=np.float32)

        tensors: List[torch.Tensor] = []
        for item in items:
            if isinstance(item, tuple):
                img, bbox = item
                crop_bgr = self.crop_image(img, bbox=bbox)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
            elif isinstance(item, Image.Image):
                pil_img = item.convert("RGB")
            elif isinstance(item, np.ndarray):
                # Assume pre-cropped BGR image
                crop_bgr = item
                if crop_bgr.size == 0:
                    pil_img = Image.new("RGB", (224, 224), color=0)
                else:
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)
            else:
                raise TypeError(f"Unsupported item type for visual extraction: {type(item)}")

            tensors.append(self.transform(pil_img))

        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model(batch_tensor)
        return feats.cpu().numpy().astype(np.float32)
