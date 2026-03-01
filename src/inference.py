"""
Emotion detection inference functions.

This module provides functions for loading models, preprocessing images,
and running emotion predictions.
"""

from typing import Optional, List, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CNN_NeuralNet


# Emotion labels (order must match training)
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def load_model(
    model_path: str,
    num_classes: int,
    in_channels: Optional[int],
    linear_in_features: int,
    device: torch.device,
) -> nn.Module:
    """Load emotion detection model from checkpoint.

    Supports loading:
    - Complete model objects
    - State dictionaries
    - Checkpoints with 'model_state' or 'state_dict' keys

    Automatically handles channel conversion between grayscale and RGB.

    Args:
        model_path: Path to model checkpoint file
        num_classes: Number of emotion classes
        in_channels: Input channels (1 or 3). If None, auto-detected from checkpoint
        linear_in_features: Flattened feature size for final linear layer
        device: torch.device for model

    Returns:
        Loaded model in eval mode on specified device
    """
    obj = torch.load(model_path, map_location=device)

    # If the checkpoint is already a full model
    if isinstance(obj, nn.Module):
        model = obj
        model.eval().to(device)
        return model

    # Extract state dict from various checkpoint formats
    if isinstance(obj, dict) and any(k in obj for k in ["model_state", "state_dict"]):
        state_dict = obj.get("model_state", obj.get("state_dict"))
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise RuntimeError(
            "Unsupported checkpoint format. Provide a full model or a state dict."
        )

    # Auto-detect input channels from conv1 weights
    detected_in = None
    for key, tensor in state_dict.items():
        if key.endswith("conv1.weight") and tensor.ndim == 4:
            detected_in = int(tensor.shape[1])
            break

    if in_channels is None:
        in_ch = detected_in if detected_in is not None else 3
    else:
        in_ch = in_channels

    # Create model
    model = CNN_NeuralNet(
        in_channels=in_ch,
        num_classes=num_classes,
        linear_in_features=linear_in_features
    )

    # Handle channel mismatch: convert between grayscale and RGB if needed
    conv_key = None
    for k in ["conv1.weight", "module.conv1.weight"]:
        if k in state_dict:
            conv_key = k
            break

    if conv_key is not None:
        w = state_dict[conv_key]
        if w.ndim == 4 and w.shape[1] != model.conv1[0].weight.shape[1]:
            # Convert RGB weights to grayscale (average channels)
            if w.shape[1] == 3 and model.conv1[0].weight.shape[1] == 1:
                state_dict[conv_key] = w.mean(dim=1, keepdim=True)
            # Convert grayscale weights to RGB (repeat channel)
            elif w.shape[1] == 1 and model.conv1[0].weight.shape[1] == 3:
                state_dict[conv_key] = w.repeat(1, 3, 1, 1)

    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys: {missing}, unexpected keys: {unexpected}")

    model.eval().to(device)
    return model


def preprocess_roi(
    roi_bgr: np.ndarray,
    img_size: int,
    in_channels: int,
    imagenet_norm: bool,
    clahe: Optional[cv2.CLAHE] = None,
) -> torch.Tensor:
    """Preprocess a face ROI for emotion prediction.

    Args:
        roi_bgr: Face region in BGR format (OpenCV convention)
        img_size: Target size for resizing (square)
        in_channels: Number of channels (1 for grayscale, 3 for RGB)
        imagenet_norm: Whether to apply ImageNet normalization
        clahe: Optional CLAHE object for histogram equalization

    Returns:
        Preprocessed tensor of shape (1, in_channels, img_size, img_size)
    """
    # Optional CLAHE preprocessing on luminance channel
    if clahe is not None:
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = clahe.apply(y)
        ycrcb = cv2.merge([y, cr, cb])
        roi_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if in_channels == 1:
        # Grayscale processing
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        roi = roi.astype(np.float32) / 255.0
        roi = roi[None, :, :]  # Add channel dimension

        if imagenet_norm:
            roi = (roi - 0.5) / 0.5
    else:
        # RGB processing
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        roi = roi.astype(np.float32) / 255.0
        roi = roi.transpose(2, 0, 1)  # HWC to CHW

        if imagenet_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            roi = (roi - mean) / std

    return torch.from_numpy(roi).unsqueeze(0)  # Add batch dimension


def predict_emotion(
    model: nn.Module,
    face_roi: np.ndarray,
    device: torch.device,
    img_size: int = 44,
    in_channels: int = 3,
    imagenet_norm: bool = True,
    clahe: Optional[cv2.CLAHE] = None,
) -> Tuple[str, float]:
    """Predict emotion from a face ROI.

    Args:
        model: Trained emotion detection model
        face_roi: Face region in BGR format
        device: torch.device for inference
        img_size: Input image size
        in_channels: Number of channels
        imagenet_norm: Whether to use ImageNet normalization
        clahe: Optional CLAHE for preprocessing

    Returns:
        Tuple of (emotion_label, confidence)
    """
    tensor = preprocess_roi(face_roi, img_size, in_channels, imagenet_norm, clahe)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else str(idx)
    confidence = float(probs[idx])

    return label, confidence


def predict_emotions_batch(
    model: nn.Module,
    frame: np.ndarray,
    face_boxes: List[Tuple[int, int, int, int]],
    device: torch.device,
    img_size: int = 44,
    in_channels: int = 3,
    imagenet_norm: bool = True,
    clahe: Optional[cv2.CLAHE] = None,
) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """Predict emotions for multiple faces in a frame.

    Args:
        model: Trained emotion detection model
        frame: Full frame in BGR format
        face_boxes: List of (x, y, w, h) face bounding boxes
        device: torch.device for inference
        img_size: Input image size
        in_channels: Number of channels
        imagenet_norm: Whether to use ImageNet normalization
        clahe: Optional CLAHE for preprocessing

    Returns:
        List of (box, emotion_label, confidence) tuples
    """
    results = []
    for (x, y, w, h) in face_boxes:
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        label, confidence = predict_emotion(
            model, roi, device, img_size, in_channels, imagenet_norm, clahe
        )
        results.append(((x, y, w, h), label, confidence))

    return results
