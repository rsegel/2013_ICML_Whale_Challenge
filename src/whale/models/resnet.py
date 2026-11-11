"""Baseline: ImageNet-pretrained ResNet-18 on spectrograms."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from whale.features.spectrogram import LogSpectrogram, spec_augment


class SpectrogramResNet(nn.Module):
    """Waveform [batch, 4000] -> one logit per clip.

    The spectrogram has one channel; it is repeated three times to match the
    pretrained first layer. With ``freeze_backbone`` only the final linear
    layer trains and batch-norm statistics stay at their ImageNet values,
    which makes the model a linear probe comparable to the Perch probe.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False, augment: bool = True):
        super().__init__()
        self.spectrogram = LogSpectrogram()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        self.freeze_backbone = freeze_backbone
        self.augment = augment
        if freeze_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad = name.startswith("fc.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # keep batch-norm statistics fixed
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spectrogram(waveform)
        if self.training and self.augment:
            spec = spec_augment(spec)
        return self.backbone(spec.expand(-1, 3, -1, -1)).squeeze(1)
