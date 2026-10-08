import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, DenseNet121_Weights, EfficientNet_B3_Weights, ResNet50_Weights, convnext_tiny, densenet121, efficientnet_b3, resnet50


class ResNetBaseline(nn.Module):
    def __init__(self, num_labels=4):
        super().__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, num_labels)

    def forward(self, images):
        return self.encoder(images)


class DenseNetBaseline(nn.Module):
    def __init__(self, num_labels=4):
        super().__init__()
        self.encoder = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.encoder.classifier = nn.Linear(self.encoder.classifier.in_features, num_labels)

    def forward(self, images):
        return self.encoder(images)


class MLPBaseline(nn.Module):
    def __init__(self, tabular_dim, num_labels=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_labels),
        )

    def forward(self, tabular):
        return self.network(tabular)


class EfficientNetB3Baseline(nn.Module):
    def __init__(self, num_labels=4):
        super().__init__()
        self.encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.encoder.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.encoder.classifier[1].in_features, num_labels)
        )

    def forward(self, images):
        return self.encoder(images)


class ConvNeXtTinyBaseline(nn.Module):
    def __init__(self, num_labels=4):
        super().__init__()
        self.encoder = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.encoder.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.encoder.classifier[2].in_features, num_labels)
        )

    def forward(self, images):
        return self.encoder(images)
