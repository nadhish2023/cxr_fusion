import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class CXRMultimodalModel(nn.Module):
    def __init__(self, tabular_dim, num_labels=4, feature_dim=256):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        image_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone
        self.image_projection = nn.Sequential(nn.Linear(image_dim, feature_dim), nn.ReLU(), nn.Dropout(0.2))
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, feature_dim), nn.ReLU(),
        )
        self.gate = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim))
        self.classifier = nn.Linear(feature_dim, num_labels)

    def forward(self, images, tabular):
        image_features = self.image_projection(self.image_encoder(images))
        tabular_features = self.tabular_encoder(tabular)
        gate = torch.sigmoid(self.gate(image_features + tabular_features))
        fused = gate * image_features + (1 - gate) * tabular_features
        return self.classifier(fused)
