import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms as transforms

class SiameseEfficientNet(nn.Module):
    """Same model architecture as training script"""
    def __init__(self, embedding_dim=256, dropout_rate=0.2):
        super(SiameseEfficientNet, self).__init__()

        # Load pre-trained EfficientNet-B0
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Remove the classifier
        self.backbone.classifier = nn.Sequential()

        # Add projection head for embeddings
        self.projection_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1280, 512),  # EfficientNet-B0 outputs 1280 features
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, embedding_dim),
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        # Extract features from backbone
        x = self.transform(x)
        x = x.unsqueeze(0)
        features = self.backbone(x)

        # Get embeddings
        embeddings = self.projection_head(features)

        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
