import torch.nn as nn
import timm [cite: 590]

class DRVisionModel(nn.Module): [cite: 663]
    def __init__(self, num_classes=5): [cite: 664]
        super(DRVisionModel, self).__init__() [cite: 665]
        self.model = timm.create_model( [cite: 666, 667, 668]
            'efficientnet_b3', [cite: 669]
            pretrained=True [cite: 670]
        )
        in_features = self.model.classifier.in_features [cite: 671, 672]
        self.model.classifier = nn.Linear(in_features, num_classes) [cite: 673]
        
    def forward(self, x): [cite: 676]
        return self.model(x) [cite: 677]
