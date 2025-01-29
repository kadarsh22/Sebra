import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_class=10, return_feat=False):
        super(MLP, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(3 * 28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        self.classifier = nn.Linear(100, num_class)
        self.return_feat = return_feat

    def forward(self, x):
        x = x.view(x.size(0), -1)
        feat = self.backbone(x)
        pred = self.classifier(feat)
        return pred
