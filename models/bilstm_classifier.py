import torch
import torch.nn as nn

class BiLSTMPoseClassifier(nn.Module):
    def __init__(self, input_size: int = 51, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return self.sigmoid(logits)
