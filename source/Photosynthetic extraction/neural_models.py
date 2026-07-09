import torch
import torch.nn as nn

# =====================================================
# IMPROVED BPNN
# =====================================================
class BPNN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden1,
        hidden2=64,
        dropout=0.0
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden1),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden2, 1)
        )

    def forward(self, x):

        return self.net(x)

# =====================================================
# LSTM
# =====================================================
class LSTMNet(nn.Module):

    def __init__(
        self,
        d,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        x = x.unsqueeze(-1)

        out, _ = self.lstm(x)

        return self.fc(out[:, -1])

# =====================================================
# CNN1D
# =====================================================
class CNN1D(nn.Module):

    def __init__(
        self,
        conv1,
        conv2,
        dense,
        kernel_size,
        dropout=0.0
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=conv1,
                kernel_size=kernel_size
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=conv1,
                out_channels=conv2,
                kernel_size=kernel_size
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),

            nn.Linear(conv2, dense),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(dense, 1)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        return self.net(x)