import os
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURE_COLUMNS, SEQUENCE_LENGTH

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Device : {DEVICE}")


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================================
# Create Sliding Window Sequences
# ==========================================================


def create_sequences(data, seq_length=SEQUENCE_LENGTH):
    """
    Converts a feature matrix into overlapping sequences.

    Input:
        data.shape = (N, num_features)

    Output:
        sequences.shape = (N-seq_length+1, seq_length, num_features)
    """

    sequences = []

    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])

    return np.asarray(sequences, dtype=np.float32)


# ==========================================================
# Dataset
# ==========================================================


class StockDataset(Dataset):

    def __init__(
        self,
        csv_file,
        sequence_length=SEQUENCE_LENGTH,
    ):

        if not os.path.exists(csv_file):
            raise FileNotFoundError(csv_file)

        self.df = pd.read_csv(csv_file)

        missing = [
            col
            for col in FEATURE_COLUMNS
            if col not in self.df.columns
        ]

        if len(missing) > 0:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        features = self.df[FEATURE_COLUMNS].values.astype(np.float32)

        self.sequences = create_sequences(
            features,
            sequence_length,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        x = torch.tensor(
            self.sequences[idx],
            dtype=torch.float32,
        )

        # Autoencoder target is the same input
        return x, x

# ==========================================================
# Residual TCN Block
# ==========================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.2,
    ):
        super().__init__()

        # Padding keeps the sequence length unchanged
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)

        # Match channel dimensions for residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):

        identity = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = self.relu2(out)
        out = self.dropout2(out)

        return out

# ==========================================================
# Encoder
# ==========================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels=32,
        latent_channels=64,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(

            ResidualBlock(
                input_channels,
                hidden_channels,
                dilation=1,
                dropout=dropout,
            ),

            ResidualBlock(
                hidden_channels,
                hidden_channels,
                dilation=2,
                dropout=dropout,
            ),

            ResidualBlock(
                hidden_channels,
                latent_channels,
                dilation=4,
                dropout=dropout,
            ),

            ResidualBlock(
                latent_channels,
                latent_channels,
                dilation=8,
                dropout=dropout,
            ),
        )

    def forward(self, x):
        return self.network(x)


# ==========================================================
# Decoder
# ==========================================================

class Decoder(nn.Module):

    def __init__(
        self,
        output_channels,
        hidden_channels=32,
        latent_channels=64,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(

            ResidualBlock(
                latent_channels,
                latent_channels,
                dilation=8,
                dropout=dropout,
            ),

            ResidualBlock(
                latent_channels,
                hidden_channels,
                dilation=4,
                dropout=dropout,
            ),

            ResidualBlock(
                hidden_channels,
                hidden_channels,
                dilation=2,
                dropout=dropout,
            ),

            ResidualBlock(
                hidden_channels,
                output_channels,
                dilation=1,
                dropout=dropout,
            ),
        )

    def forward(self, x):
        return self.network(x)


# ==========================================================
# TCN Autoencoder
# ==========================================================

class TCNAutoencoder(nn.Module):

    def __init__(
        self,
        input_features=len(FEATURE_COLUMNS),
        hidden_channels=32,
        latent_channels=64,
        dropout=0.2,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_channels=input_features,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            dropout=dropout,
        )

        self.decoder = Decoder(
            output_channels=input_features,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Input shape:
            (batch, sequence_length, features)

        Conv1D expects:
            (batch, channels, sequence_length)
        """

        # (B, Seq, Features) -> (B, Features, Seq)
        x = x.permute(0, 2, 1)

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        # Back to (B, Seq, Features)
        reconstructed = reconstructed.permute(0, 2, 1)

        return reconstructed

# ==========================================================
# Training Function
# ==========================================================

from torch.utils.data import DataLoader


def train_model(
    train_csv,
    model_save_path="saved_models/tcn_autoencoder.pth",
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
):

    # Dataset
    train_dataset = StockDataset(train_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Training Samples : {len(train_dataset)}")

    # Model
    model = TCNAutoencoder().to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_loss = float("inf")

    print("\nTraining Started...\n")

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for inputs, targets in train_loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        scheduler.step(epoch_loss)

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] "
            f"Loss = {epoch_loss:.6f}"
        )

        # Save Best Model
        if epoch_loss < best_loss:

            best_loss = epoch_loss

            os.makedirs(
                os.path.dirname(model_save_path),
                exist_ok=True,
            )

            torch.save(
                model.state_dict(),
                model_save_path,
            )

            print(" Best model saved.")

    print("\nTraining Completed.")
    print(f"Best Loss : {best_loss:.6f}")

    return model

def save_threshold(
    train_csv,
    model_path,
    threshold_path="saved_models/threshold.json",
):

    dataset = StockDataset(train_csv)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
    )

    model = TCNAutoencoder().to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
        )
    )

    model.eval()

    criterion = nn.MSELoss(reduction="none")

    errors = []

    with torch.no_grad():

        for x, _ in loader:

            x = x.to(DEVICE)

            output = model(x)

            loss = criterion(output, x)

            loss = loss.mean(dim=(1,2))

            errors.extend(loss.cpu().numpy())

    threshold = np.percentile(errors,95)

    with open(threshold_path,"w") as f:

        json.dump(
            {"threshold":float(threshold)},
            f,
            indent=4,
        )

    print(f"Threshold Saved : {threshold:.6f}")
# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    TRAIN_CSV = "feature_extract/bank_nifty_train.csv"

    MODEL_PATH = "saved_models/tcn_autoencoder.pth"

    train_model(
        train_csv=TRAIN_CSV,
        model_save_path=MODEL_PATH,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
    )
    save_threshold(
        TRAIN_CSV,
        MODEL_PATH,
    )