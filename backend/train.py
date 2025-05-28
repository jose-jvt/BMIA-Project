import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import scipy.io as sio


def load_mat_file(path, data_key=None):
    """
    Carga un .mat y extrae el volumen y el label.
    Si data_key es None, busca 'volume' o 'connectivity'.
    label_key debe existir dentro del .mat para regresión.
    """
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    # Elegir dato
    if data_key and data_key in mat:
        arr = mat[data_key]
    elif "volume" in mat:
        arr = mat["volume"]
    elif "connectivity" in mat:
        arr = mat["connectivity"]
    else:
        raise KeyError(f"No se encontró clave de datos en {{path}}")

    return np.array(arr, dtype=np.float32)


class MatFolderDataset(Dataset):
    """
    Dataset basado en un CSV con IDs y un directorio de subcarpetas.
    Lee CSV, extrae ID y últimos tres caracteres.
    Cada ID corresponde a una subcarpeta que contiene un .mat.
    La etiqueta se extrae de una columna del CSV.
    """

    def __init__(
        self,
        csv_path,
        root_dir,
        data_key="connectivity",
        label_col=None,
        transform=None,
    ):
        # Leer CSV
        self.df = pd.read_csv(csv_path)
        self.df["last3"] = self.df["ID"].str[-2:]
        self.root_dir = root_dir
        self.ids = self.df["last3"].tolist()
        self.data_key = data_key
        self.label_col = "MoCA"
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Obtener ID y etiqueta
        sample_id = str(int(self.ids[idx]))
        row = self.df.iloc[idx]
        if self.label_col is None or self.label_col not in self.df.columns:
            raise ValueError(
                "Debe especificar 'label_col' existente en CSV para regresión."
            )
        label = int(row[self.label_col])

        # Ruta a la carpeta y carga del .mat
        folder = os.path.join(self.root_dir, sample_id)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"No se encontró carpeta para ID {sample_id}")
        mats = [f for f in os.listdir(folder) if f.lower().endswith(".mat")]
        if not mats:
            raise FileNotFoundError(f"No se encontró archivo .mat en {folder}")
        mat_path = os.path.join(folder, mats[0])
        arr = load_mat_file(mat_path, self.data_key)

        if self.transform:
            arr = self.transform(arr)

        target = torch.tensor(label, dtype=torch.float32)
        return arr, target / 30


class RegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            print(f"Outputs: {outputs}, targets: {targets}")
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def main():
    # Transforms: resize to 256x256, convert to tensor, normalize to [0,1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = MatFolderDataset(
        csv_path="C:\\Users\\josev\\OneDrive\\Documentos\\MASTER\\BMIA\\Project\\BMIA-Project\\Dataset\\PDAM_DemografGeral.csv",
        root_dir="C:\\Users\\josev\\OneDrive\\Documentos\\MASTER\\BMIA\\Project\\BMIA-Project\\Dataset",
        transform=transform,
    )
    # Split
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    best_val_loss = float("inf")
    for epoch in range(1, 100 + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{10000} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "modelo.pth")
            print(f"Saved Best Model to {'modelo.pth'}")


if __name__ == "__main__":
    main()
