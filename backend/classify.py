from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import io
import base64
from pathlib import Path

# Para procesar .nii
import nibabel as nib
from scipy import ndimage

# Para procesar .mat
import scipy.io as sio

# Para procesar imágenes comunes
from PIL import Image
import numpy as np
import os, sys
import uuid

# PyTorch
import torch
import torch.nn.functional as F
from torchvision import transforms
from types import SimpleNamespace

import torch.nn as nn
from scipy.io import loadmat
from train import RegressionCNN


def load_matrix_mat(matrix_path) -> np.ndarray:
    # Cargamos el .mat
    mat = loadmat(matrix_path, squeeze_me=True, struct_as_record=False)
    connectivity = mat["connectivity"]
    return np.array(connectivity, dtype=np.float32)


def process_with_model(mat_array: np.ndarray, torch_model: torch.nn.Module, device):
    # 1. Convertir numpy array a tensor de Torch (float32)
    tensor = torch.from_numpy(mat_array).float()
    print(f"Shape of tensor before unsqueeze: {tensor.shape}")  # (H, W)

    # 2. Añadir dimensión de canal y de batch
    #    Pasa de (H, W) → (1, 1, H, W) para que conv2d lo acepte
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    print(f"Shape of tensor after unsqueeze: {tensor.shape}")  # (1, 1, H, W)

    # 3. Mover al dispositivo (CPU/GPU)
    tensor = tensor.to(device)

    # 4. Inferencia
    with torch.no_grad():
        logits = torch_model(tensor)
        print(logits)
        score = logits.squeeze().item()

    return {"disease_score": float(score)}


def model_inference(matrix_path, model_path):
    # Inicializar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instanciar y cargar pesos
    model = RegressionCNN().to(device)
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")
    model.eval()

    try:
        mat_array = load_matrix_mat(matrix_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    model.to(device)

    classification = process_with_model(mat_array, model, device)

    return classification


if __name__ == "__main__":
    resultado = model_inference(
        matrix_path="C:\\Users\\josev\\OneDrive\\Documentos\\MASTER\\BMIA\\Project\\BMIA-Project\\Dataset\\1\\whole_brain_ROIs.mat",
        model_path="C:\\Users\\josev\\OneDrive\\Documentos\\MASTER\\BMIA\\Project\\BMIA-Project\\backend\\modelo.pth",
    )
    print(resultado)
