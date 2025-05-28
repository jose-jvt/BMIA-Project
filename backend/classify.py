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
    # Apply the same preprocessing as training
    tensor = mat_array.to(device)

    # Inferencia
    with torch.no_grad():
        logits = torch_model(tensor)
        score = logits.squeeze().item()

    classification = {"disease_score": float(score)}
    return classification


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
