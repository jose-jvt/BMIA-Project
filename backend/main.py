# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import io
import base64
from fastapi.staticfiles import StaticFiles

# Para procesar .nii
import nibabel as nib

# Para procesar imágenes comunes
from PIL import Image
import numpy as np
import os, sys

# PyTorch
import torch
from .CNN_pretrained_model.AD_pretrained_utilities import (
    CNN,
)  # Clase del modelo ADNI

app = FastAPI()

# Determina la ruta base: si estamos en .exe, usa _MEIPASS; si no, __file__
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

# Ruta al directorio de modelos (coloca aquí tu peso .pt)
models_dir = os.path.join(base_path, "models")
weights_path = os.path.join(models_dir, "AD_pretrained_weights.pt")
if not os.path.isfile(weights_path):
    raise RuntimeError(f"No se encontró el archivo de pesos en: {weights_path}")

# Cargar modelo preentrenado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=1)
model.activation = torch.nn.Sigmoid()  # para salida en [0,1]
checkpoint = torch.load(weights_path, map_location=device)
# Ajusta esta línea si tu checkpoint incluye un dict con 'model_state_dict'
model.load_state_dict(checkpoint)
model.to(device).eval()

static_dir = os.path.join(base_path, "build")
if not os.path.isdir(static_dir):
    raise RuntimeError(f"No se encontró la carpeta estática en: {static_dir}")

# Monta build/ como ruta raíz
app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessResponse(BaseModel):
    image_base64: str
    classification: Dict[str, Any]


def load_image_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("jpg", "jpeg", "png"):
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(img)
    elif ext in ("nii", "nii.gz"):
        img_nii = nib.load(io.BytesIO(file_bytes))
        return img_nii.get_fdata()
    else:
        raise ValueError(f"Formato no soportado: {ext}")


def process_with_model(
    image_array: np.ndarray, model_name: str
) -> (np.ndarray, Dict[str, Any]):
    """
    Procesa volúmenes 3D NIfTI con el modelo ADNI preentrenado.
    Devuelve la MIP para visualización y el score de enfermedad.
    """
    # image_array: np.ndarray shape (X,Y,Z)
    # 1) Preprocesamiento: reescalar a 96x96x73
    volume = image_array.astype(np.float32)
    # Normalizar a [0,1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # [1,1,X,Y,Z]
    tensor = torch.nn.functional.interpolate(
        tensor, size=(73, 96, 96), mode="trilinear", align_corners=False
    )
    tensor = tensor.to(device)

    # 2) Inferencia
    with torch.no_grad():
        output = model(tensor)
        score = output.item()  # en [0,1]

    # 3) Generar imagen de salida: MIP sobre eje X
    mip = np.max(volume, axis=0)  # shape (Y,Z)
    # Escalar a 0-255
    mip = (mip * 255).astype(np.uint8)

    classification = {"model_used": model_name, "disease_score": float(score)}
    return mip, classification


@app.post("/api/process", response_model=ProcessResponse)
async def process_image(image: UploadFile = File(...), model: str = Form(...)):
    # 1. Leer bytes
    file_bytes = await image.read()
    try:
        img_array = load_image_bytes(file_bytes, image.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Procesar con IA
    processed_arr, classification = process_with_model(img_array, model)

    # 3. Convertir processed_arr a imagen PNG y luego a base64
    if processed_arr.dtype != np.uint8:
        processed_arr = np.clip(processed_arr, 0, 255).astype(np.uint8)

    mode = "L"  # MIP es 2D
    img_pil = Image.fromarray(processed_arr, mode=mode)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return ProcessResponse(image_base64=b64_str, classification=classification)
