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

# PyTorch\import torch
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from .CNN_pretrained_model.AD_pretrained_utilities import CNN  # Clase del modelo ADNI

app = FastAPI()

# Determina la ruta base: si estamos en .exe, usa _MEIPASS; si no, __file__
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

# Ruta al directorio de modelos y pesos
models_dir = os.path.join(base_path, "CNN_pretrained_model")
weights_ad = os.path.join(models_dir, "AD_pretrained_weights.pt")
if not os.path.isfile(weights_ad):
    raise RuntimeError(f"No se encontró el archivo de pesos AD en: {weights_ad}")

# Definir parámetros de la CNN según arquitectura ADNI
param_ad = SimpleNamespace(
    n_conv=8,
    kernels=[(3, 3, 3)] * 8,
    pooling=[(2, 2, 2) if i in [0, 2, 4, 6] else (0, 0, 0) for i in range(8)],
    in_channels=[1, 8, 8, 16, 16, 32, 32, 64],
    out_channels=[8, 8, 16, 16, 32, 32, 64, 64],
    dropout=0.2,
    fweights=[256, 2],
)

# Inicializar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo ADNI preentrenado en CPU (carga inicial)
model_ad = CNN(param_ad)
checkpoint_ad = torch.load(weights_ad, map_location="cpu")
model_ad.load_state_dict(checkpoint_ad)
model_ad.eval()  # en CPU por defecto

# Mapear nombres de modelos disponibles a sus instancias y dispositivos
# Cada entrada: (modelo, preferencia_cuda)
dmodels = {
    "pretrained_ad": (model_ad, True),  # cargar en GPU si se selecciona
    # "otro_modelo": (otro_modelo_instancia, False),
}

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
    image_array: np.ndarray, torch_model: torch.nn.Module
) -> (np.ndarray, Dict[str, Any]):
    """
    Procesa volúmenes 3D NIfTI con el modelo dado.
    Devuelve la MIP para visualización y el score de enfermedad.
    """
    # Normalizar y reescalar volumen a 96x96x73
    volume = image_array.astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor, size=(73, 96, 96), mode="trilinear", align_corners=False
    ).to(device)

    # Inferencia
    with torch.no_grad():
        logits = torch_model(tensor)
        score = logits.squeeze().item()

    # Generar imagen MIP (eje X)
    mip = np.max(volume, axis=0)
    mip = (mip * 255).astype(np.uint8)

    classification = {"disease_score": float(score)}
    return mip, classification


@app.post("/api/process", response_model=ProcessResponse)
async def process_image(image: UploadFile = File(...), model: str = Form(...)):
    # Leer bytes
    file_bytes = await image.read()
    try:
        img_array = load_image_bytes(file_bytes, image.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Verificar modelo solicitado
    if model not in dmodels:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo desconocido '{model}'. Modelos disponibles: {list(dmodels.keys())}",
        )
    torch_model, use_cuda = dmodels[model]

    # Mover modelo a dispositivo adecuado
    if use_cuda and torch.cuda.is_available():
        torch_model.to(device)
        model_device = device
    else:
        torch_model.to("cpu")
        model_device = "cpu"

    # Procesar con IA
    processed_arr, classification = process_with_model(
        img_array, torch_model, model_device
    )

    # Convertir a PNG base64
    if processed_arr.dtype != np.uint8:
        processed_arr = np.clip(processed_arr, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(processed_arr, mode="L")
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Incluir nombre de modelo en respuesta
    classification["model_used"] = model
    return ProcessResponse(image_base64=b64_str, classification=classification)(
        image_base64=b64_str, classification=classification
    )
