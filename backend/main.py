from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import io
import base64
from fastapi.staticfiles import StaticFiles

# Para procesar .nii
import nibabel as nib
from scipy import ndimage

# Para procesar imágenes comunes
from PIL import Image
import numpy as np
import os, sys

# PyTorch
import torch
import torch.nn.functional as F
from torchvision import transforms
from types import SimpleNamespace
from CNN_pretrained_model.AD_pretrained_utilities import CNN  # Clase del modelo ADNI

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


# =====================
#  DATA TRANSFORMS
# =====================
def resize_data_volume_by_scale(data: np.ndarray, scale):
    """
    Resize the data based on the provided scale (float or 3-list).
    """
    scale_list = [scale] * 3 if isinstance(scale, float) else scale
    return ndimage.zoom(data, zoom=scale_list, order=0)


def normalize_intensity(
    img_tensor: torch.Tensor, normalization: str = "mean"
) -> torch.Tensor:
    """
    Normalize tensor: 'mean' uses non-zero voxels, 'max' scales to [0,1].
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / (std_val + 1e-8)
    elif normalization == "max":
        mx, mn = img_tensor.max(), img_tensor.min()
        img_tensor = (img_tensor - mn) / (mx - mn + 1e-8)
    return img_tensor


def img_processing(
    volume: np.ndarray, scaling: float = 0.5, final_size=(96, 96, 73)
) -> np.ndarray:
    """
    Resize first by a uniform scaling, then adjust to final dimensions.
    """
    # initial down/up-sample
    vol = resize_data_volume_by_scale(volume, scale=scaling)
    # compute scale factors to reach final_size
    factors = [final_size[i] / vol.shape[i] for i in range(3)]
    return resize_data_volume_by_scale(vol, scale=factors)


def torch_norm(volume: np.ndarray) -> torch.Tensor:
    """
    Convert numpy volume to torch tensor and apply intensity normalization.
    """
    tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = normalize_intensity(tensor, normalization="mean")
    return tensor


# =====================
#       APP SETUP
# =====================
app = FastAPI()
# (rest unmodified...)
# Montar estáticos, CORS, modelos, etc.


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
    image_array: np.ndarray, torch_model: torch.nn.Module, device
) -> (np.ndarray, Dict[str, Any]):
    # Apply the same preprocessing as training
    # 1) Resize & crop/pad to (96,96,73)
    vol = img_processing(image_array, scaling=0.5, final_size=(96, 96, 73))
    # 2) Convert to torch tensor and normalize
    tensor = torch_norm(vol).to(device)

    # Inferencia
    with torch.no_grad():
        logits = torch_model(tensor)
        score = logits.squeeze().item()

    # Generar imagen MIP (eje X)
    mip = np.max(vol, axis=0)
    mip = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
    mip = (mip * 255).astype(np.uint8)

    classification = {"disease_score": float(score)}
    return mip, classification


@app.post("/api/process", response_model=ProcessResponse)
async def process_image(image: UploadFile = File(...), model: str = Form(...)):
    file_bytes = await image.read()
    try:
        img_array = load_image_bytes(file_bytes, image.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if model not in dmodels:
        raise HTTPException(...)
    torch_model, use_cuda = dmodels[model]

    # Move model to correct device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    torch_model.to(device)

    processed_arr, classification = process_with_model(img_array, torch_model, device)

    # Convert to base64 PNG
    img_pil = Image.fromarray(processed_arr, mode="L")
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    classification["model_used"] = model
    return ProcessResponse(image_base64=b64_str, classification=classification)
