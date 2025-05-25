from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import io
import base64
from pathlib import Path
from fastapi.staticfiles import StaticFiles

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
from CNN_pretrained_model.AD_pretrained_utilities import CNN  # Clase del modelo ADNI
import torch.nn as nn
from scipy.io import loadmat

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
from types import SimpleNamespace

param_ad = SimpleNamespace(
    n_conv=8,
    kernels=[(3, 3, 3)] * 8,
    pooling=[
        (4, 4, 4),
        (0, 0, 0),
        (3, 3, 3),
        (0, 0, 0),
        (2, 2, 2),
        (0, 0, 0),
        (2, 2, 2),
        (0, 0, 0),
    ],
    in_channels=[1, 8, 8, 16, 16, 32, 32, 64],
    out_channels=[8, 8, 16, 16, 32, 32, 64, 64],
    dropout=0.0,
    fweights=[256, 2],
)


# Inicializar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo ADNI preentrenado en CPU (carga inicial)
model_ad = CNN(param_ad)
checkpoint_ad = torch.load(weights_ad, map_location="cpu")
model_ad.load_state_dict(checkpoint_ad)
model_ad.eval()  # en CPU por defecto

# 2) Extrae cuántas entradas tenía la capa original
#    En tu clase, `self.f[-1]` es el último nn.Linear
old_last_fc = model_ad.f[-1]
in_features = old_last_fc.in_features

# 3) Sustituye esa capa por una secuencia Linear→Softmax
model_ad.f[-1] = nn.Sequential(nn.Linear(in_features, 1), nn.Softmax(dim=1)).to(device)

# 4) (Opcional) Re-inicializa los pesos de la nueva capa
nn.init.kaiming_normal_(model_ad.f[-1][0].weight, nonlinearity="linear")
model_ad.f[-1][0].bias.data.zero_()


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
    """
    Soporta: .jpg/.png, .nii/.nii.gz, y .mat (variable 'volume' por defecto).
    """
    # Detectar extensión
    lower = filename.lower()
    if lower.endswith((".nii", ".nii.gz")):
        ext = "nii"
    else:
        ext = lower.rsplit(".", 1)[-1]

    # Rutas temporales para NIfTI
    if ext in ("nii"):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix="." + ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name
        try:
            img_nii = nib.load(tmp_path)
            return img_nii.get_fdata()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Imágenes comunes
    elif ext in ("jpg", "jpeg", "png"):
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(img)

    # Archivos MATLAB
    elif ext == "mat":

        # Definimos el directorio tmp y lo creamos si no existe
        tmp_dir = Path("tmp")
        # tmp_dir.mkdir(parents=True, exist_ok=True)

        # Generamos un nombre único
        unique_name = f"{uuid.uuid4().hex}.mat"
        tmp_path = tmp_dir / unique_name

        # Escribimos el fichero
        tmp_path.write_bytes(file_bytes)

        try:
            # Cargamos el .mat
            mat = loadmat(tmp_path, squeeze_me=True, struct_as_record=False)
            print("1")

            if "volume" in mat:
                vol = mat["volume"]
                print("2")
            else:
                connectivity = mat["connectivity"]
                print("3")
                # if not connectivity:
                #     raise ValueError(
                #         "No se encontró una variable de volumen 3D en el .mat"
                #     )
                print("4")

            return np.array(connectivity, dtype=np.float32)
        finally:
            # Siempre limpiamos
            try:
                # tmp_path.unlink()
                pass
            except OSError:
                pass
    else:
        raise ValueError(f"Formato no soportado: .{ext}")


def process_with_model(
    image_array: np.ndarray, torch_model: torch.nn.Module, device
) -> (np.ndarray, Dict[str, Any]):
    # Apply the same preprocessing as training
    # 1) Resize & crop/pad to (96,96,73)
    vol = img_processing(image_array, scaling=0.5, final_size=(96, 96, 73))
    # 2) Convert to torch tensor and normalize
    print("Image shape:", vol.shape)
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
        print("entra1")
        img_array = load_image_bytes(file_bytes, image.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    print("entra2")
    if model not in dmodels:
        raise HTTPException(...)
    torch_model, use_cuda = dmodels[model]

    # Move model to correct device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    torch_model.to(device)

    print("entra3")
    processed_arr, classification = process_with_model(img_array, torch_model, device)

    print("entra4")
    # Convert to base64 PNG
    img_pil = Image.fromarray(processed_arr, mode="L")
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    print("entra5")
    classification["model_used"] = model
    return ProcessResponse(image_base64=b64_str, classification=classification)
