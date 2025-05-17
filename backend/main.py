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

app = FastAPI()

# Determina la ruta base: si estamos en .exe, usa _MEIPASS; si no, __file__
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

static_dir = os.path.join(base_path, "build")
if not os.path.isdir(static_dir):
    raise RuntimeError(f"No se encontró la carpeta estática en: {static_dir}")

# Monta build/ como ruta raíz
app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

# Habilitar CORS para que tu frontend (en localhost:3000 o el origen que uses) pueda llamar
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
    Aquí llamarías a tu lógica de IA. Ejemplo ilustrativo:
    - image_array: ndarray (H×W×C o H×W×D)
    - model_name: str
    Debe devolver:
      1) processed_array: ndarray procesado para devolver como imagen
      2) classification: diccionario con la salida (probabilidades, etiquetas…)
    """
    # --- Ejemplo dummy: invertir canales y clasificar aleatoriamente ---
    processed = np.flip(image_array, axis=-1)
    classification = {
        "model_used": model_name,
        "labels": {
            "clase_A": float(np.random.rand()),
            "clase_B": float(np.random.rand()),
        },
    }
    return processed, classification


@app.post("/api/process", response_model=ProcessResponse)
async def process_image(image: UploadFile = File(...), model: str = Form(...)):
    # 1. Leer bytes
    file_bytes = await image.read()
    try:
        img_array = load_image_bytes(file_bytes, image.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Procesar con tu IA
    processed_arr, classification = process_with_model(img_array, model)

    # 3. Convertir processed_arr a imagen PNG y luego a base64
    #    Asumimos que processed_arr cabe en un uint8 0–255
    if processed_arr.dtype != np.uint8:
        processed_arr = np.clip(processed_arr, 0, 255).astype(np.uint8)

    # Para arrays 2D o 3D
    if processed_arr.ndim == 2:
        mode = "L"
    elif processed_arr.shape[2] == 3:
        mode = "RGB"
    else:
        # En casos más complejos podrías hacer un slice, un MIP, etc.
        mode = "RGB"
        processed_arr = processed_arr[..., :3]

    img_pil = Image.fromarray(processed_arr, mode=mode)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return ProcessResponse(image_base64=b64_str, classification=classification)
