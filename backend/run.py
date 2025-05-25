# run.py

import main  # ← fuerza a PyInstaller a incluir main.py
import uvicorn
import sys
import os

if __name__ == "__main__":

    # Cambia al directorio del bundle (o de desarrollo) para resolver rutas relativas
    if getattr(sys, "frozen", False):
        bundle_dir = sys._MEIPASS
    else:
        bundle_dir = os.path.dirname(__file__)

    os.chdir(bundle_dir)

    # Importa aquí para forzar a PyInstaller a incluir main.py
    import main

    import uvicorn

    uvicorn.run("main:app", host="localhost", port=5000)
