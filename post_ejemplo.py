import requests

url = "http://localhost:8000/api/process"
files = {
    "image": (
        "imagen.png",
        open("imagen.png", "rb"),
        "application/octet-stream",
    )
}
data = {"model": "pretrained_ad"}

resp = requests.post(url, files=files, data=data)
if resp.ok:
    result = resp.json()
    print("Clasificación:", result["classification"])
    # Para decodificar y guardar la imagen resultante:
    import base64

    with open("mip.png", "wb") as f:
        f.write(base64.b64decode(result["image_base64"]))
else:
    print("Error:", resp.status_code, resp.text)
