import { useState } from 'react';

export default function ImageProcessor() {
    const [file, setFile] = useState(null);
    const [model, setModel] = useState('');
    const [resultImageUrl, setResultImageUrl] = useState('');
    const [classification, setClassification] = useState(null);
    const [loading, setLoading] = useState(false);

    const models = [
        { value: 'model_v1', label: 'Modelo v1 (CNN)' },
        { value: 'model_unet', label: 'Modelo U-Net' },
        { value: 'model_custom', label: 'Modelo personalizado' },
        // …otros modelos
    ];

    const onFileChange = e => {
        setFile(e.target.files[0]);
        // opcional: preview local si es .jpg/.png
    };

    const onModelChange = e => {
        setModel(e.target.value);
    };

    const onSubmit = async e => {
        e.preventDefault();
        if (!file || !model) return alert('Selecciona fichero y modelo');
        setLoading(true);

        const form = new FormData();
        form.append('image', file);
        form.append('model', model);

        try {
            const res = await fetch('https://tu-backend/api/process', {
                method: 'POST',
                body: form
            });

            if (!res.ok) throw new Error(`Server retornó ${res.status}`);

            // Cabeceras: Content-Type: image/png y application/json
            // Leer la parte imagen como blob
            const contentType = res.headers.get('Content-Type') || '';
            if (contentType.startsWith('image/')) {
                const blob = await res.blob();
                setResultImageUrl(URL.createObjectURL(blob));
                // luego pedimos JSON separado, o usar multipart: aquí simplificamos
                const json = await res.json(); // si viene combinado
                setClassification(json.classification);
            } else {
                // si el backend devuelve JSON con campo de imagen en base64:
                const json = await res.json();
                setResultImageUrl(`data:image/png;base64,${json.image_base64}`);
                setClassification(json.classification);
            }
        } catch (err) {
            console.error(err);
            alert('Error procesando la imagen');
        } finally {
            setLoading(false);
        }
    };

    return (
        <form onSubmit={onSubmit}>
            <div>
                <label>Selecciona la imagen:</label>
                <input type="file" accept=".jpg,.jpeg,.png,.nii" onChange={onFileChange} />
            </div>

            <div>
                <label>Elige el modelo IA:</label>
                <select value={model} onChange={onModelChange}>
                    <option value="" disabled>-- selecciona modelo --</option>
                    {models.map(m => (
                        <option key={m.value} value={m.value}>{m.label}</option>
                    ))}
                </select>
            </div>

            <button type="submit" disabled={loading}>
                {loading ? 'Procesando…' : 'Enviar'}
            </button>

            {resultImageUrl && (
                <div>
                    <h3>Imagen procesada:</h3>
                    <img src={resultImageUrl} alt="Procesada" style={{ maxWidth: '100%' }} />
                </div>
            )}

            {classification && (
                <div>
                    <h3>Clasificación:</h3>
                    <pre>{JSON.stringify(classification, null, 2)}</pre>
                </div>
            )}
        </form>
    );
}

