# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from app.utils import predict_image

app = FastAPI(title="End-to-End AI: Waste Classifier")

# Serve file statis (opsional, untuk tampilan preview)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Waste Classifier</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial; max-width: 600px; margin: 40px auto; padding: 20px; }
            .upload-box { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            img#preview { max-width: 100%; margin-top: 10px; display: none; }
            .result { margin-top: 20px; padding: 15px; background: #f0f8ff; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>♻️ Waste Classifier (Organic vs Anorganic)</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-box">
                <p>Pilih gambar sampah:</p>
                <input type="file" id="fileInput" accept="image/*" required />
                <img id="preview" />
            </div>
            <button type="submit">Prediksi</button>
        </form>
        <div id="result"></div>

        <script>
            // Preview gambar
            document.getElementById('fileInput').onchange = function(e) {
                const [file] = e.target.files;
                if (file) {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const img = document.getElementById('preview');
                        img.src = reader.result;
                        img.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            };

            // Kirim ke API
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);

                document.getElementById('result').innerHTML = '<p>Memproses...</p>';

                try {
                    const res = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    if (res.ok) {
                        document.getElementById('result').innerHTML = 
                            `<div class="result"><strong>Hasil Prediksi:</strong><br>` +
                            `${data.class} (${(data.confidence * 100).toFixed(1)}% confidence)</div>`;
                    } else {
                        throw new Error(data.detail || 'Error');
                    }
                } catch (err) {
                    document.getElementById('result').innerHTML = 
                        `<p style="color:red;">Error: ${err.message}</p>`;
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validasi tipe file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    
    # Simpan sementara
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Prediksi
        class_name, confidence = predict_image(file_path)
        return {"class": class_name, "confidence": float(confidence)}
    finally:
        # Hapus file sementara
        if os.path.exists(file_path):
            os.remove(file_path)