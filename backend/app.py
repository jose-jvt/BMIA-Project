from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
import uuid
from datetime import datetime
import subprocess

from processors.dsi_processor import DSIProcessor

app = Flask(__name__)
CORS(app, origins=['http://localhost:8080', 'file://'])

# Config
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
for subdir in ['connectivity_matrix', 'pytorch_model', 'dmri']:
    os.makedirs(os.path.join(UPLOAD_FOLDER, subdir), exist_ok=True)

jobs = {}

@app.route('/api/export', methods=['POST'])
def export_to_cm():
    bval_path = None
    bvec_path = None

    atlas = request.json['processingMethod']
    if not atlas:
        return jsonify({"error": "Atlas is required"}), 400

    dsi_path = request.json['dsiPath']
    if not dsi_path:
        return jsonify({"error": "DSI Studio path is required"}), 400

    file_path = request.json['dmriFile']
    if not file_path:
        return jsonify({"error": "dMRI file is required"}), 400

    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        bval_name = request.json['bval']
        bvec_name = request.json['bvec']
        if not bval_name or not bvec_name:
            return jsonify({"error": "When processing NIFTI files .bval and .bvec are required"}), 400

        bval_path = os.path.join(os.path.dirname(file_path), bval_name)
        bvec_path = os.path.join(os.path.dirname(file_path), bvec_name)

    pipeline = DSIProcessor(
        dsi_studio_path=dsi_path,
        input_path=file_path,
        output_prefix=os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split('.')[0]+'_CM'),
        atlas=atlas,
        bval_file=bval_path,  # Only for NIfTI
        bvec_file=bvec_path  # Only for NIfTI
    )

    pipeline.run()

    return jsonify({"message": f"Connectivity matrix exported at {os.path.dirname(file_path)}",}), 200

@app.route('/api/execute_model', methods=['POST'])
def execute_model():
    print(request.json)
    return jsonify({"message": "PyTorch model initiated"}), 200


if __name__ == '__main__':
    print('Python server running at http://localhost:3000')
    app.run(debug=True, host='0.0.0.0', port=3000)

