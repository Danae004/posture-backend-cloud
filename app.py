# app.py - Versión Cloud Optimizada
import base64
import logging
import os
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# Configurar logging para la nube
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Variables globales para modelos
modelo = None
encoder = None
pose = None
current_reference_sequence = []    
reference_fps = 1                  
current_reference_index = 0        
current_tolerance = 1.0            

def load_models():
    """Cargar modelos al iniciar la app - VERSIÓN CLOUD"""
    global modelo, encoder, pose
    
    try:
        # Cargar modelo de posturas
        modelo = joblib.load('modelo_posturas.pkl')
        logger.info("✅ modelo_posturas.pkl cargado")
        
        # Cargar encoder
        encoder = joblib.load('encoder.pkl')
        logger.info("✅ encoder.pkl cargado")
        
        # Configurar MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5)
        logger.info("✅ MediaPipe Pose cargado")
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {e}")
        raise e

# Cargar modelos al inicio
load_models()

def evaluate_posture(landmarks, posture_label, reference_landmarks=None, tolerance_scale=1.0):
    """Evaluación mejorada que devuelve 'Bien' o 'Mal' y una breve sugerencia."""
    if not landmarks or not posture_label:
        return 'Sin evaluación', 'No hay datos suficientes', {'avg_distance': None, 'max_distance': None}

    # Convertir la lista plana a pares (x,y,z) por punto
    pts = [(landmarks[i], landmarks[i+1], landmarks[i+2]) for i in range(0, len(landmarks), 3)]

    # Índices comunes en MediaPipe Pose
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    NOSE = 0

    feedback = 'Bien'
    reason = 'Postura dentro de parámetros esperados'

    try:
        ls = pts[LEFT_SHOULDER]
        rs = pts[RIGHT_SHOULDER]
        lh = pts[LEFT_HIP]
        rh = pts[RIGHT_HIP]
        nose = pts[NOSE]
        
        # Si hay landmarks de referencia, calcular similitud con umbrales CALIBRADOS
        avg_distance = None
        max_distance = None
        if reference_landmarks and len(reference_landmarks) == len(landmarks):
            ref_pts = [(reference_landmarks[i], reference_landmarks[i+1], reference_landmarks[i+2]) 
                       for i in range(0, len(reference_landmarks), 3)]
            
            # Calcular distancia euclidiana promedio entre landmarks
            distances = []
            for i in range(min(len(pts), len(ref_pts))):
                p = pts[i]
                ref_p = ref_pts[i]
                dist = np.sqrt((p[0] - ref_p[0])**2 + (p[1] - ref_p[1])**2 + (p[2] - ref_p[2])**2)
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            
            # Ajustar umbrales según factor de tolerancia
            t = float(tolerance_scale)
            if avg_distance < 0.05 * t and max_distance < 0.12 * t:
                feedback = 'Bien'
                similarity_pct = max(0, 100 - int(avg_distance * 1000))
                reason = f'✓ Excelente coincidencia con la referencia ({similarity_pct}%)'
            elif avg_distance < 0.09 * t and max_distance < 0.18 * t:
                feedback = 'Bien'
                similarity_pct = max(0, 100 - int(avg_distance * 1000))
                reason = f'✓ Buena postura, muy cercana a la referencia ({similarity_pct}%)'
            elif avg_distance < 0.15 * t and max_distance < 0.30 * t:
                feedback = 'Mal'
                reason = f'⚠ La postura se desvía. Ajusta tu alineación.'
            else:
                feedback = 'Mal'
                reason = f'✗ La postura es muy diferente. Intenta nuevamente.'
        else:
            # Heurísticas especializadas si no hay referencia
            shoulder_avg_y = (ls[1] + rs[1]) / 2
            hip_avg_y = (lh[1] + rh[1]) / 2
            shoulder_avg_x = (ls[0] + rs[0]) / 2
            hip_avg_x = (lh[0] + rh[0]) / 2
            
            trunk_tilt_y = abs(hip_avg_y - shoulder_avg_y)
            trunk_tilt_x = abs(hip_avg_x - shoulder_avg_x)
            
            shoulder_diff = abs(ls[1] - rs[1])
            hip_diff = abs(lh[1] - rh[1])

            if posture_label == 'espondilolisis':
                if trunk_tilt_y > 0.03 or trunk_tilt_x > 0.05:
                    feedback = 'Bien'
                    reason = '✓ Curvatura de espondilolisis detectada correctamente'
                else:
                    feedback = 'Mal'
                    reason = '✗ No hay suficiente curvatura. Aumenta la flexión de la columna.'
            
            elif posture_label == 'lumbalgia mecánica inespecífica':
                if abs(trunk_tilt_y) < 0.05 and abs(trunk_tilt_x) < 0.04:
                    feedback = 'Bien'
                    reason = '✓ Alineación óptima para lumbalgia mecánica'
                else:
                    feedback = 'Mal'
                    reason = '✗ Desalineación. Alinea hombros directamente sobre caderas.'
            
            elif posture_label == 'escoliosis lumbar':
                if shoulder_diff < 0.02 and hip_diff < 0.02:
                    feedback = 'Bien'
                    reason = '✓ Postura simétrica, buena para escoliosis'
                else:
                    feedback = 'Mal'
                    reason = f'✗ Inclinación lateral detectada. Nivela hombros y pelvis.'
            
            elif posture_label == 'hernia de disco lumbar':
                if trunk_tilt_y < 0.08 and abs(trunk_tilt_x) < 0.06:
                    feedback = 'Bien'
                    reason = '✓ Posición neutra segura para hernia de disco'
                else:
                    feedback = 'Mal'
                    reason = '✗ Posición de riesgo. Mantén la espalda neutral y apoyada.'
    
    except Exception as e:
        return 'Sin evaluación', f'Error al calcular landmarks: {str(e)}', {'avg_distance': None, 'max_distance': None}

    return feedback, reason, {'avg_distance': float(avg_distance) if avg_distance is not None else None,
                              'max_distance': float(max_distance) if max_distance is not None else None}

def classify_posture(frame):
    """Clasificar postura en un frame - VERSIÓN CLOUD"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        # Extraer landmarks en forma plana [x,y,z,...]
        landmarks = [coord for landmark in results.pose_landmarks.landmark
                     for coord in [landmark.x, landmark.y, landmark.z]]

        # Predecir postura
        try:
            posture = encoder.inverse_transform(modelo.predict([landmarks]))[0]
        except Exception:
            posture = None

        return frame, posture, landmarks

    return frame, None, None

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    status = "healthy" if modelo is not None else "models_not_loaded"
    return jsonify({
        "status": status,
        "message": "Backend de análisis postural - Cloud",
        "models_loaded": modelo is not None
    })

# Endpoint principal para evaluación de frames
@app.route('/api/evaluate_frame', methods=['POST'])
def evaluate_frame():
    """Endpoint para evaluar una imagen enviada por la app móvil."""
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'success': False, 'message': 'No JSON payload'}), 400

        img_b64 = payload.get('image')
        if not img_b64:
            return jsonify({'success': False, 'message': 'No image provided'}), 400

        # Soporta data URI o solo base64
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]

        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image data'}), 400

        # Usar las funciones existentes para clasificar y evaluar
        frame_proc, posture, landmarks = classify_posture(frame)

        # Seleccionar referencia sincronizada si existe
        ref_landmarks_to_use = None
        if current_reference_sequence:
            idx = min(max(0, current_reference_index), len(current_reference_sequence)-1)
            ref_landmarks_to_use = current_reference_sequence[idx]

        feedback_label, feedback_reason, metrics = evaluate_posture(
            landmarks, posture, ref_landmarks_to_use, tolerance_scale=current_tolerance
        )

        return jsonify({
            'success': True,
            'posture': posture,
            'feedback': feedback_label,
            'reason': feedback_reason,
            'metrics': metrics
        })

    except Exception as e:
        logger.error(f"Error en evaluate_frame: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/sync_reference_time', methods=['POST'])
def sync_reference_time_http():
    """Endpoint HTTP para sincronizar tiempo del video (para app móvil)."""
    global current_reference_index, reference_fps
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'success': False}), 400
        
        t = float(payload.get('current_time', 0))
        idx = int(round(t * reference_fps))
        current_reference_index = max(0, idx)
        
        return jsonify({'success': True, 'index': current_reference_index})
    except Exception as e:
        logger.error(f"Error en sync_reference_time: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/set_tolerance', methods=['POST'])
def set_tolerance():
    """Endpoint para ajustar tolerancia."""
    global current_tolerance
    try:
        payload = request.get_json()
        t = float(payload.get('tolerance', 1.0))
        current_tolerance = max(0.3, min(3.0, t))
        return jsonify({'success': True, 'tolerance': current_tolerance})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Endpoint simple para verificar que el servidor está vivo
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Backend de Análisis Postural",
        "status": "running",
        "version": "1.0-cloud"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)    