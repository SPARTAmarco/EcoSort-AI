#!/usr/bin/env python3
"""
EcoSort AI — Server Flask Lite (TFLite / ai-edge-litert)
Versione leggera senza TensorFlow completo: usa ai-edge-litert.
"""

import os
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import time

app = Flask(__name__)

# ── Configurazione ──────────────────────────────────────────────────────────
IMG_SIZE          = 224

CLASSI = {
    0: "carta_e_cartone",
    1: "plastica",
    2: "vetro_e_metallo"
}
LABELS = {
    "carta_e_cartone": ("Carta e Cartone", "#3B82F6"),
    "plastica":        ("Plastica",        "#F59E0B"),
    "vetro_e_metallo": ("Vetro e Metallo", "#10B981"),
    "indifferenziata": ("Indifferenziata", "#9CA3AF"),
}

# ── Stato globale ────────────────────────────────────────────────────────────
interpreter    = None
input_details  = None
output_details = None

# ── Caricamento modello TFLite ───────────────────────────────────────────────
def carica_modello():
    global interpreter, input_details, output_details

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "newbest_model.tflite"
    )
    print(f"[EcoSort] Caricamento modello TFLite: {model_path}")

    # Prova ai-edge-litert (pacchetto incluso nel python-embedded)
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=model_path)
        print("[EcoSort] Runtime: ai-edge-litert")
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=model_path)
            print("[EcoSort] Runtime: tflite-runtime")
        except ImportError:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=model_path)
            print("[EcoSort] Runtime: tensorflow lite (fallback)")

    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"[EcoSort] Modello pronto. "
          f"Input shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
    # Diagnostica quantizzazione
    print(f"[EcoSort] Input dtype:         {input_details[0]['dtype']}")
    print(f"[EcoSort] Input quantization:  {input_details[0]['quantization']}")


def preprocessa_tflite(img_bytes: bytes) -> np.ndarray:
    """
    Preprocessing per TFLite INT8 con input uint8:
    resize 224x224, nessuna normalizzazione — il runtime gestisce internamente
    la riscalatura tramite zero_point e scale registrati durante la calibrazione.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    return np.expand_dims(arr, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


# ── Endpoint /ping ───────────────────────────────────────────────────────────
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "model_loaded": interpreter is not None})


# ── Endpoint /classifica ─────────────────────────────────────────────────────
@app.route("/classifica", methods=["POST"])
def classifica():
    try:
        data = request.get_json(force=True)
        if "image" not in data:
            return jsonify({"errore": "Campo 'image' mancante nel body JSON"}), 400

        img_bytes = base64.b64decode(data["image"])

        # Input tensor (TFLite INT8 si aspetta uint8 grezzo [0, 255])
        interpreter.set_tensor(input_details[0]['index'], preprocessa_tflite(img_bytes))

        # Inferenza
        t0 = time.perf_counter()
        interpreter.invoke()
        elapsed = (time.perf_counter() - t0) * 1000

        # Output tensor
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Determina se già probabilità (softmax) o logit raw
        if abs(float(output.sum()) - 1.0) < 0.02:
            probs = output.astype(np.float64)
        else:
            probs = softmax(output.astype(np.float64))

        best_idx  = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        tutte     = {CLASSI[i]: round(float(probs[i]), 6) for i in range(len(probs))}

        classe_id    = CLASSI[best_idx]
        nome, colore = LABELS[classe_id]

        return jsonify({
            "categoria":    classe_id,
            "nome":         nome,
            "colore":       colore,
            "confidenza":   round(best_prob, 6),
            "tempo_ms":     round(elapsed, 1),
            "tutte":        tutte,
            "sotto_soglia": False,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"errore": str(e)}), 500


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    carica_modello()
    app.run(host="127.0.0.1", port=5891, debug=False)
