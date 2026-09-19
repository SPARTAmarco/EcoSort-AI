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
SOGLIA_CONFIDENZA = 0.73   # Solo se manca config.json: miglior soglia fissa del benchmark

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
RUNTIME        = "TFLite"
input_details  = None
output_details = None

# ── Caricamento modello TFLite ───────────────────────────────────────────────
def carica_modello():
    global interpreter, input_details, output_details, RUNTIME

    qui = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(qui, "rifiuti.tflite")            # Release v1.0.0 (float16)
    if not os.path.exists(model_path):
        model_path = os.path.join(qui, "newbest_model.tflite")  # nome vecchio
    print(f"[EcoSort] Caricamento modello TFLite: {model_path}")

    # Prova ai-edge-litert (pacchetto incluso nel python-embedded)
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=model_path, num_threads=4)
        RUNTIME = "TFLite \u00b7 LiteRT"
        print("[EcoSort] Runtime: ai-edge-litert")
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=model_path)
            RUNTIME = "TFLite \u00b7 tflite-runtime"
            print("[EcoSort] Runtime: tflite-runtime")
        except ImportError:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=model_path)
            RUNTIME = "TFLite \u00b7 TensorFlow"
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
    Solo resize a 224x224: la normalizzazione di EfficientNetB0 e' dentro il
    modello, che si aspetta pixel grezzi [0, 255]. Il dtype (float32 per il
    float16, uint8 per l'INT8) si legge dall'interprete, come fa il Pi.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype(input_details[0]['dtype'])
    return np.expand_dims(arr, axis=0)


# ── Regola di decisione (identica al Raspberry Pi) ──────────────────────────
# Si legge config.json della Release v1.0.0: temperatura di calibrazione e
# matrice di costo. Stessa formula di training/ecosort_decisione.py:
#     azione* = argmin_a  SUM_c  P(c|x) * Costo[c][a]
# con "indifferenziata" come quarta azione possibile. Se config.json manca si
# ripiega sulla soglia fissa SOGLIA_CONFIDENZA.
AZIONI = ["carta_e_cartone", "plastica", "vetro_e_metallo", "indifferenziata"]
REGOLA = {"temperatura": 1.0, "costo": None}


def carica_regola():
    import json
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if not os.path.exists(cfg_path):
        print(f"[EcoSort] config.json assente: uso la soglia fissa {SOGLIA_CONFIDENZA}")
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    REGOLA["temperatura"] = float(cfg.get("temperatura", 1.0))
    if cfg.get("usa_regola_costo", True) and "matrice_costo" in cfg:
        REGOLA["costo"] = np.array(cfg["matrice_costo"], dtype=np.float64)
    print(f"[EcoSort] Regola: {'costo atteso' if REGOLA['costo'] is not None else 'soglia fissa'}"
          f" | temperatura {REGOLA['temperatura']:.3f}")


def decidi(probs: np.ndarray):
    """Ritorna (classe, probabilita calibrate, sotto_soglia)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    p = np.exp(np.log(p) / REGOLA["temperatura"])
    p = p / p.sum()
    if REGOLA["costo"] is not None:
        azione = int((p @ REGOLA["costo"]).argmin())
        if azione == 3:
            return int(p.argmax()), p, True
        return azione, p, False
    best = int(p.argmax())
    return best, p, bool(p[best] < SOGLIA_CONFIDENZA)


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

        # Input tensor: pixel grezzi [0, 255] nel dtype del modello
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

        best_idx, probs, sotto = decidi(probs)
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
            "sotto_soglia": sotto,
            "motore":       RUNTIME,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"errore": str(e)}), 500


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    carica_modello()
    carica_regola()
    app.run(host="127.0.0.1", port=5891, debug=False)
