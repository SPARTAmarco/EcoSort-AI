#!/usr/bin/env python3
"""
EcoSort AI — Server Flask locale
Gira in background, riceve immagini da JavaFX e risponde con la classificazione.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silenzia TensorFlow

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import time

app = Flask(__name__)

# ── Configurazione ──────────────────────────────────────────────────────────
IMG_SIZE  = 224
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



# ── Stato globale ───────────────────────────────────────────────────────────
model           = None
usa_softmax_raw = False   # True se il modello non ha Softmax nel layer finale

# ── Patch compatibilità Keras ───────────────────────────────────────────────
def _patcha_keras():
    """
    Monkey-patch per compatibilità con modelli salvati con Keras 3.x
    che includono 'quantization_config' nel config dei layer Dense/Conv2D.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers as kl

        _orig_dense = kl.Dense.__init__
        def _dense_init(self, *args, quantization_config=None, **kwargs):
            _orig_dense(self, *args, **kwargs)
        kl.Dense.__init__ = _dense_init

        _orig_conv = kl.Conv2D.__init__
        def _conv_init(self, *args, quantization_config=None, **kwargs):
            _orig_conv(self, *args, **kwargs)
        kl.Conv2D.__init__ = _conv_init

        print("[EcoSort] Patch Keras applicata (compatibilità quantization_config)")
    except Exception as e:
        print(f"[EcoSort] Patch Keras saltata: {e}")


# ── Caricamento modello ─────────────────────────────────────────────────────
def carica_modello():
    global model, usa_softmax_raw
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newbest_model.keras")
    print(f"[EcoSort] Caricamento modello: {model_path}")

    _patcha_keras()
    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[EcoSort] Modello pronto (compile=False)")
    except Exception as e1:
        print(f"[EcoSort] Primo tentativo fallito ({e1}), provo con custom_objects={{}}...")
        try:
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})
            print("[EcoSort] Modello pronto (custom_objects)")
        except Exception as e2:
            raise RuntimeError(f"Impossibile caricare il modello: {e2}") from e2

    # Controlla se l'ultimo layer è già Softmax o meno
    last_layer = model.layers[-1]
    last_cfg   = last_layer.get_config()
    activation = last_cfg.get("activation", "")
    if isinstance(activation, dict):
        activation = activation.get("class_name", "")
    activation = str(activation).lower()

    usa_softmax_raw = ("softmax" not in activation)
    print(f"[EcoSort] Layer finale: {last_layer.__class__.__name__} | "
          f"activation={activation} | applica_softmax={usa_softmax_raw}")


def preprocessa(img_bytes: bytes) -> np.ndarray:
    """
    Solo resize a 224x224: la normalizzazione di EfficientNetB0 e' incorporata
    nel modello, che si aspetta pixel grezzi in [0, 255]. Stesso percorso del Pi
    (classifica_pi.py): se i due divergessero, il modello sbaglierebbe in silenzio.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)   # [0, 255], nessuna scalatura
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


# ── Softmax numericamente stabile ───────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


# ── Endpoint /ping ──────────────────────────────────────────────────────────
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ── Endpoint /classifica ────────────────────────────────────────────────────
@app.route("/classifica", methods=["POST"])
def classifica():
    try:
        data = request.get_json(force=True)
        if "image" not in data:
            return jsonify({"errore": "Campo 'image' mancante nel body JSON"}), 400

        img_bytes = base64.b64decode(data["image"])
        arr       = preprocessa(img_bytes)

        # Inferenza
        t0     = time.perf_counter()
        output = model.predict(arr, verbose=0)[0]   # shape: (3,)
        elapsed = (time.perf_counter() - t0) * 1000

        # Converti in probabilità reali
        if usa_softmax_raw:
            probs = softmax(output.astype(np.float64))
        else:
            # L'ultimo layer ha già Softmax, usiamo direttamente le probabilità predette
            probs = output.astype(np.float64)

        best_idx, probs, sotto = decidi(probs)
        best_prob = float(probs[best_idx])
        tutte     = {CLASSI[i]: round(float(probs[i]), 6) for i in range(len(probs))}

        classe_id = CLASSI[best_idx]
        nome, colore = LABELS[classe_id]

        return jsonify({
            "categoria":  classe_id,
            "nome":       nome,
            "colore":     colore,
            "confidenza": round(best_prob, 6),
            "tempo_ms":   round(elapsed, 1),
            "tutte":      tutte,
            "sotto_soglia": sotto,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"errore": str(e)}), 500


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    carica_modello()
    carica_regola()
    app.run(host="127.0.0.1", port=5891, debug=False)
