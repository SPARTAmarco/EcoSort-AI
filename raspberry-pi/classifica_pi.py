"""
EcoSort AI — Inferenza sul Raspberry Pi 4B con LiteRT (TFLite).

INSTALLAZIONE (Debian Trixie, Python 3.13, venv non attivo):

    pip install ai-edge-litert numpy pillow --break-system-packages

NOTA: NON usare `pip install tflite-runtime`. Quel pacchetto e' fermo alla
2.14.0 di ottobre 2023 e ha wheel solo fino a Python 3.11: sul tuo Pi non si
installa. Il successore e' ai-edge-litert, che ha wheel manylinux aarch64
per cp310-cp314.

File da avere nella stessa cartella:
    rifiuti.tflite   config.json   ecosort_decisione.py

USO
    python3 classifica_pi.py --bench            misura la latenza reale
    python3 classifica_pi.py --foto prova.jpg   classifica un file
    python3 classifica_pi.py                    INVIO per scattare con la camera
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image

from ecosort_decisione import COSTO, CLASSI, AZIONI, BIDONI, azione_ottima

# ai-edge-litert e' il pacchetto attuale; gli altri due sono fallback per
# ambienti piu vecchi, cosi lo script non muore per un problema di packaging.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError:
            sys.exit("Nessun runtime TFLite. Installa:\n"
                     "  pip install ai-edge-litert --break-system-packages")

QUI = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(QUI, 'config.json')))
MODELLO = os.path.join(QUI, cfg.get('modello_tflite', 'rifiuti.tflite'))
COSTO_M = np.array(cfg.get('matrice_costo', COSTO.tolist()))
USA_COSTO = cfg.get('usa_regola_costo', True)
SOGLIA = cfg.get('soglia_fallback', 0.70)
IMG_SIZE = (224, 224)

# 4 thread = i 4 core del Pi 4B. Di default ne userebbe uno solo e la latenza
# triplica: e' l'ottimizzazione piu redditizia dell'intero deployment.
interp = Interpreter(model_path=MODELLO, num_threads=4)
interp.allocate_tensors()
INP = interp.get_input_details()[0]
OUTP = interp.get_output_details()[0]
DTYPE = INP['dtype']

print(f"Modello: {os.path.basename(MODELLO)} "
      f"({cfg.get('quantizzazione','?')}, input {np.dtype(DTYPE).name})")
print(f"Regola: {'costo atteso' if USA_COSTO else f'soglia {SOGLIA}'}")


def prepara(img):
    """Nessuna normalizzazione: il preprocessing e' dentro il modello.
    Qui si fa solo resize. Se questa funzione e il training divergessero,
    il modello sbaglierebbe in modo silenzioso e inspiegabile."""
    img = img.convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
    return np.asarray(img)[None, ...].astype(DTYPE)


def classifica(img):
    x = prepara(img)
    t0 = time.perf_counter()
    interp.set_tensor(INP['index'], x)
    interp.invoke()
    probs = interp.get_tensor(OUTP['index'])[0].astype(np.float64)
    ms = (time.perf_counter() - t0) * 1000

    if USA_COSTO:
        az, costi = azione_ottima(probs[None, :], COSTO_M)
        return int(az[0]), probs, costi[0], ms
    az = int(probs.argmax()) if probs.max() >= SOGLIA else 3
    return az, probs, None, ms


def stampa(azione, probs, costi, ms):
    print(f"\n>>> {AZIONI[azione].upper()}  ->  bidone {BIDONI[azione]}   ({ms:.0f} ms)")
    for i, c in enumerate(CLASSI):
        print(f"    {c:>18} {probs[i]*100:6.2f}%  {'#' * int(probs[i] * 30)}")
    if costi is not None:
        print("    costo atteso: " + "  ".join(
            f"{AZIONI[a][:12]}={costi[a]:.2f}" for a in range(4)))


def bench(n=30):
    x = np.random.randint(0, 255, (1, *IMG_SIZE, 3)).astype(DTYPE)
    for _ in range(5):
        interp.set_tensor(INP['index'], x); interp.invoke()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        interp.set_tensor(INP['index'], x); interp.invoke()
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    print(f"\nLatenza su {n} inferenze (4 thread):")
    print(f"  media {t.mean():.0f} ms | mediana {np.median(t):.0f} ms | "
          f"p95 {np.percentile(t,95):.0f} ms | min {t.min():.0f} ms")
    print(f"  budget 2-3 s: {'OK' if np.percentile(t,95) < 2000 else 'FUORI BUDGET'}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--foto')
    a = ap.parse_args()

    if a.bench:
        bench()
    elif a.foto:
        stampa(*classifica(Image.open(a.foto)))
    else:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (1280, 960)}))
        cam.start(); time.sleep(2)
        print("\nINVIO per scattare, Ctrl+C per uscire")
        try:
            while True:
                input()
                stampa(*classifica(Image.fromarray(cam.capture_array())))
        except KeyboardInterrupt:
            cam.stop()
            print("\nchiuso")
