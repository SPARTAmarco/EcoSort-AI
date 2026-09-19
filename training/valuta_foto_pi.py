#!/usr/bin/env python3
"""
EcoSort AI — Valuta un modello GIA ADDESTRATO sulle foto reali della scatola.

Serve quando le foto del Pi arrivano dopo il training: dice quanto vale
davvero il modello nelle condizioni di lavoro, senza riaddestrare niente.
Gira ovunque ci sia TensorFlow (Colab, PC) e accetta sia .keras sia .tflite.

    python3 valuta_foto_pi.py --modello newbest_model.keras --foto foto_pi
    python3 valuta_foto_pi.py --modello rifiuti.tflite --foto foto_pi --config config.json

Con --ricalibra ristima la temperatura su meta delle foto e la applica
all'altra meta: e' il modo corretto di adattare la calibrazione al dominio
reale senza barare (la T non puo' essere stimata sugli stessi dati su cui poi
dichiari i risultati).
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image

from ecosort_decisione import (COSTO, CLASSI, AZIONI, azione_ottima, costo_medio,
                               costo_medio_argmax, costo_medio_soglia, applica_temperatura,
                               stima_temperatura, errore_calibrazione, gravi, riepilogo)

ESTENSIONI = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def carica_foto(cartella, img_size):
    """Stesso preprocessing di classifica_pi.py: solo resize, niente altro."""
    x, y, nomi = [], [], []
    for i, c in enumerate(CLASSI):
        d = os.path.join(cartella, c)
        if not os.path.isdir(d):
            sys.exit(f"Manca la sottocartella {d}")
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(ESTENSIONI):
                with Image.open(os.path.join(d, f)) as im:
                    im = im.convert('RGB').resize(img_size, Image.BILINEAR)
                    x.append(np.asarray(im, dtype=np.uint8))
                y.append(i)
                nomi.append(os.path.join(c, f))
    if not x:
        sys.exit(f"Nessuna immagine in {cartella}")
    return np.stack(x), np.array(y), nomi


def apri_modello(percorso):
    """Ritorna (funzione_di_predizione, dimensione_input).
    La dimensione la chiede al modello invece di darla per scontata: se un
    giorno riaddestri a 192 o 256, questo script continua a funzionare."""
    if percorso.endswith('.tflite'):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=percorso, num_threads=4)
        interp.allocate_tensors()
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
        h, w = int(inp['shape'][1]), int(inp['shape'][2])

        def f(x):
            probs = np.zeros((len(x), 3), dtype=np.float64)
            for i, img in enumerate(x):
                interp.set_tensor(inp['index'], img[None, ...].astype(inp['dtype']))
                interp.invoke()
                probs[i] = interp.get_tensor(out['index'])[0]
            return probs
        return f, (w, h)

    # .keras: il preprocessing e' dentro il backbone (EfficientNet/MobileNetV3
    # vogliono [0,255]), quindi si passa l'uint8 cosi com'e, in float32
    from tensorflow import keras
    m = keras.models.load_model(percorso)
    forma = m.input_shape
    h, w = int(forma[1]), int(forma[2])
    return (lambda x: m.predict(x.astype('float32'), verbose=0).astype(np.float64)), (w, h)


def stampa_report(y, probs, T, titolo):
    from sklearn.metrics import classification_report, confusion_matrix
    p = applica_temperatura(probs, T)
    az, _ = azione_ottima(p)
    print(f"\n{'='*70}\n  {titolo}  (T = {T:.3f})\n{'='*70}")
    print(classification_report(y, p.argmax(1), target_names=CLASSI, digits=4, zero_division=0))
    print("Confusione argmax (riga = reale):\n", confusion_matrix(y, p.argmax(1), labels=[0, 1, 2]))
    print("Confusione con regola a costo (4a colonna = indifferenziata):\n",
          confusion_matrix(y, az, labels=[0, 1, 2, 3])[:3, :])
    r = riepilogo(y, probs, T, etichetta=titolo)
    print(f"\n  accuratezza          : {r['accuracy']*100:.2f}%")
    print(f"  ECE                  : {r['ece_prima']:.4f} -> {r['ece_dopo']:.4f}")
    print(f"  costo argmax         : {r['costo_argmax']:.4f}")
    print(f"  costo soglia 0.70    : {r['costo_soglia70']:.4f}")
    print(f"  costo regola a costo : {r['costo_decisione']:.4f}   <-- quello che conta")
    print(f"  coverage             : {r['coverage']*100:.1f}%  "
          f"(il resto va in indifferenziata)")
    print(f"  errori gravi         : {r['gravi_argmax']} (argmax) -> {r['gravi_decisione']} (a costo)")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modello', required=True, help='.keras oppure .tflite')
    ap.add_argument('--foto', default='foto_pi', help='cartella con le tre sottocartelle')
    ap.add_argument('--config', default='config.json', help='per leggere la temperatura')
    ap.add_argument('--ricalibra', action='store_true',
                    help='ristima T su meta delle foto e valuta sull altra meta')
    ap.add_argument('--peggiori', type=int, default=10, help='quanti errori peggiori elencare')
    a = ap.parse_args()

    T = 1.0
    if os.path.exists(a.config):
        T = float(json.load(open(a.config)).get('temperatura', 1.0))
        print(f"Temperatura da {a.config}: {T:.3f}")

    predici, img_size = apri_modello(a.modello)
    print(f"Input del modello: {img_size[0]}x{img_size[1]}")
    x, y, nomi = carica_foto(a.foto, img_size)
    print(f"{len(x)} foto  {dict(zip(CLASSI, np.bincount(y, minlength=3).tolist()))}")
    probs = predici(x)

    r = stampa_report(y, probs, T, f"{os.path.basename(a.modello)} sulle foto della scatola")

    if a.ricalibra:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        meta = len(y) // 2
        cal, val = idx[:meta], idx[meta:]
        T2, _ = stima_temperatura(probs[cal], y[cal])
        print(f"\nTemperatura ristimata sulle foto della scatola: {T:.3f} -> {T2:.3f}")
        stampa_report(y[val], probs[val], T2, "meta di controllo, temperatura del dominio reale")
        print("\nSe il costo migliora, aggiorna 'temperatura' in config.json con "
              f"{T2:.3f}: e' l'unico numero da cambiare, il modello resta lo stesso.")

    # gli errori che costano di piu: e' li che si capisce cosa manca al dataset
    p = applica_temperatura(probs, T)
    az, _ = azione_ottima(p)
    costi = COSTO[y, az]
    ordine = np.argsort(-costi)[:a.peggiori]
    print(f"\n{'='*70}\n  I {a.peggiori} errori piu costosi\n{'='*70}")
    for i in ordine:
        if costi[i] == 0:
            continue
        print(f"  costo {costi[i]:.1f}  {nomi[i]:<45} -> {AZIONI[az[i]]:<16} "
              f"(p: " + " ".join(f"{c[:5]} {p[i][j]*100:.0f}%" for j, c in enumerate(CLASSI)) + ")")


if __name__ == '__main__':
    main()
