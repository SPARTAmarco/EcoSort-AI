"""
EcoSort AI — Conversione in TFLite/LiteRT con verifica dell'accuratezza.

Converte newbest_model.keras in tre varianti, le valuta TUTTE sul test set e
sceglie la migliore. La quantizzazione non e' gratis: puo' costare punti di
accuratezza, e nessuno se ne accorge se non la si misura.

Due scelte progettuali importanti:

1. IL PREPROCESSING VA DENTRO IL MODELLO. Il .tflite accetta direttamente
   l'immagine uint8 [0,255] che esce dalla camera. Cosi e' impossibile che il
   preprocessing sul Pi diverga da quello usato in training: e' il bug numero
   uno del deployment e sparisce per costruzione.

2. L'OUTPUT RESTA float32 anche nella versione INT8. Le probabilita' servono
   alla regola di decisione a costo: quantizzarle a 8 bit significa una
   risoluzione di 1/256, che degrada le soglie implicite.

    python3 converti_tflite.py
"""
import os, json, sys, time
import numpy as np
import tensorflow as tf
from tensorflow import keras

import ecosort_benchmark as B
from ecosort_decisione import COSTO, CLASSI, azione_ottima, costo_medio, costo_medio_argmax

# Importare il benchmark attiva mixed_float16 (serve al training su T4).
# Per la conversione serve float32 puro: in float16 i range di quantizzazione
# vengono calcolati su tensori gia degradati e l'INT8 esce peggiore.
keras.mixed_precision.set_global_policy('float32')

OUT = B.OUT_DIR
N_REPRESENTATIVE = 300     # immagini per calibrare i range di quantizzazione


def wrappa_con_preprocessing(model, nome_backbone):
    """Incorpora il preprocessing nel grafo: input uint8 [0,255] -> probabilita."""
    prep = B.BACKBONE[nome_backbone]['prep']
    inp = keras.Input(shape=(*B.IMG_SIZE, 3), dtype='float32', name='immagine')
    x = prep(inp)
    out = model(x, training=False)
    return keras.Model(inp, out, name='ecosort_deploy')


def _converter(wrapper):
    """Keras 3 non sempre digerisce from_keras_model: fallback via SavedModel."""
    try:
        return tf.lite.TFLiteConverter.from_keras_model(wrapper)
    except Exception as e:
        print(f"  from_keras_model fallito ({type(e).__name__}), uso SavedModel")
        tmp = os.path.join(OUT, '_export_tmp')
        wrapper.export(tmp)
        return tf.lite.TFLiteConverter.from_saved_model(tmp)


def converti(wrapper, modo, rep_data=None):
    c = _converter(wrapper)
    if modo == 'float32':
        pass
    elif modo == 'float16':
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
    elif modo == 'int8':
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = rep_data
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                       tf.lite.OpsSet.TFLITE_BUILTINS]
        c.inference_input_type = tf.uint8      # l'immagine della camera, cosi com'e
        c.inference_output_type = tf.float32   # probabilita in piena precisione
    return c.convert()


def valuta_tflite(percorso, x_test, y_test, n_bench=20):
    interp = tf.lite.Interpreter(model_path=percorso, num_threads=4)
    interp.allocate_tensors()
    inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    dtype_in = inp_d['dtype']

    probs = np.zeros((len(x_test), 3), dtype=np.float64)
    for i, img in enumerate(x_test):
        x = img[None, ...]
        x = x.astype(np.uint8) if dtype_in == np.uint8 else x.astype(np.float32)
        interp.set_tensor(inp_d['index'], x)
        interp.invoke()
        probs[i] = interp.get_tensor(out_d['index'])[0]

    # latenza su CPU x86: proxy relativo, il numero vero si misura sul Pi
    x = x_test[:1].astype(np.uint8 if dtype_in == np.uint8 else np.float32)
    for _ in range(3):
        interp.set_tensor(inp_d['index'], x); interp.invoke()
    t0 = time.perf_counter()
    for _ in range(n_bench):
        interp.set_tensor(inp_d['index'], x); interp.invoke()
    ms = (time.perf_counter() - t0) / n_bench * 1000

    y_pred = probs.argmax(1)
    azioni, _ = azione_ottima(probs)
    return dict(
        accuracy=float((y_pred == y_test).mean()),
        costo_argmax=costo_medio_argmax(y_test, probs),
        costo_decisione=costo_medio(y_test, azioni),
        gravi=int(((y_test == 2) & np.isin(azioni, [0, 1])).sum()),
        mb=os.path.getsize(percorso) / 1e6,
        ms=ms,
    )


def main():
    cfg_path = os.path.join(OUT, 'config.json')
    cfg = json.load(open(cfg_path))
    nome = cfg['backbone']
    keras_path = os.path.join(OUT, cfg.get('modello_keras', 'newbest_model.keras'))
    if not os.path.exists(keras_path):
        sys.exit(f"Manca {keras_path}: esegui prima train_finale.py")

    print(f"Modello: {keras_path}  (backbone {nome})")
    model = keras.models.load_model(keras_path)
    wrapper = wrappa_con_preprocessing(model, nome)

    # dati: test set in uint8 [0,255], esattamente come arrivera dalla camera
    train_raw, val_raw, test_raw, _ = B.carica_split()
    x_test = np.stack([x.numpy() for x, _ in test_raw])
    y_test = B.etichette(test_raw)
    print(f"Test set: {len(x_test)} immagini")

    x_rep = np.stack([x.numpy() for x, _ in train_raw.take(N_REPRESENTATIVE)])

    def rep_data():
        # deve coprire la varieta reale del dataset: se calibri su 10 immagini
        # tutte chiare, i range di quantizzazione saranno sbagliati
        for i in range(len(x_rep)):
            yield [x_rep[i][None, ...].astype(np.float32)]

    risultati = {}
    for modo in ['float32', 'float16', 'int8']:
        print(f"\n--- conversione {modo} ---")
        try:
            blob = converti(wrapper, modo, rep_data if modo == 'int8' else None)
        except Exception as e:
            print(f"  FALLITA: {type(e).__name__}: {e}")
            continue
        percorso = os.path.join(OUT, f'rifiuti_{modo}.tflite')
        open(percorso, 'wb').write(blob)
        r = valuta_tflite(percorso, x_test, y_test)
        risultati[modo] = r
        print(f"  {r['mb']:.1f} MB | acc {r['accuracy']*100:.2f}% | "
              f"costo {r['costo_decisione']:.4f} | gravi {r['gravi']} | {r['ms']:.0f} ms (x86)")

    if not risultati:
        sys.exit("Nessuna conversione riuscita")

    base = risultati.get('float32')
    print(f"\n{'='*76}\n  CONFRONTO — quanto costa la quantizzazione\n{'='*76}")
    print(f"{'variante':<12}{'MB':>8}{'accuracy':>11}{'delta':>9}{'costo':>9}{'gravi':>8}{'ms x86':>9}")
    for m, r in risultati.items():
        d = (r['accuracy'] - base['accuracy']) * 100 if base else 0.0
        print(f"{m:<12}{r['mb']:>8.1f}{r['accuracy']*100:>10.2f}%{d:>+9.2f}"
              f"{r['costo_decisione']:>9.4f}{r['gravi']:>8}{r['ms']:>9.0f}")

    # Scelta: INT8 solo se il risparmio non costa accuratezza. Su un progetto
    # dove il vetro nel bidone sbagliato pesa 9.0, mezzo punto di accuratezza
    # vale piu di 3 MB di file.
    scelta = 'float32'
    for m in ['int8', 'float16']:
        if m in risultati and base:
            perdita = (base['accuracy'] - risultati[m]['accuracy']) * 100
            peggior_costo = risultati[m]['costo_decisione'] - base['costo_decisione']
            if perdita <= 1.0 and peggior_costo <= 0.02:
                scelta = m
                break
    finale = os.path.join(OUT, 'rifiuti.tflite')
    with open(finale, 'wb') as f:
        f.write(open(os.path.join(OUT, f'rifiuti_{scelta}.tflite'), 'rb').read())

    r = risultati[scelta]
    print(f"\nSCELTA: {scelta}  ->  rifiuti.tflite  ({r['mb']:.1f} MB)")
    if scelta == 'float32':
        print("  INT8 e float16 perdevano troppa accuratezza: tenuta la piena precisione.")
    else:
        print(f"  perdita accettabile: {(base['accuracy']-r['accuracy'])*100:+.2f} punti, "
              f"{base['mb']/r['mb']:.1f}x piu piccolo")

    cfg.update(modello_tflite='rifiuti.tflite', quantizzazione=scelta,
               input_uint8=(scelta == 'int8'), preprocessing_incorporato=True,
               accuracy_tflite=r['accuracy'], costo_tflite=r['costo_decisione'])
    json.dump(cfg, open(cfg_path, 'w'), indent=2)
    print(f"config.json aggiornato")
    print("\nDa copiare sul Pi: rifiuti.tflite, config.json, "
          "ecosort_decisione.py, classifica_pi.py")


if __name__ == '__main__':
    main()
