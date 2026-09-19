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

COSA CAMBIA NELLA v2
--------------------
1. DATASET RAPPRESENTATIVO STRATIFICATO. Prima si prendevano le prime N
   immagini del training set, che e' ordinato per classe: la calibrazione
   INT8 vedeva solo carta. Ora e' un campione mescolato e bilanciato.
2. LA TEMPERATURA E' PARTE DELLA VALUTAZIONE. Ogni variante viene giudicata
   con le probabilita calibrate, cioe' esattamente come funzionera sul Pi.
3. LA SCELTA GUARDA GLI ERRORI GRAVI, non solo l'accuratezza: una variante
   che mette anche un solo pezzo di vetro in piu nel macero viene scartata.
"""
import os, json, sys, time
import numpy as np
import tensorflow as tf
from tensorflow import keras

import ecosort_benchmark as B
from ecosort_decisione import (COSTO, CLASSI, azione_ottima, costo_medio,
                               costo_medio_argmax, applica_temperatura, gravi)

# Importare il benchmark attiva mixed_float16 (serve al training su T4).
# Per la conversione serve float32 puro: in float16 i range di quantizzazione
# vengono calcolati su tensori gia degradati e l'INT8 esce peggiore.
keras.mixed_precision.set_global_policy('float32')

OUT = B.OUT_DIR
N_REPRESENTATIVE = 300     # immagini per calibrare i range di quantizzazione
MAX_PERDITA_ACC  = 1.0     # punti percentuali
MAX_PEGGIO_COSTO = 0.02    # costo medio per rifiuto


def carica_in_float32(nome_backbone, keras_path):
    """
    Carica i pesi dentro un'architettura FLOAT32 ricostruita da zero.

    Perche' non basta keras.models.load_model(): il modello e' stato
    addestrato con mixed_float16 e ogni layer si porta dietro quella policy
    dentro il file .keras. Cambiare la policy globale non li tocca, e il
    converter TFLite non sa convertire op in float16: fallisce con
    ERROR_NEEDS_FLEX_OPS su Conv2D, Sigmoid, Mul... (l'intero grafo).

    Ricostruendo l'architettura sotto policy float32 e caricando solo i PESI
    si ottiene lo stesso modello, numericamente identico a meno del
    troncamento float16 gia avvenuto, ma in un grafo che TFLite converte.
    """
    keras.mixed_precision.set_global_policy('float32')
    model, _ = B.costruisci(nome_backbone)
    model.load_weights(keras_path)
    return model


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


def valuta_tflite(percorso, x_test, y_test, T, n_bench=20):
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
    t = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        interp.set_tensor(inp_d['index'], x); interp.invoke()
        t.append((time.perf_counter() - t0) * 1000)

    # esattamente la catena che girera sul Pi: probabilita -> temperatura -> costo
    p_cal = applica_temperatura(probs, T)
    azioni, _ = azione_ottima(p_cal)
    return dict(
        accuracy=float((p_cal.argmax(1) == y_test).mean()),
        costo_argmax=costo_medio_argmax(y_test, p_cal),
        costo_decisione=costo_medio(y_test, azioni),
        coverage=float((azioni != 3).mean()),
        gravi=gravi(y_test, azioni),
        mb=os.path.getsize(percorso) / 1e6,
        ms=float(np.median(t)),
    )


def campione_rappresentativo(percorsi, etichette, n):
    """Campione bilanciato e mescolato.
    Le liste in split.json sono ordinate per classe: prendere le prime n
    significava calibrare la quantizzazione su una sola classe."""
    percorsi, etichette = np.array(percorsi), np.array(etichette)
    rng = np.random.default_rng(0)
    per_classe = max(1, n // len(np.unique(etichette)))
    scelti = []
    for c in np.unique(etichette):
        idx = np.where(etichette == c)[0]
        scelti += list(rng.choice(idx, size=min(per_classe, len(idx)), replace=False))
    rng.shuffle(scelti)
    return percorsi[np.array(scelti)], etichette[np.array(scelti)]


def main():
    cfg_path = os.path.join(OUT, 'config.json')
    cfg = json.load(open(cfg_path))
    nome = cfg['backbone']
    T = float(cfg.get('temperatura', 1.0))
    keras_path = os.path.join(OUT, cfg.get('modello_keras', 'newbest_model.keras'))
    if not os.path.exists(keras_path):
        sys.exit(f"Manca {keras_path}: esegui prima train_finale.py")

    print(f"Modello: {keras_path}  (backbone {nome}, temperatura {T:.3f})")
    print("Ricostruzione dell'architettura in float32 e caricamento dei pesi...")
    model = carica_in_float32(nome, keras_path)
    wrapper = wrappa_con_preprocessing(model, nome)

    # dati: test set in uint8 [0,255], esattamente come arrivera dalla camera
    sp = B.carica_split_grezzo()
    test_raw = B.dataset_da(sp['test'], sp['y_test'], 'test_conv')
    x_test = np.stack([x.numpy() for x, _ in test_raw])
    y_test = np.array(sp['y_test'])
    print(f"Test set: {len(x_test)} immagini — {sp.get('origine_test','?')}")

    rp, ry = campione_rappresentativo(sp['train'], sp['y_train'], N_REPRESENTATIVE)
    rep_raw = B.dataset_da(rp, ry, 'rep')
    x_rep = np.stack([x.numpy() for x, _ in rep_raw])
    print(f"Campione per la quantizzazione: {len(x_rep)} immagini "
          f"{dict(zip(CLASSI, np.bincount(ry, minlength=3).tolist()))}")

    def rep_data():
        # deve coprire la varieta reale del dataset: se calibri su immagini
        # tutte della stessa classe, i range di quantizzazione sono sbagliati
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
        r = valuta_tflite(percorso, x_test, y_test, T)
        risultati[modo] = r
        print(f"  {r['mb']:.1f} MB | acc {r['accuracy']*100:.2f}% | "
              f"costo {r['costo_decisione']:.4f} | gravi {r['gravi']} | {r['ms']:.0f} ms (x86)")

    if not risultati:
        sys.exit("Nessuna conversione riuscita")

    base = risultati.get('float32')
    print(f"\n{'='*84}\n  CONFRONTO — quanto costa la quantizzazione\n{'='*84}")
    print(f"{'variante':<12}{'MB':>8}{'accuracy':>11}{'delta':>9}{'costo':>9}"
          f"{'cover':>8}{'gravi':>8}{'ms x86':>9}")
    for m, r in risultati.items():
        d = (r['accuracy'] - base['accuracy']) * 100 if base else 0.0
        print(f"{m:<12}{r['mb']:>8.1f}{r['accuracy']*100:>10.2f}%{d:>+9.2f}"
              f"{r['costo_decisione']:>9.4f}{r['coverage']*100:>7.1f}%{r['gravi']:>8}{r['ms']:>9.0f}")

    # Scelta: INT8 solo se il risparmio non costa accuratezza. Su un progetto
    # dove il vetro nel bidone sbagliato pesa 9.0, mezzo punto di accuratezza
    # vale piu di 3 MB di file — e un errore grave in piu vale piu di tutto.
    scelta = 'float32'
    for m in ['int8', 'float16']:
        if m in risultati and base:
            perdita = (base['accuracy'] - risultati[m]['accuracy']) * 100
            peggior_costo = risultati[m]['costo_decisione'] - base['costo_decisione']
            piu_gravi = risultati[m]['gravi'] > base['gravi']
            if perdita <= MAX_PERDITA_ACC and peggior_costo <= MAX_PEGGIO_COSTO and not piu_gravi:
                scelta = m
                break
    finale = os.path.join(OUT, 'rifiuti.tflite')
    with open(finale, 'wb') as f:
        f.write(open(os.path.join(OUT, f'rifiuti_{scelta}.tflite'), 'rb').read())

    r = risultati[scelta]
    print(f"\nSCELTA: {scelta}  ->  rifiuti.tflite  ({r['mb']:.1f} MB)")
    if scelta == 'float32':
        print("  INT8 e float16 perdevano troppo: tenuta la piena precisione.")
    else:
        print(f"  perdita accettabile: {(base['accuracy']-r['accuracy'])*100:+.2f} punti, "
              f"{base['mb']/r['mb']:.1f}x piu piccolo, stessi errori gravi")

    cfg.update(modello_tflite='rifiuti.tflite', quantizzazione=scelta,
               input_uint8=(scelta == 'int8'), preprocessing_incorporato=True,
               accuracy_tflite=r['accuracy'], costo_tflite=r['costo_decisione'],
               gravi_tflite=r['gravi'])
    json.dump(cfg, open(cfg_path, 'w'), indent=2)
    print("config.json aggiornato")
    print("\nDa copiare sul Pi: rifiuti.tflite, config.json, "
          "ecosort_decisione.py, classifica_pi.py")


if __name__ == '__main__':
    main()
