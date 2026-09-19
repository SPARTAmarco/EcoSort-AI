"""
EcoSort AI — Training FINALE del modello definitivo.

Da eseguire DOPO ecosort_benchmark.py. Prende il backbone vincente e lo
riaddestra su TRAIN + VALIDATION uniti, per le epoche che si erano rivelate
ottimali durante il benchmark.

Perche': la validation ha gia svolto il suo lavoro (scegliere backbone ed
epoche). Tenerla fuori dal training finale significa buttare via il 20% dei
dati per niente. Il test set invece resta intoccato: e' l'unica cosa che
ancora non ha influenzato nessuna decisione.

    python3 train_finale.py

COSA CAMBIA NELLA v2
--------------------
1. FETTA DI CALIBRAZIONE. Il 15% della vecchia validation resta fuori dal
   training: serve a ristimare la temperatura sul modello definitivo. Sono
   ~250 immagini, cioe' il 3% del dataset: un prezzo minimo per non dover
   riusare una T stimata su un altro modello (o, peggio, stimarla sul test).
2. EPOCHE RISCALATE. Con il 25% di dati in piu ogni epoca contiene piu step;
   le epoche ottimali del benchmark vengono scalate di conseguenza.
3. Stesso schedule AdamW + cosine del benchmark, cosi il modello finale vede
   esattamente la stessa ricetta che ha vinto il confronto.
4. IL MODELLO SI SALVA IN FLOAT32. Il training usa mixed_float16 per la T4, ma
   un .keras con i layer in float16 non e' convertibile in TFLite. Si salvano
   i pesi e si ricostruisce l'architettura in float32 prima di salvare.
"""
import os, json, sys, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

import ecosort_benchmark as B
from ecosort_decisione import (COSTO, CLASSI, pesi_training, stima_temperatura,
                               applica_temperatura, errore_calibrazione)

OUT = B.OUT_DIR
FRAZ_CALIBRAZIONE = 0.15      # quota della vecchia validation tenuta fuori
FATTORE_EPOCHE    = 1.15      # +25% di dati -> qualche epoca in piu


def taglia_calibrazione(percorsi, etichette, frazione, seed=42):
    """Divide la validation in (da_allenare, per_calibrare), stratificata."""
    percorsi, etichette = np.array(percorsi), np.array(etichette)
    rng = np.random.default_rng(seed)
    cal, tr = [], []
    for c in np.unique(etichette):
        idx = np.where(etichette == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * frazione)))
        cal += list(idx[:n]); tr += list(idx[n:])
    cal, tr = np.array(sorted(cal)), np.array(sorted(tr))
    return (percorsi[tr], etichette[tr]), (percorsi[cal], etichette[cal])


def main():
    cfg_path = os.path.join(OUT, 'config.json')
    if not os.path.exists(cfg_path):
        sys.exit("Manca config.json: esegui prima ecosort_benchmark.py")
    cfg = json.load(open(cfg_path))
    nome = cfg['backbone']
    ep1 = int(math.ceil((cfg.get('epoche_fase1') or B.FASE1_EPOCHE) * FATTORE_EPOCHE))
    ep2 = int(math.ceil((cfg.get('epoche_fase2') or B.FASE2_EPOCHE) * FATTORE_EPOCHE))
    print(f"Backbone vincente : {nome}")
    print(f"Epoche del benchmark: {cfg.get('epoche_fase1')}/{cfg.get('epoche_fase2')} "
          f"-> riscalate a {ep1}/{ep2}")

    sp = B.carica_split_grezzo()
    (val_tr_p, val_tr_y), (cal_p, cal_y) = taglia_calibrazione(
        sp['val'], sp['y_val'], FRAZ_CALIBRAZIONE)

    pieno_p = list(sp['train']) + list(val_tr_p)
    pieno_y = list(sp['y_train']) + list(val_tr_y)
    print(f"Training finale su {len(pieno_p)} immagini "
          f"({len(sp['train'])} train + {len(val_tr_p)} val)")
    print(f"Calibrazione su {len(cal_p)} immagini tenute fuori")
    print(f"Test su {len(sp['test'])} immagini mai viste: {sp.get('origine_test','?')}")

    pieno_raw = B.dataset_da(pieno_p, pieno_y, 'pieno')
    cal_raw   = B.dataset_da(cal_p, cal_y, 'calib')
    test_raw  = B.dataset_da(sp['test'], sp['y_test'], 'test_finale')
    y_pieno = np.array(pieno_y)
    y_cal   = np.array(cal_y)
    y_test  = np.array(sp['y_test'])

    bil = dict(enumerate(compute_class_weight('balanced',
                                              classes=np.unique(y_pieno), y=y_pieno)))
    pesi = pesi_training(COSTO, bil)
    print(f"Pesi (bilanciamento x costo): {dict(zip(CLASSI, np.round(pesi, 3)))}")

    prep = B.BACKBONE[nome]['prep']
    train_ds = B.prepara(pieno_raw, prep, True, pesi)
    cal_ds   = B.prepara(cal_raw,  prep, False)
    test_ds  = B.prepara(test_raw, prep, False)
    steps = max(1, len(y_pieno) // B.BATCH_SIZE)

    model, base = B.costruisci(nome)

    # Nessun early stopping e nessun checkpoint: non c'e' una validation su cui
    # fermarsi. Si usano esattamente le epoche indicate dal benchmark.
    print(f"\n--- FASE 1 ({ep1} epoche) ---")
    B.compila(model, B.LR_FASE1, steps, ep1)
    model.fit(train_ds, epochs=ep1, verbose=1)

    print(f"\n--- FASE 2 ({ep2} epoche) ---")
    base.trainable = True
    taglio = int(len(base.layers) * (1 - B.FRAZ_SBLOCCO))
    for l in base.layers[:taglio]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    B.compila(model, B.LR_FASE2, steps, ep2)
    model.fit(train_ds, epochs=ep2, verbose=1)

    # --- salvataggio in FLOAT32 ------------------------------------------ #
    # Il training gira in mixed_float16 e ogni layer si porta dietro quella
    # policy dentro il .keras. Il converter TFLite non sa tradurre op in
    # float16 (ERROR_NEEDS_FLEX_OPS) e anche il server Flask del desktop e'
    # piu veloce in float32 su CPU. Si salvano quindi i pesi, si ricostruisce
    # la stessa architettura sotto policy float32 e si ricarica: il modello e'
    # numericamente identico, ma nasce gia convertibile.
    percorso = os.path.join(OUT, 'newbest_model.keras')
    pesi_tmp = os.path.join(OUT, '_pesi_finali.weights.h5')
    model.save_weights(pesi_tmp)
    keras.mixed_precision.set_global_policy('float32')
    model, _ = B.costruisci(nome)
    model.load_weights(pesi_tmp)
    model.save(percorso)
    os.remove(pesi_tmp)
    print(f"\nModello definitivo salvato in float32: {percorso}")

    # --- calibrazione sul set tenuto fuori ------------------------------- #
    p_cal = model.predict(cal_ds, verbose=0).astype(np.float64)[:len(y_cal)]
    T, _ = stima_temperatura(p_cal, y_cal)
    print(f"\nTemperatura stimata sul set di calibrazione: T = {T:.3f}")
    print(f"  ECE {errore_calibrazione(p_cal, y_cal):.4f} -> "
          f"{errore_calibrazione(applica_temperatura(p_cal, T), y_cal):.4f}")

    # --- valutazione finale ---------------------------------------------- #
    print("\n--- Valutazione sul test set (mai usato in training) ---")
    from ecosort_decisione import riepilogo, azione_ottima, costo_medio
    from sklearn.metrics import classification_report, confusion_matrix

    p_test = B.probabilita(model, test_ds, len(y_test))
    r = riepilogo(y_test, p_test, T, etichetta=nome)
    p_fin = applica_temperatura(p_test, T)
    az, _ = azione_ottima(p_fin)
    print(classification_report(y_test, p_fin.argmax(1), target_names=CLASSI, digits=4))
    print("Confusione con regola a costo (4a colonna = indifferenziata):\n",
          confusion_matrix(y_test, az, labels=[0, 1, 2, 3])[:3, :])
    for k, v in r.items():
        if k != 'etichetta':
            print(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    np.save(os.path.join(OUT, 'probs_finale.npy'), p_test)

    cfg.update({k: r[k] for k in
                ('accuracy', 'costo_argmax', 'costo_soglia70', 'costo_decisione',
                 'coverage', 'gravi_argmax', 'gravi_decisione', 'ece_prima', 'ece_dopo')})
    cfg['temperatura'] = float(T)
    cfg['modello_keras'] = 'newbest_model.keras'
    cfg['n_test'] = int(len(y_test))
    json.dump(cfg, open(cfg_path, 'w'), indent=2)
    print("\nconfig.json aggiornato con le metriche finali e la temperatura")


if __name__ == '__main__':
    main()
