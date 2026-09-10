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
"""
import os, json, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

import ecosort_benchmark as B
from ecosort_decisione import COSTO, CLASSI, pesi_training

OUT = B.OUT_DIR


def main():
    cfg_path = os.path.join(OUT, 'config.json')
    if not os.path.exists(cfg_path):
        sys.exit("Manca config.json: esegui prima ecosort_benchmark.py")
    cfg = json.load(open(cfg_path))
    nome = cfg['backbone']
    ep1 = cfg.get('epoche_fase1') or B.FASE1_EPOCHE
    ep2 = cfg.get('epoche_fase2') or B.FASE2_EPOCHE
    print(f"Backbone vincente : {nome}")
    print(f"Epoche ottimali   : fase1={ep1}  fase2={ep2}")

    train_raw, val_raw, test_raw, _ = B.carica_split()
    # train + val uniti: ~+25% di dati rispetto al benchmark
    pieno_raw = train_raw.concatenate(val_raw)
    y_pieno = B.etichette(pieno_raw)
    y_test = B.etichette(test_raw)
    print(f"Training finale su {len(y_pieno)} immagini, test su {len(y_test)}")

    bil = dict(enumerate(compute_class_weight('balanced',
                                              classes=np.unique(y_pieno), y=y_pieno)))
    pesi = pesi_training(COSTO, bil)
    prep = B.BACKBONE[nome]['prep']
    train_ds = B.prepara(pieno_raw, prep, True, pesi)
    test_ds = B.prepara(test_raw, prep, False)

    model, base = B.costruisci(nome)

    # Nessun early stopping e nessun checkpoint: non c'e' validation.
    # Usiamo esattamente le epoche che il benchmark ha indicato come ottimali.
    print(f"\n--- FASE 1 ({ep1} epoche) ---")
    B.compila(model, 1e-3)
    model.fit(train_ds, epochs=ep1, verbose=1)

    print(f"\n--- FASE 2 ({ep2} epoche) ---")
    base.trainable = True
    taglio = int(len(base.layers) * (1 - B.FRAZ_SBLOCCO))
    for l in base.layers[:taglio]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    B.compila(model, 1e-5)
    model.fit(train_ds, epochs=ep2, verbose=1)

    percorso = os.path.join(OUT, 'newbest_model.keras')
    model.save(percorso)
    print(f"\nModello definitivo salvato: {percorso}")

    print("\n--- Valutazione finale sul test set (mai usato finora) ---")
    r = B.valuta(model, test_ds, y_test, nome)
    cfg.update({k: r[k] for k in
                ('accuracy', 'macro_f1', 'costo_argmax', 'costo_soglia70',
                 'costo_decisione', 'coverage_decisione', 'gravi_decisione')})
    cfg['modello_keras'] = 'newbest_model.keras'
    json.dump(cfg, open(cfg_path, 'w'), indent=2)
    print(f"config.json aggiornato con le metriche finali")


if __name__ == '__main__':
    main()
