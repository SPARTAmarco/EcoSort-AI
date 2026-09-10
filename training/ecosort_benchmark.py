"""
EcoSort AI — Benchmark multi-backbone con selezione a costo pesato.
Pensato per Colab FREE (T4): pipeline tf.data veloce, mixed precision,
e stato salvabile su Drive per riprendere se la sessione cade.

USO IN COLAB
------------
  !pip install -q scikit-learn
  # (opzionale ma consigliato su free tier)
  from google.colab import drive; drive.mount('/content/drive')
  %run ecosort_benchmark.py
"""
import os, sys, json, time, gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ecosort_decisione import (COSTO, CLASSI, AZIONI, azione_ottima, costo_medio,
                               costo_medio_argmax, costo_medio_soglia,
                               pesi_training, soglie_implicite, stampa_matrice)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
DATASET_DIR   = '/content/dataset'
SPLIT_JSON    = 'split.json'      # prodotto da prepara_dataset.py — obbligatorio
TEST_PI_DIR   = '/content/foto_pi'  # foto scattate dalla camera nella scatola
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
SEED          = 42
VAL_FRAC      = 0.30          # 30% -> poi diviso a meta in val (15%) e test (15%)
FASE1_EPOCHE  = 15            # head, base congelata
FASE2_EPOCHE  = 10            # fine-tuning
FRAZ_SBLOCCO  = 0.30          # frazione finale di layer da sbloccare in fase 2
LABEL_SMOOTH  = 0.05          # riduce l'overconfidence -> soglie piu affidabili

# Cartella di lavoro: usa Drive se montato, cosi il free tier puo riprendere
OUT_DIR = '/content/drive/MyDrive/ecosort' if os.path.isdir('/content/drive/MyDrive') else '/content/ecosort'
os.makedirs(OUT_DIR, exist_ok=True)
STATO_JSON = os.path.join(OUT_DIR, 'stato_benchmark.json')

tf.keras.utils.set_random_seed(SEED)

# T4 ha i tensor core: mixed precision quasi raddoppia la velocita
if tf.config.list_physical_devices('GPU'):
    keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision float16 attiva")
print("GPU:", tf.config.list_physical_devices('GPU') or "NESSUNA (sara lentissimo)")

# ============================================================================
# BACKBONE CANDIDATI — tutti compatibili con la finestra 2-3 s su Pi 4B
# ResNet50V2 e' incluso solo come riferimento di qualita: e' quasi certamente
# fuori budget sul Pi (~7 GFLOPs), ma serve a sapere quanto si perde.
# ============================================================================
def _identita(x):
    return x   # EfficientNet / MobileNetV3 normalizzano internamente: vogliono [0,255]

BACKBONE = {
    'MobileNetV3Large': dict(
        fn=lambda: keras.applications.MobileNetV3Large(
            weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3),
            include_preprocessing=True),
        prep=_identita, gflops=0.22),
    'EfficientNetB0': dict(
        fn=lambda: keras.applications.EfficientNetB0(
            weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3)),
        prep=_identita, gflops=0.39),
    'EfficientNetV2B0': dict(
        fn=lambda: keras.applications.EfficientNetV2B0(
            weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3),
            include_preprocessing=True),
        prep=_identita, gflops=0.72),
    'ResNet50V2': dict(
        fn=lambda: keras.applications.ResNet50V2(
            weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3)),
        prep=keras.applications.resnet_v2.preprocess_input, gflops=6.97),
}
DA_TESTARE = ['MobileNetV3Large', 'EfficientNetB0', 'EfficientNetV2B0', 'ResNet50V2']

# ============================================================================
# DATI — tf.data invece di ImageDataGenerator.
# ImageDataGenerator e' single-thread e su T4 tiene la GPU al 30%: e' il vero
# collo di bottiglia del training originale, non il modello.
# ============================================================================
def _carica_immagine(percorso, etichetta):
    img = tf.io.decode_image(tf.io.read_file(percorso), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method='bilinear')
    return tf.cast(img, tf.uint8), tf.one_hot(etichetta, 3)


def _da_lista(percorsi, classi):
    """Costruisce un tf.data da una lista esplicita di file.
    Gli split arrivano da split.json, non da validation_split: cosi sono
    deduplicati, stratificati e identici tra un esperimento e l'altro."""
    idx = {c: i for i, c in enumerate(classi)}
    etichette = [idx[os.path.basename(os.path.dirname(p))] for p in percorsi]
    ds = tf.data.Dataset.from_tensor_slices((list(percorsi), etichette))
    return ds.map(_carica_immagine, num_parallel_calls=tf.data.AUTOTUNE).cache()


def carica_split():
    if not os.path.exists(SPLIT_JSON):
        sys.exit(f"Manca {SPLIT_JSON}. Esegui prima:\n"
                 f"  python3 prepara_dataset.py --dataset {DATASET_DIR} --modo pi")
    sp = json.load(open(SPLIT_JSON))
    classi = sp['classi']
    assert classi == CLASSI, f"Ordine classi inatteso: {classi} (atteso {CLASSI})"

    train_raw = _da_lista(sp['train'], classi)
    val_raw   = _da_lista(sp['val'], classi)

    if sp['test']:
        test_raw = _da_lista(sp['test'], classi)
        origine = f"split interno ({len(sp['test'])} img)"
    elif os.path.isdir(TEST_PI_DIR):
        # IL test set che conta: foto reali scattate nella scatola.
        # Misura il domain shift (luce fissa, sfondo, angolo) che il dataset
        # web non contiene e che e' il vero motivo per cui i modelli crollano.
        percorsi = [os.path.join(TEST_PI_DIR, c, f)
                    for c in classi for f in sorted(os.listdir(os.path.join(TEST_PI_DIR, c)))]
        test_raw = _da_lista(percorsi, classi)
        origine = f"FOTO REALI dal Pi ({len(percorsi)} img)"
    else:
        print(f"\nATTENZIONE: nessun test set. Manca {TEST_PI_DIR} e split['test'] e vuoto.\n"
              f"Uso la validation come test: i numeri saranno OTTIMISTICI.\n")
        test_raw, origine = val_raw, "validation (stima ottimistica!)"

    print(f"Test set: {origine}")
    return train_raw, val_raw, test_raw, classi


AUGMENT = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08, fill_mode='nearest'),
    layers.RandomTranslation(0.15, 0.15, fill_mode='nearest'),
    layers.RandomZoom(0.15, fill_mode='nearest'),
    layers.RandomBrightness(0.2, value_range=(0, 255)),
    layers.RandomContrast(0.2),
], name='augmentation')

def prepara(ds, prep, training, pesi=None, shuffle_buf=2000):
    if training:
        ds = ds.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    if training:
        ds = ds.map(lambda x, y: (AUGMENT(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    # clip: RandomContrast puo spingere i valori fuori da [0,255] e le
    # normalizzazioni interne di EfficientNet/MobileNet assumono quel range
    ds = ds.map(lambda x, y: (prep(tf.clip_by_value(tf.cast(x, tf.float32), 0., 255.)), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if pesi is not None:
        # sample_weight = costo medio di sbagliare la classe reale.
        # Solo sul TRAIN: la validation deve restare non pesata, altrimenti
        # le metriche di early stopping non sono confrontabili tra modelli.
        w = tf.constant(pesi, dtype=tf.float32)
        ds = ds.map(lambda x, y: (x, y, tf.reduce_sum(y * w, axis=-1)),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def etichette(ds):
    return np.concatenate([y.numpy() for _, y in ds.batch(512)]).argmax(1)

# ============================================================================
# MODELLO
# ============================================================================
def costruisci(nome):
    base = BACKBONE[nome]['fn']()
    base.trainable = False
    x = keras.Input(shape=(*IMG_SIZE, 3))
    h = base(x, training=False)
    h = layers.GlobalAveragePooling2D()(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dense(256, activation='relu')(h)
    h = layers.Dropout(0.4)(h)
    # softmax SEMPRE in float32: in float16 satura e rovina la calibrazione,
    # e la calibrazione e' cio' su cui si regge la regola di decisione.
    out = layers.Dense(3, activation='softmax', dtype='float32')(h)
    return keras.Model(x, out), base

def compila(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy'],
    )

def callback(percorso, patience):
    # Tutti e tre sullo STESSO monitor: nel codice originale ModelCheckpoint
    # seguiva val_accuracy mentre EarlyStopping ripristinava su val_loss,
    # quindi il file salvato e il modello in memoria erano due modelli diversi.
    return [
        keras.callbacks.EarlyStopping('val_loss', patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5,
                                          patience=max(2, patience - 2),
                                          min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(percorso, monitor='val_loss',
                                        save_best_only=True, verbose=0),
    ]

# ============================================================================
# VALUTAZIONE
# ============================================================================
def latenza_cpu_ms(model, n=15):
    """Proxy della latenza sul Pi. x86 != ARM, ma il RAPPORTO tra modelli tiene.
    Per il numero vero usa benchmark_pi.py sul Raspberry."""
    x = tf.random.uniform((1, *IMG_SIZE, 3), 0, 255)
    with tf.device('/CPU:0'):
        for _ in range(3):
            model(x, training=False)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x, training=False)
        return (time.perf_counter() - t0) / n * 1000

def valuta(model, test_ds, y_test, nome):
    probs = model.predict(test_ds, verbose=0).astype(np.float64)
    probs = probs[:len(y_test)]
    y_pred = probs.argmax(1)
    azioni, _ = azione_ottima(probs)

    r = dict(
        backbone=nome,
        accuracy=float((y_pred == y_test).mean()),
        macro_f1=float(f1_score(y_test, y_pred, average='macro')),
        costo_argmax=costo_medio_argmax(y_test, probs),
        costo_soglia70=costo_medio_soglia(y_test, probs, 0.70),
        costo_decisione=costo_medio(y_test, azioni),
        coverage_decisione=float((azioni != 3).mean()),
        coverage_soglia70=float((probs.max(1) >= 0.70).mean()),
        # errori gravi: vetro_e_metallo mandato nel blu o nel giallo
        gravi_argmax=int(((y_test == 2) & np.isin(y_pred, [0, 1])).sum()),
        gravi_decisione=int(((y_test == 2) & np.isin(azioni, [0, 1])).sum()),
        params_M=float(model.count_params() / 1e6),
        gflops=BACKBONE[nome]['gflops'],
        latenza_cpu_ms=float(latenza_cpu_ms(model)),
    )
    print(f"\n--- {nome} ---")
    print(classification_report(y_test, y_pred, target_names=CLASSI, digits=4))
    print("Matrice di confusione (argmax):\n", confusion_matrix(y_test, y_pred))
    print("Matrice reale con regola a costo (colonna 4 = indifferenziata):\n",
          confusion_matrix(y_test, azioni, labels=[0, 1, 2, 3])[:3, :])
    for k, v in r.items():
        if k != 'backbone':
            print(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    np.save(os.path.join(OUT_DIR, f'probs_{nome}.npy'), probs)
    return r

# ============================================================================
# MAIN
# ============================================================================
def main():
    stampa_matrice()
    print("\nSOGLIE IMPLICITE dalla matrice (sostituiscono il 70% fisso):")
    for c, riv, p in soglie_implicite():
        print(f"  {c:>18} vs {riv:<18}: {p*100:5.1f}%")

    train_raw, val_raw, test_raw, nomi = carica_split()
    y_train = etichette(train_raw)
    y_test  = etichette(test_raw)
    print(f"\nClassi: {nomi}")
    print(f"Train {len(y_train)} | Val {len(etichette(val_raw))} | Test {len(y_test)}")
    print("Distribuzione train:", dict(zip(nomi, np.bincount(y_train))))

    bilanciato = dict(enumerate(compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)))
    pesi = pesi_training(COSTO, bilanciato)
    print(f"Pesi finali (bilanciamento x costo): {dict(zip(nomi, np.round(pesi, 3)))}")

    stato = json.load(open(STATO_JSON)) if os.path.exists(STATO_JSON) else {'risultati': []}
    fatti = {r['backbone'] for r in stato['risultati']}

    for nome in DA_TESTARE:
        if nome in fatti:
            print(f"\n[skip] {nome} gia completato")
            continue
        print(f"\n{'='*70}\n  {nome}\n{'='*70}")
        t0 = time.time()
        prep = BACKBONE[nome]['prep']
        train_ds = prepara(train_raw, prep, True, pesi)
        val_ds   = prepara(val_raw,   prep, False)
        test_ds  = prepara(test_raw,  prep, False)

        model, base = costruisci(nome)
        percorso = os.path.join(OUT_DIR, f'{nome}.keras')

        compila(model, 1e-3)
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=FASE1_EPOCHE,
                       callbacks=callback(percorso, 4), verbose=1)
        # epoca migliore (1-based): serve per il retraining finale su train+val,
        # dove non avremo una validation su cui fare early stopping
        ep1 = int(np.argmin(h1.history['val_loss'])) + 1

        # FASE 2 — fine-tuning dell'ultima frazione di layer.
        base.trainable = True
        taglio = int(len(base.layers) * (1 - FRAZ_SBLOCCO))
        for l in base.layers[:taglio]:
            l.trainable = False
        # I BatchNormalization restano congelati: con 8k immagini ricalcolare
        # le statistiche running distrugge i pesi pre-addestrati.
        for l in base.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False
        print(f"Layer allenabili: {sum(1 for l in base.layers if l.trainable)}/{len(base.layers)}")

        compila(model, 1e-5)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=FASE2_EPOCHE,
                       callbacks=callback(percorso, 3), verbose=1)
        ep2 = int(np.argmin(h2.history['val_loss'])) + 1

        r = valuta(model, test_ds, y_test, nome)
        r['minuti_training'] = round((time.time() - t0) / 60, 1)
        r['epoche_fase1'], r['epoche_fase2'] = ep1, ep2
        stato['risultati'].append(r)
        json.dump(stato, open(STATO_JSON, 'w'), indent=2)
        print(f"[salvato] stato in {STATO_JSON}")

        del model, base
        keras.backend.clear_session()
        gc.collect()

    classifica(stato['risultati'])

def classifica(ris):
    if not ris:
        return
    ris = sorted(ris, key=lambda r: min(r['costo_decisione'], r['costo_soglia70']))
    print(f"\n{'='*100}\n  CLASSIFICA FINALE — ordinata per COSTO, non per accuracy\n{'='*100}")
    print(f"{'backbone':<20}{'acc':>7}{'macroF1':>9}{'c.argmax':>10}{'c.sogl70':>10}"
          f"{'c.decis':>9}{'cover':>8}{'gravi':>7}{'ms':>8}{'min':>7}")
    for r in ris:
        print(f"{r['backbone']:<20}{r['accuracy']*100:>6.2f}%{r['macro_f1']:>9.4f}"
              f"{r['costo_argmax']:>10.4f}{r['costo_soglia70']:>10.4f}"
              f"{r['costo_decisione']:>9.4f}{r['coverage_decisione']*100:>7.1f}%"
              f"{r['gravi_decisione']:>7}{r['latenza_cpu_ms']:>8.0f}{r.get('minuti_training',0):>7.1f}")
    v = ris[0]
    print(f"\nVINCITORE: {v['backbone']}")
    print(f"  costo minimo {min(v['costo_decisione'], v['costo_soglia70']):.4f} per rifiuto")
    print(f"  regola migliore: {'decisione bayesiana' if v['costo_decisione'] <= v['costo_soglia70'] else 'soglia 0.70'}")
    print(f"  errori gravi (vetro nel bidone sbagliato): {v['gravi_argmax']} -> {v['gravi_decisione']}")

    cfg = dict(labels=CLASSI, azioni=AZIONI, backbone=v['backbone'],
               epoche_fase1=v.get('epoche_fase1'), epoche_fase2=v.get('epoche_fase2'),
               matrice_costo=COSTO.tolist(), soglia_fallback=0.70,
               usa_regola_costo=bool(v['costo_decisione'] <= v['costo_soglia70']))
    json.dump(cfg, open(os.path.join(OUT_DIR, 'config.json'), 'w'), indent=2)
    print(f"\nconfig.json scritto in {OUT_DIR}")

if __name__ == '__main__':
    main()
