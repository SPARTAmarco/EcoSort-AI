"""
EcoSort AI — Benchmark multi-backbone con selezione a costo pesato.
Pensato per Colab FREE (T4): pipeline tf.data veloce, mixed precision,
stato salvabile su Drive per riprendere se la sessione cade.

USO IN COLAB
------------
  !pip install -q scikit-learn
  from google.colab import drive; drive.mount('/content/drive')
  !python ecosort_benchmark.py

COSA CAMBIA NELLA v2 (e perche')
--------------------------------
1. SELEZIONE SUL COSTO, NON SULLA LOSS. Early stopping e checkpoint seguono
   `val_costo`: il costo medio per rifiuto calcolato con la regola bayesiana
   sulla validation, dopo calibrazione. Prima si sceglieva l'epoca con la
   val_loss migliore e poi si misurava il costo: due obiettivi diversi.
2. CALIBRAZIONE (temperature scaling). Le reti sono sovra-confidenti; la
   regola a costo si regge sulle probabilita. Un parametro T stimato sulla
   validation abbassa il costo senza toccare l'accuratezza.
3. ETICHETTE DA split.json. Prima ogni run decodificava l'intero dataset tre
   volte solo per sapere le classi: minuti buttati prima di ogni training.
4. CACHE SU DISCO. Le immagini decodificate (~1,2 GB in uint8) stavano in RAM
   per tre dataset in parallelo. Su disco Colab la RAM resta libera e la cache
   si riusa fra i quattro backbone.
5. ADAMW + COSINE DECAY CON WARMUP al posto di Adam a learning rate fisso.
6. AUGMENTATION FOTOMETRICA (hue, saturazione, qualita JPEG): il dataset web e'
   fatto di foto pulite, la camera della scatola no.
7. TTA in valutazione (media con l'immagine specchiata): dice quanto si
   guadagnerebbe pagando il doppio della latenza sul Pi.
"""
import os, sys, json, time, gc, hashlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ecosort_decisione import (COSTO, CLASSI, AZIONI, azione_ottima, costo_medio,
                               costo_medio_argmax, costo_medio_soglia, applica_temperatura,
                               stima_temperatura, errore_calibrazione, gravi, riepilogo,
                               pesi_training, soglie_implicite, stampa_matrice)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
DATASET_DIR   = '/content/dataset'
SPLIT_JSON    = 'split.json'      # prodotto da prepara_dataset.py — obbligatorio
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
SEED          = 42
FASE1_EPOCHE  = 15            # head, base congelata
FASE2_EPOCHE  = 12            # fine-tuning
FRAZ_SBLOCCO  = 0.30          # frazione finale di layer da sbloccare in fase 2
LABEL_SMOOTH  = 0.05          # riduce l'overconfidence -> soglie piu affidabili
LR_FASE1      = 1e-3
LR_FASE2      = 1e-4          # con i BN congelati si puo' osare piu di 1e-5
WEIGHT_DECAY  = 1e-4
WARMUP_FRAZ   = 0.10          # 10% degli step in warmup lineare
PATIENCE_1    = 5
PATIENCE_2    = 4
AUG_FOTOMETRICA = True
USA_TTA_IN_VALUTAZIONE = True

# Cartella di lavoro: usa Drive se montato, cosi il free tier puo riprendere
OUT_DIR = '/content/drive/MyDrive/ecosort' if os.path.isdir('/content/drive/MyDrive') else '/content/ecosort'
os.makedirs(OUT_DIR, exist_ok=True)
STATO_JSON = os.path.join(OUT_DIR, 'stato_benchmark.json')

# La cache va su disco LOCALE anche se OUT_DIR e' su Drive: Drive e' lento e
# la cache viene riletta a ogni epoca.
CACHE_DIR = '/content/cache_ecosort'
os.makedirs(CACHE_DIR, exist_ok=True)

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
    # niente resize_with_pad: sul Pi classifica_pi.py fa lo stesso resize
    # deformante. Meglio una deformazione coerente che due preprocessing diversi.
    return tf.cast(img, tf.uint8), tf.one_hot(etichetta, 3)


def _da_lista(percorsi, etichette, tag, firma):
    """Costruisce un tf.data da una lista esplicita di file + etichette.
    Gli split arrivano da split.json, non da validation_split: cosi sono
    deduplicati, stratificati e identici tra un esperimento e l'altro."""
    ds = tf.data.Dataset.from_tensor_slices((list(percorsi), list(etichette)))
    ds = ds.map(_carica_immagine, num_parallel_calls=tf.data.AUTOTUNE)
    # La cache su file include la firma dello split nel nome: se cambi
    # split.json non ti ritrovi con le immagini del run precedente.
    return ds.cache(os.path.join(CACHE_DIR, f'{tag}_{firma}'))


def carica_split():
    if not os.path.exists(SPLIT_JSON):
        sys.exit(f"Manca {SPLIT_JSON}. Esegui prima:\n"
                 f"  python3 prepara_dataset.py --dataset {DATASET_DIR} "
                 f"--foto-pi /content/foto_pi --modo pi")
    sp = json.load(open(SPLIT_JSON))
    classi = sp['classi']
    assert classi == CLASSI, f"Ordine classi inatteso: {classi} (atteso {CLASSI})"
    if 'y_train' not in sp:
        sys.exit("split.json e' in formato v1 (senza etichette). Rigeneralo con "
                 "la nuova prepara_dataset.py.")
    if not sp['test']:
        sys.exit("split.json non ha test set. Rigeneralo con --foto-pi, oppure "
                 "usa --modo classico.")

    firma = hashlib.md5(json.dumps([sp['train'], sp['val'], sp['test']]).encode()).hexdigest()[:8]
    y = {k: np.array(sp[f'y_{k}'], dtype=np.int64) for k in ('train', 'val', 'test')}
    ds = {k: _da_lista(sp[k], y[k], k, firma) for k in ('train', 'val', 'test')}
    print(f"Split '{sp.get('modo')}' | test: {sp.get('origine_test', '?')}")
    return ds, y, classi


def carica_split_grezzo():
    """Il contenuto di split.json cosi com'e: serve a train_finale.py per
    ritagliarsi la fetta di calibrazione."""
    return json.load(open(SPLIT_JSON))


def dataset_da(percorsi, etichette, tag):
    """Dataset cached a partire da una lista arbitraria di file."""
    firma = hashlib.md5(json.dumps(list(percorsi)).encode()).hexdigest()[:8]
    return _da_lista(percorsi, etichette, tag, firma)


# --- augmentation -----------------------------------------------------------
AUGMENT = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08, fill_mode='nearest'),
    layers.RandomTranslation(0.15, 0.15, fill_mode='nearest'),
    layers.RandomZoom(0.15, fill_mode='nearest'),
    layers.RandomBrightness(0.2, value_range=(0, 255)),
    layers.RandomContrast(0.2),
], name='augmentation')


def _aug_fotometrica(x, y):
    """Colore e compressione, per immagine singola e su uint8.
    Il dataset web e' fatto di foto pulite e ben illuminate; dentro la scatola
    c'e' una luce fissa, un bilanciamento del bianco diverso e il JPEG della
    picamera. Questa e' la parte di domain shift che si puo' simulare gratis."""
    x = tf.image.random_hue(x, 0.03)
    x = tf.image.random_saturation(x, 0.80, 1.25)
    x = tf.image.random_jpeg_quality(x, 55, 100)
    return x, y


def prepara(ds, prep, training, pesi=None, shuffle_buf=2048):
    if training:
        ds = ds.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
        if AUG_FOTOMETRICA:
            ds = ds.map(_aug_fotometrica, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if training:
        ds = ds.map(lambda x, y: (AUGMENT(tf.cast(x, tf.float32), training=True), y),
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


def specchia(ds):
    """Stesso dataset con le immagini specchiate: serve al TTA."""
    return ds.map(lambda x, y: (tf.image.flip_left_right(x), y),
                  num_parallel_calls=tf.data.AUTOTUNE)


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


def compila(model, lr_picco, steps_per_epoca, epoche):
    """AdamW + cosine decay con warmup.
    Il warmup evita che i primi batch, con la testa inizializzata a caso,
    spostino i pesi pre-addestrati; il cosine chiude l'addestramento con
    passi piccoli, che e' quello che rende le probabilita meno rumorose."""
    totale = max(1, steps_per_epoca * epoche)
    warmup = int(totale * WARMUP_FRAZ)
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr_picco / 20.0,
        decay_steps=max(1, totale - warmup),
        alpha=0.02,
        warmup_target=lr_picco,
        warmup_steps=warmup,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=sched, weight_decay=WEIGHT_DECAY),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy'],
    )


class MetricheCosto(keras.callbacks.Callback):
    """
    A fine epoca: predice sulla validation, stima la temperatura, calcola il
    COSTO MEDIO con la regola bayesiana e lo mette dentro logs come
    'val_costo'. EarlyStopping e ModelCheckpoint possono cosi seguire la
    metrica che conta davvero invece della loss.

    Deve stare PRIMA degli altri callback nella lista: Keras li esegue in
    ordine e condivide lo stesso dizionario logs.
    """
    def __init__(self, val_ds, y_val):
        super().__init__()
        self.val_ds, self.y_val = val_ds, y_val
        self.storico = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probs = self.model.predict(self.val_ds, verbose=0).astype(np.float64)[:len(self.y_val)]
        T, _ = stima_temperatura(probs, self.y_val)
        p = applica_temperatura(probs, T)
        az, _ = azione_ottima(p)
        logs['val_costo'] = costo_medio(self.y_val, az)
        logs['val_T'] = T
        logs['val_gravi'] = float(gravi(self.y_val, az))
        logs['val_coverage'] = float((az != 3).mean())
        self.storico.append(dict(epoca=epoch + 1, costo=logs['val_costo'], T=T))
        print(f"\n    val_costo {logs['val_costo']:.4f} | T {T:.2f} | "
              f"coverage {logs['val_coverage']*100:.1f}% | gravi {int(logs['val_gravi'])}")


def callback(percorso, patience, val_ds, y_val):
    # Niente ReduceLROnPlateau: con un CosineDecay schedule il learning rate
    # non e' piu una variabile modificabile a mano, e il callback fallirebbe.
    mc = MetricheCosto(val_ds, y_val)
    return [
        mc,
        keras.callbacks.EarlyStopping('val_costo', mode='min', patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(percorso, monitor='val_costo', mode='min',
                                        save_best_only=True, verbose=0),
    ], mc


# ============================================================================
# VALUTAZIONE
# ============================================================================
def latenza_cpu_ms(model, n=20):
    """Proxy della latenza sul Pi. x86 != ARM, ma il RAPPORTO tra modelli tiene.
    Mediana e non media: su Colab una singola inferenza puo' essere interrotta
    dallo scheduler e falsare tutto. Il numero vero si misura con
    classifica_pi.py --bench sul Raspberry."""
    x = tf.random.uniform((1, *IMG_SIZE, 3), 0, 255)
    with tf.device('/CPU:0'):
        for _ in range(5):
            model(x, training=False)
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x, training=False)
            t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def probabilita(model, ds, n, tta=False):
    probs = model.predict(ds, verbose=0).astype(np.float64)[:n]
    if tta:
        probs = (probs + model.predict(specchia(ds), verbose=0).astype(np.float64)[:n]) / 2.0
    return probs


def valuta(model, val_ds, y_val, test_ds, y_test, nome):
    """
    La temperatura si stima SULLA VALIDATION e si applica al test.
    Stimarla sul test sarebbe barare: sarebbe un parametro adattato ai dati
    su cui si dichiarano i risultati.
    """
    p_val = probabilita(model, val_ds, len(y_val))
    T, _ = stima_temperatura(p_val, y_val)

    p_test_grezze = probabilita(model, test_ds, len(y_test))
    r = riepilogo(y_test, p_test_grezze, T, etichetta=nome)
    r['backbone'] = nome
    r['macro_f1'] = float(f1_score(y_test, applica_temperatura(p_test_grezze, T).argmax(1),
                                   average='macro'))
    r['params_M'] = float(model.count_params() / 1e6)
    r['gflops'] = BACKBONE[nome]['gflops']
    r['latenza_cpu_ms'] = latenza_cpu_ms(model)

    if USA_TTA_IN_VALUTAZIONE:
        p_tta = probabilita(model, test_ds, len(y_test), tta=True)
        az_tta, _ = azione_ottima(applica_temperatura(p_tta, T))
        r['costo_tta'] = costo_medio(y_test, az_tta)
        r['accuracy_tta'] = float((applica_temperatura(p_tta, T).argmax(1) == y_test).mean())

    p_cal = applica_temperatura(p_test_grezze, T)
    az, _ = azione_ottima(p_cal)
    print(f"\n--- {nome} ---")
    print(classification_report(y_test, p_cal.argmax(1), target_names=CLASSI, digits=4))
    print("Matrice di confusione (argmax):\n", confusion_matrix(y_test, p_cal.argmax(1)))
    print("Matrice con regola a costo (4a colonna = indifferenziata):\n",
          confusion_matrix(y_test, az, labels=[0, 1, 2, 3])[:3, :])
    for k, v in r.items():
        if k not in ('backbone', 'etichetta'):
            print(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    np.save(os.path.join(OUT_DIR, f'probs_{nome}.npy'), p_test_grezze)
    return r


# ============================================================================
# MAIN
# ============================================================================
def main():
    stampa_matrice()
    print("\nSOGLIE IMPLICITE dalla matrice (sostituiscono il 70% fisso):")
    for c, riv, p in soglie_implicite():
        print(f"  {c:>18} vs {riv:<18}: {p*100:5.1f}%")

    ds, y, nomi = carica_split()
    print(f"\nClassi: {nomi}")
    print(f"Train {len(y['train'])} | Val {len(y['val'])} | Test {len(y['test'])}")
    print("Distribuzione train:", dict(zip(nomi, np.bincount(y['train']).tolist())))

    bilanciato = dict(enumerate(compute_class_weight(
        'balanced', classes=np.unique(y['train']), y=y['train'])))
    pesi = pesi_training(COSTO, bilanciato)
    print(f"Pesi finali (bilanciamento x costo): {dict(zip(nomi, np.round(pesi, 3)))}")

    steps = max(1, len(y['train']) // BATCH_SIZE)
    stato = json.load(open(STATO_JSON)) if os.path.exists(STATO_JSON) else {'risultati': []}
    fatti = {r['backbone'] for r in stato['risultati']}

    for nome in DA_TESTARE:
        if nome in fatti:
            print(f"\n[skip] {nome} gia completato")
            continue
        print(f"\n{'='*70}\n  {nome}\n{'='*70}")
        t0 = time.time()
        prep = BACKBONE[nome]['prep']
        train_ds = prepara(ds['train'], prep, True, pesi)
        val_ds   = prepara(ds['val'],   prep, False)
        test_ds  = prepara(ds['test'],  prep, False)

        model, base = costruisci(nome)
        percorso = os.path.join(OUT_DIR, f'{nome}.keras')

        # --- FASE 1: solo la testa -------------------------------------- #
        compila(model, LR_FASE1, steps, FASE1_EPOCHE)
        cbs, mc1 = callback(percorso, PATIENCE_1, val_ds, y['val'])
        model.fit(train_ds, validation_data=val_ds, epochs=FASE1_EPOCHE,
                  callbacks=cbs, verbose=1)
        ep1 = int(min(mc1.storico, key=lambda s: s['costo'])['epoca'])

        # --- FASE 2: fine-tuning dell'ultima frazione di layer ----------- #
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

        compila(model, LR_FASE2, steps, FASE2_EPOCHE)
        cbs, mc2 = callback(percorso, PATIENCE_2, val_ds, y['val'])
        model.fit(train_ds, validation_data=val_ds, epochs=FASE2_EPOCHE,
                  callbacks=cbs, verbose=1)
        ep2 = int(min(mc2.storico, key=lambda s: s['costo'])['epoca'])

        r = valuta(model, val_ds, y['val'], test_ds, y['test'], nome)
        r['minuti_training'] = round((time.time() - t0) / 60, 1)
        r['epoche_fase1'], r['epoche_fase2'] = ep1, ep2
        stato['risultati'] = [x for x in stato['risultati'] if x['backbone'] != nome]
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
    ris = sorted(ris, key=lambda r: r['costo_decisione'])
    print(f"\n{'='*112}\n  CLASSIFICA FINALE — ordinata per COSTO, non per accuracy\n{'='*112}")
    print(f"{'backbone':<20}{'acc':>7}{'macroF1':>9}{'ECE':>7}{'T':>6}{'c.argmax':>10}"
          f"{'c.sogl70':>10}{'c.decis':>9}{'cover':>8}{'gravi':>7}{'ms':>7}{'min':>7}")
    for r in ris:
        print(f"{r['backbone']:<20}{r['accuracy']*100:>6.2f}%{r['macro_f1']:>9.4f}"
              f"{r['ece_dopo']:>7.3f}{r['temperatura']:>6.2f}{r['costo_argmax']:>10.4f}"
              f"{r['costo_soglia70']:>10.4f}{r['costo_decisione']:>9.4f}"
              f"{r['coverage']*100:>7.1f}%{r['gravi_decisione']:>7}"
              f"{r['latenza_cpu_ms']:>7.0f}{r.get('minuti_training',0):>7.1f}")
    v = ris[0]
    print(f"\nVINCITORE: {v['backbone']}")
    print(f"  costo {v['costo_decisione']:.4f} per rifiuto  "
          f"(argmax {v['costo_argmax']:.4f}, soglia 0.70 {v['costo_soglia70']:.4f}, "
          f"migliore soglia fissa {v['costo_soglia_ottima']:.4f} @ {v['soglia_ottima']:.2f})")
    print(f"  calibrazione: T={v['temperatura']:.3f}, ECE {v['ece_prima']:.4f} -> {v['ece_dopo']:.4f}")
    print(f"  errori gravi (vetro nel bidone sbagliato): {v['gravi_argmax']} -> {v['gravi_decisione']}")
    if 'costo_tta' in v:
        print(f"  con TTA (2x latenza): costo {v['costo_tta']:.4f}, acc {v['accuracy_tta']*100:.2f}%")

    usa_costo = v['costo_decisione'] <= v['costo_soglia70']
    cfg = dict(labels=CLASSI, azioni=AZIONI, backbone=v['backbone'],
               epoche_fase1=v.get('epoche_fase1'), epoche_fase2=v.get('epoche_fase2'),
               matrice_costo=COSTO.tolist(), soglia_fallback=0.70,
               temperatura=v['temperatura'], usa_regola_costo=bool(usa_costo),
               costo_atteso_benchmark=v['costo_decisione'])
    json.dump(cfg, open(os.path.join(OUT_DIR, 'config.json'), 'w'), indent=2)
    print(f"\nconfig.json scritto in {OUT_DIR}")


if __name__ == '__main__':
    main()
