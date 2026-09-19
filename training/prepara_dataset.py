"""
EcoSort AI — Preparazione del dataset: inventario, deduplicazione, split stratificato.

DA ESEGUIRE UNA VOLTA SOLA, prima di qualsiasi training.
Produce split.json con le liste di file E le etichette gia risolte, cosi ogni
training successivo usa ESATTAMENTE gli stessi split (riproducibilita) e non
c'e' modo che un'immagine passi da validation a training tra un esperimento e
l'altro.

    python3 prepara_dataset.py --dataset /content/dataset --modo pi \
                               --foto-pi /content/foto_pi

NOVITA v2
  - le etichette finiscono dentro split.json: il benchmark non deve piu
    decodificare l'intero dataset solo per sapere quali classi ha (erano tre
    passate complete su 8.000 immagini prima ancora di iniziare a addestrare)
  - --foto-pi: le foto scattate dalla camera diventano il test set ufficiale,
    con controllo di leakage incrociato contro il dataset web
  - scansione hash in parallelo (I/O bound: sui 2 core di Colab va 2-3x)
  - verifica anti-leakage a COPPIE (la vecchia assert controllava
    l'intersezione dei tre insiemi insieme, che e' quasi sempre vuota anche
    quando due split si sovrappongono)

Modi:
  --modo pi        80/20 sul dataset web, il TEST vero sono le foto del Pi
                   (consigliato: misura cio' che conta davvero)
  --modo classico  70/15/15 tutto interno al dataset web
  --modo marco     80/10/10
"""
import os, sys, json, hashlib, argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

ESTENSIONI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SOGLIA_HAMMING = 5      # <=5 bit di differenza su 64 = quasi-duplicato
POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

MODI = {'pi': (0.80, 0.20, 0.00), 'classico': (0.70, 0.15, 0.15), 'marco': (0.80, 0.10, 0.10)}


def elenca(dataset_dir, classi_attese=None):
    if not os.path.isdir(dataset_dir):
        sys.exit(f"Cartella inesistente: {dataset_dir}")
    classi = sorted(d for d in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, d)))
    if classi_attese is not None and classi != classi_attese:
        sys.exit(f"Classi in {dataset_dir}: {classi}\nAttese: {classi_attese}\n"
                 f"Le sottocartelle devono avere gli stessi nomi, o le etichette "
                 f"del test set non corrisponderanno a quelle del training.")
    percorsi, etichette = [], []
    for i, c in enumerate(classi):
        for f in sorted(os.listdir(os.path.join(dataset_dir, c))):
            if os.path.splitext(f)[1].lower() in ESTENSIONI:
                percorsi.append(os.path.join(dataset_dir, c, f))
                etichette.append(i)
    return classi, np.array(percorsi), np.array(etichette)


def dhash(percorso, size=8):
    """Hash percettivo a 64 bit: sopravvive a resize, ricompressione JPEG e
    piccoli ritagli. Serve proprio a questo, perche' i tre dataset di partenza
    (TrashNet, Garbage Classification v2, Garbage 12) si sovrappongono e
    contengono le stesse foto in risoluzioni diverse."""
    with Image.open(percorso) as im:
        im = im.convert('L').resize((size + 1, size), Image.LANCZOS)
        a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def _scansiona_uno(p):
    """md5 + dhash di un singolo file. Ritorna None se il file e' illeggibile."""
    try:
        with open(p, 'rb') as f:
            m = hashlib.md5(f.read()).hexdigest()
        return m, dhash(p), None
    except Exception as e:
        return None, np.full(8, 255, dtype=np.uint8), str(e)


def scansiona(percorsi, workers=8):
    """Ritorna md5, hash percettivi e maschera dei file validi.
    I file corrotti vengono ESCLUSI dal dataset: se restassero, il primo
    batch che li incontra fa crashare il training a meta epoca.

    Parallelizzato con thread: e' lavoro di I/O e decodifica, non di CPU pura,
    quindi il GIL non e' il collo di bottiglia."""
    md5, hashes, corrotti = [], [], []
    validi = np.ones(len(percorsi), dtype=bool)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (m, h, err) in enumerate(pool.map(_scansiona_uno, percorsi, chunksize=32)):
            md5.append(m)
            hashes.append(h)
            if err is not None:
                corrotti.append((percorsi[i], err))
                validi[i] = False
            if (i + 1) % 2000 == 0:
                print(f"  scansionate {i+1}/{len(percorsi)}", flush=True)
    assert len(md5) == len(percorsi) == len(hashes)
    return md5, np.array(hashes), corrotti, validi


class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def trova(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def unisci(self, a, b):
        ra, rb = self.trova(a), self.trova(b)
        if ra != rb: self.p[rb] = ra


def distanze(blocco_a, blocco_b):
    """Distanza di Hamming fra due blocchi di hash a 64 bit, vettorizzata."""
    xor = blocco_a[:, None, :] ^ blocco_b[None, :, :]
    return POPCOUNT[xor].sum(axis=2)


def trova_duplicati(md5, hashes, blocco=256):
    n = len(md5)
    uf = UnionFind(n)
    # 1) duplicati byte-identici
    per_md5 = defaultdict(list)
    for i, h in enumerate(md5):
        if h: per_md5[h].append(i)
    esatti = 0
    for gruppo in per_md5.values():
        for j in gruppo[1:]:
            uf.unisci(gruppo[0], j); esatti += 1
    # 2) quasi-duplicati per distanza di Hamming
    quasi = 0
    for i0 in range(0, n, blocco):
        blk = hashes[i0:i0 + blocco]
        dist = distanze(blk, hashes)
        for r in range(blk.shape[0]):
            i = i0 + r
            for j in np.where(dist[r] <= SOGLIA_HAMMING)[0]:
                if j > i:
                    if uf.trova(i) != uf.trova(j): quasi += 1
                    uf.unisci(i, j)
        if (i0 // blocco) % 10 == 0:
            print(f"  confronti {min(i0+blocco, n)}/{n}", flush=True)
    cluster = defaultdict(list)
    for i in range(n):
        cluster[uf.trova(i)].append(i)
    return cluster, esatti, quasi


def leakage_incrociato(hashes_a, hashes_b, blocco=256):
    """Quante immagini di A hanno un quasi-duplicato in B.
    Serve per il test set del Pi: se una foto della scatola e' identica a una
    del dataset web (capita se hai fotografato lo schermo o riusato immagini),
    il test set e' contaminato e i numeri finali sono una bugia."""
    if len(hashes_a) == 0 or len(hashes_b) == 0:
        return np.zeros(len(hashes_a), dtype=bool)
    sporche = np.zeros(len(hashes_a), dtype=bool)
    for i0 in range(0, len(hashes_a), blocco):
        d = distanze(hashes_a[i0:i0 + blocco], hashes_b)
        sporche[i0:i0 + blocco] = (d <= SOGLIA_HAMMING).any(axis=1)
    return sporche


def split_stratificato(etichette, fr, seed=42):
    """Split stratificato per classe: garantisce che ogni split abbia le stesse
    proporzioni. validation_split di Keras NON lo garantisce (fa shuffle e taglia)."""
    ftr, fva, fte = fr
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for c in np.unique(etichette):
        idx = np.where(etichette == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(round(n * ftr)), int(round(n * fva))
        idx_tr += list(idx[:n_tr])
        idx_va += list(idx[n_tr:n_tr + n_va])
        idx_te += list(idx[n_tr + n_va:])
    # dtype esplicito: con --modo pi il test e' vuoto, e np.array([]) sarebbe
    # float64, che non si puo' usare come indice (la v1 crashava proprio qui)
    return (np.array(v, dtype=np.int64) for v in (idx_tr, idx_va, idx_te))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='/content/dataset')
    ap.add_argument('--foto-pi', default=None,
                    help='cartella con le foto scattate dalla camera: diventano il test set')
    ap.add_argument('--modo', default='pi', choices=list(MODI))
    ap.add_argument('--out', default='split.json')
    ap.add_argument('--tieni-duplicati', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    classi, percorsi, etichette = elenca(a.dataset)
    print(f"Classi: {classi}")
    print(f"Immagini totali: {len(percorsi)}")
    for i, c in enumerate(classi):
        n = (etichette == i).sum()
        print(f"  {c:>18}: {n:5d}  ({n/len(etichette)*100:4.1f}%)")
    sbil = np.bincount(etichette).max() / np.bincount(etichette).min()
    print(f"Rapporto di sbilanciamento: {sbil:.2f}x "
          f"({'ok' if sbil < 1.5 else 'da compensare con i pesi'})")

    print("\nScansione hash (md5 + percettivo)...")
    md5, hashes, corrotti, validi = scansiona(percorsi)
    if corrotti:
        print(f"ATTENZIONE: {len(corrotti)} file corrotti, esclusi:")
        for p, e in corrotti[:10]:
            print(f"  {p}: {e}")

    print("\nRicerca duplicati...")
    cluster, esatti, quasi = trova_duplicati(md5, hashes)
    # un rappresentante per cluster, scartando i file corrotti
    da_tenere = np.array(sorted(v[0] for v in cluster.values() if validi[v[0]]))
    rimossi = len(percorsi) - len(da_tenere)
    print(f"  duplicati byte-identici : {esatti}")
    print(f"  quasi-duplicati (hamming<={SOGLIA_HAMMING}): {quasi}")
    print(f"  immagini uniche          : {len(da_tenere)} ({rimossi} rimosse, "
          f"{rimossi/len(percorsi)*100:.1f}%)")

    # I duplicati sono il motivo per cui l'accuratezza gonfia: la stessa foto
    # in train e in validation significa che il modello la ricorda, non la classifica.
    gruppi_grossi = sorted((len(v), k) for k, v in cluster.items() if len(v) > 1)[-5:]
    if gruppi_grossi:
        print("  cluster piu grandi:")
        for n, k in reversed(gruppi_grossi):
            print(f"    {n} copie di {os.path.basename(percorsi[cluster[k][0]])}")

    if a.tieni_duplicati:
        da_tenere = np.where(validi)[0]
        print("  [--tieni-duplicati] nessuna rimozione")

    idx_tr, idx_va, idx_te = split_stratificato(etichette[da_tenere], MODI[a.modo], a.seed)
    idx_tr, idx_va, idx_te = da_tenere[idx_tr], da_tenere[idx_va], da_tenere[idx_te]

    # ------------------------------------------------------------------ #
    # Test set dalle foto reali del Pi
    # ------------------------------------------------------------------ #
    test_percorsi = [percorsi[i] for i in idx_te]
    test_etichette = [int(etichette[i]) for i in idx_te]
    origine_test = f"split interno del dataset web ({len(idx_te)} img)"

    if a.foto_pi:
        print(f"\nTest set dalle foto del Pi: {a.foto_pi}")
        _, p_pi, y_pi = elenca(a.foto_pi, classi_attese=classi)
        print(f"  {len(p_pi)} foto trovate")
        for i, c in enumerate(classi):
            print(f"    {c:>18}: {(y_pi == i).sum():4d}")
        _, h_pi, corr_pi, val_pi = scansiona(p_pi)
        if corr_pi:
            print(f"  {len(corr_pi)} foto corrotte, escluse")
        # nessuna foto del Pi deve assomigliare a una del dataset web
        h_train = hashes[np.concatenate([idx_tr, idx_va])]
        sporche = leakage_incrociato(h_pi, h_train)
        tieni = val_pi & ~sporche
        if sporche.any():
            print(f"  ATTENZIONE: {int(sporche.sum())} foto del Pi hanno un quasi-duplicato "
                  f"nel dataset web -> escluse dal test set")
        if idx_te.size:
            print(f"  Nota: lo split '{a.modo}' aveva gia {len(idx_te)} immagini di test; "
                  f"le foto del Pi le sostituiscono (sono il test che conta).")
        test_percorsi = [p for p, t in zip(p_pi, tieni) if t]
        test_etichette = [int(y) for y, t in zip(y_pi, tieni) if t]
        origine_test = f"FOTO REALI dalla camera ({len(test_percorsi)} img)"

    # ------------------------------------------------------------------ #
    print(f"\nSplit '{a.modo}' {MODI[a.modo]}:")
    gruppi = [('train', [percorsi[i] for i in idx_tr], [int(etichette[i]) for i in idx_tr]),
              ('val',   [percorsi[i] for i in idx_va], [int(etichette[i]) for i in idx_va]),
              ('test',  test_percorsi, test_etichette)]
    for nome, pp, yy in gruppi:
        if not pp:
            print(f"  {nome:>6}: 0")
            continue
        dist = np.bincount(np.array(yy), minlength=len(classi))
        ic = 1.96 * np.sqrt(0.95 * 0.05 / len(pp)) * 100
        print(f"  {nome:>6}: {len(pp):5d}  {dict(zip(classi, dist))}  IC95 +-{ic:.2f}%")
    print(f"  test: {origine_test}")

    # verifica anti-leakage: a coppie, non sull'intersezione dei tre insiemi
    s_tr, s_va, s_te = (set(g[1]) for g in gruppi)
    for (na, sa), (nb, sb) in [(('train', s_tr), ('val', s_va)),
                               (('train', s_tr), ('test', s_te)),
                               (('val', s_va), ('test', s_te))]:
        comuni = sa & sb
        assert not comuni, f"LEAKAGE {na}/{nb}: {len(comuni)} file in comune, es. {list(comuni)[:3]}"
    print("  verifica leakage a coppie: OK")

    json.dump({
        'classi': classi,
        'modo': a.modo,
        'seed': a.seed,
        'origine_test': origine_test,
        'train': gruppi[0][1], 'y_train': gruppi[0][2],
        'val':   gruppi[1][1], 'y_val':   gruppi[1][2],
        'test':  gruppi[2][1], 'y_test':  gruppi[2][2],
    }, open(a.out, 'w'), indent=1)
    print(f"\nScritto {a.out} — usalo in ecosort_benchmark.py (SPLIT_JSON)")


if __name__ == '__main__':
    main()
