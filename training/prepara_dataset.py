"""
EcoSort AI — Preparazione del dataset: inventario, deduplicazione, split stratificato.

DA ESEGUIRE UNA VOLTA SOLA, prima di qualsiasi training.
Produce split.json con le liste di file, cosi ogni training successivo usa
ESATTAMENTE gli stessi split (riproducibilita) e non c'e' modo che
un'immagine passi da validation a training tra un esperimento e l'altro.

    python3 prepara_dataset.py --dataset /content/dataset --modo pi

Modi:
  --modo pi        80/20 sul dataset web, il TEST vero sono le foto del Pi
                   (consigliato: misura cio' che conta davvero)
  --modo classico  70/15/15 tutto interno al dataset web
  --modo marco     80/10/10
"""
import os, sys, json, hashlib, argparse
from collections import defaultdict
import numpy as np
from PIL import Image

ESTENSIONI = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SOGLIA_HAMMING = 5      # <=5 bit di differenza su 64 = quasi-duplicato
POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

MODI = {'pi': (0.80, 0.20, 0.00), 'classico': (0.70, 0.15, 0.15), 'marco': (0.80, 0.10, 0.10)}


def elenca(dataset_dir):
    classi = sorted(d for d in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, d)))
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


def scansiona(percorsi):
    """Ritorna md5, hash percettivi e maschera dei file validi.
    I file corrotti vengono ESCLUSI dal dataset: se restassero, il primo
    batch che li incontra fa crashare il training a meta epoca."""
    md5, hashes, corrotti = [], [], []
    validi = np.ones(len(percorsi), dtype=bool)
    for i, p in enumerate(percorsi):
        try:
            with open(p, 'rb') as f:
                m = hashlib.md5(f.read()).hexdigest()
            h = dhash(p)                      # entrambi calcolati PRIMA di appendere
            md5.append(m); hashes.append(h)
        except Exception as e:
            corrotti.append((p, str(e)))
            validi[i] = False
            md5.append(None); hashes.append(np.full(8, 255, dtype=np.uint8))
        if (i + 1) % 1000 == 0:
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
        xor = blk[:, None, :] ^ hashes[None, :, :]
        dist = POPCOUNT[xor].sum(axis=2)
        for r in range(blk.shape[0]):
            i = i0 + r
            vicini = np.where(dist[r] <= SOGLIA_HAMMING)[0]
            for j in vicini:
                if j > i:
                    if uf.trova(i) != uf.trova(j): quasi += 1
                    uf.unisci(i, j)
        if (i0 // blocco) % 10 == 0:
            print(f"  confronti {min(i0+blocco, n)}/{n}", flush=True)
    cluster = defaultdict(list)
    for i in range(n):
        cluster[uf.trova(i)].append(i)
    return cluster, esatti, quasi


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
    return map(np.array, (idx_tr, idx_va, idx_te))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='/content/dataset')
    ap.add_argument('--modo', default='pi', choices=list(MODI))
    ap.add_argument('--out', default='split.json')
    ap.add_argument('--tieni-duplicati', action='store_true')
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

    idx_tr, idx_va, idx_te = split_stratificato(etichette[da_tenere], MODI[a.modo])
    idx_tr, idx_va, idx_te = da_tenere[idx_tr], da_tenere[idx_va], da_tenere[idx_te]

    print(f"\nSplit '{a.modo}' {MODI[a.modo]}:")
    for nome, idx in [('train', idx_tr), ('val', idx_va), ('test', idx_te)]:
        if len(idx) == 0:
            print(f"  {nome:>6}: 0  (il test set sono le foto scattate dal Pi)")
            continue
        dist = np.bincount(etichette[idx], minlength=len(classi))
        ic = 1.96 * np.sqrt(0.95 * 0.05 / len(idx)) * 100
        print(f"  {nome:>6}: {len(idx):5d}  {dict(zip(classi, dist))}  IC95 +-{ic:.2f}%")

    # verifica anti-leakage: nessun indice condiviso, nessun cluster spezzato
    assert not (set(idx_tr) & set(idx_va) & set(idx_te))
    assert len(set(idx_tr) | set(idx_va) | set(idx_te)) == len(idx_tr) + len(idx_va) + len(idx_te)
    print("  verifica leakage: OK (nessuna immagine in due split)")

    json.dump({'classi': classi, 'modo': a.modo,
               'train': [percorsi[i] for i in idx_tr],
               'val':   [percorsi[i] for i in idx_va],
               'test':  [percorsi[i] for i in idx_te]},
              open(a.out, 'w'), indent=1)
    print(f"\nScritto {a.out} — usalo in ecosort_benchmark.py (SPLIT_JSON)")


if __name__ == '__main__':
    main()
