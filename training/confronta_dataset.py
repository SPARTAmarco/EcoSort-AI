"""
EcoSort AI — Confronta due cartelle-dataset e dice come si relazionano.

Risponde a: uno contiene l'altro? si sovrappongono a meta? sono disgiunti?
Usa md5 (identita byte) + dhash (identita percettiva, sopravvive a resize e
ricompressione). Se A e' contenuto in B, unirli non serve a niente.

    python3 confronta_dataset.py /content/dataset_rifiuti /content --classi carta_e_cartone plastica vetro_e_metallo
"""
import os, sys, hashlib, argparse
from collections import defaultdict
import numpy as np
from PIL import Image

EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
SOGLIA = 5


def dhash(p, size=8):
    with Image.open(p) as im:
        im = im.convert('L').resize((size + 1, size), Image.LANCZOS)
        a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def raccogli(base, classi):
    out = []
    for c in classi:
        d = os.path.join(base, c)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(EXT):
                out.append((os.path.join(d, f), c))
    return out


def firma(files, etichetta):
    md5s, hashes, validi = [], [], []
    for i, (p, c) in enumerate(files):
        try:
            with open(p, 'rb') as fh:
                m = hashlib.md5(fh.read()).hexdigest()
            h = dhash(p)
            md5s.append(m); hashes.append(h); validi.append((p, c))
        except Exception:
            pass
        if (i + 1) % 2000 == 0:
            print(f"  [{etichetta}] {i+1}/{len(files)}", flush=True)
    return md5s, np.array(hashes) if hashes else np.zeros((0, 8), np.uint8), validi


def quasi_uguali(hA, hB, blocco=512):
    """Per ogni immagine di A, esiste una quasi-uguale in B?"""
    if len(hA) == 0 or len(hB) == 0:
        return np.zeros(len(hA), dtype=bool)
    trovata = np.zeros(len(hA), dtype=bool)
    for i0 in range(0, len(hA), blocco):
        blk = hA[i0:i0 + blocco]
        d = POPCOUNT[blk[:, None, :] ^ hB[None, :, :]].sum(axis=2)
        trovata[i0:i0 + blocco] = (d <= SOGLIA).any(axis=1)
        print(f"  confronto {min(i0+blocco, len(hA))}/{len(hA)}", flush=True)
    return trovata


def interni(hashes):
    """Duplicati DENTRO lo stesso dataset."""
    n = len(hashes)
    visti = np.zeros(n, dtype=bool)
    dup = 0
    for i0 in range(0, n, 512):
        blk = hashes[i0:i0 + 512]
        d = POPCOUNT[blk[:, None, :] ^ hashes[None, :, :]].sum(axis=2)
        for r in range(blk.shape[0]):
            i = i0 + r
            if visti[i]:
                continue
            simili = np.where(d[r] <= SOGLIA)[0]
            for j in simili:
                if j > i and not visti[j]:
                    visti[j] = True; dup += 1
    return dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--classi', nargs='+', required=True)
    ap.add_argument('--salta-interni', action='store_true')
    args = ap.parse_args()

    fa, fb = raccogli(args.a, args.classi), raccogli(args.b, args.classi)
    print(f"A = {args.a}: {len(fa)} immagini")
    print(f"B = {args.b}: {len(fb)} immagini")
    if not fa or not fb:
        sys.exit("Una delle due cartelle e vuota")

    print("\nCalcolo hash...")
    ma, ha, va = firma(fa, 'A')
    mb, hb, vb = firma(fb, 'B')

    setb = set(mb)
    esatti = sum(1 for m in ma if m in setb)
    print(f"\n--- IDENTITA BYTE (md5) ---")
    print(f"  immagini di A presenti identiche in B: {esatti}/{len(ma)} "
          f"({esatti/len(ma)*100:.1f}%)")

    print(f"\n--- IDENTITA PERCETTIVA (dhash, hamming<={SOGLIA}) ---")
    in_b = quasi_uguali(ha, hb)
    print(f"  immagini di A che esistono anche in B: {in_b.sum()}/{len(ha)} "
          f"({in_b.mean()*100:.1f}%)")
    in_a = quasi_uguali(hb, ha)
    print(f"  immagini di B che esistono anche in A: {in_a.sum()}/{len(hb)} "
          f"({in_a.mean()*100:.1f}%)")

    print(f"\n--- VERDETTO ---")
    pa, pb = in_b.mean(), in_a.mean()
    if pb > 0.97:
        print(f"  B e' interamente contenuto in A.")
        print(f"  -> USA SOLO A ({args.a}). Unirli non aggiunge nulla.")
    elif pa > 0.97:
        print(f"  A e' interamente contenuto in B.")
        print(f"  -> USA SOLO B ({args.b}).")
    elif pa > 0.5 or pb > 0.5:
        nuovi = (~in_a).sum()
        print(f"  Sovrapposizione parziale.")
        print(f"  -> A + dedup e' la scelta migliore: B aggiunge solo {nuovi} immagini nuove.")
    else:
        print(f"  Quasi disgiunti: unirli darebbe circa "
              f"{len(ha) + (~in_a).sum()} immagini uniche.")
        print(f"  -> conviene unirli e poi passare prepara_dataset.py")

    if not args.salta_interni:
        print(f"\n--- DUPLICATI INTERNI ---")
        da, db = interni(ha), interni(hb)   # calcolati una volta sola: sono O(n^2)
        print(f"  dentro A: {da} ({da/len(ha)*100:.1f}%)")
        print(f"  dentro B: {db} ({db/len(hb)*100:.1f}%)")

    print("\nDistribuzione per classe:")
    for nome, v in [('A', va), ('B', vb)]:
        d = defaultdict(int)
        for _, c in v:
            d[c] += 1
        tot = sum(d.values())
        sb = max(d.values()) / min(d.values()) if d else 0
        print(f"  {nome}: {dict(d)}  sbilanciamento {sb:.2f}x  totale {tot}")


if __name__ == '__main__':
    main()
