# Training

Pipeline completa dal dataset grezzo al file `.tflite` pronto per il Raspberry Pi.
Gira su **Google Colab** (GPU T4 del piano gratuito): il notebook
[`EcoSort_Colab.ipynb`](EcoSort_Colab.ipynb) guida passo per passo e richiama i moduli.

## Ordine di esecuzione

```text
0. EcoSort_Colab.ipynb      apri questo e segui le celle
1. confronta_dataset.py     [solo con più dataset] dice se uno contiene l'altro
2. prepara_dataset.py       dedup + split stratificato       →  split.json
3. ecosort_benchmark.py     4 backbone, selezione a costo    →  config.json
4. train_finale.py          riaddestra il vincitore su train+val → newbest_model.keras
5. converti_tflite.py       float32 / float16 / INT8 + verifica  → rifiuti.tflite
```

## I moduli

| file | ruolo | gira su |
|---|---|---|
| `ecosort_decisione.py` | matrice di costo + regola bayesiana + soglie implicite | Colab **e** Pi |
| `confronta_dataset.py` | confronta due dataset per identità byte (md5) e percettiva (dhash) | Colab |
| `prepara_dataset.py` | inventario, deduplicazione, split stratificato riproducibile | Colab |
| `ecosort_benchmark.py` | addestra 4 backbone con pipeline identica e li confronta a costo | Colab |
| `train_finale.py` | training definitivo del vincitore su train + validation | Colab |
| `converti_tflite.py` | conversione in 3 varianti, tutte valutate sul test set | Colab |

`ecosort_decisione.py` è condiviso con il Raspberry Pi: va copiato sul Pi accanto a
`classifica_pi.py` (il Passo 6 del notebook prepara già il pacchetto pronto).

## Modalità di split

`prepara_dataset.py` accetta tre modi:

| modo | proporzioni | quando |
|---|---|---|
| `pi` | 80 / 20 / — | il test set vero sono le foto scattate dalla camera nella scatola |
| `classico` | 70 / 15 / 15 | tutto interno al dataset web |
| `marco` | 80 / 10 / 10 | via di mezzo |

```bash
python3 prepara_dataset.py --dataset /content/dataset --modo pi
```

Lo split viene salvato in `split.json`: ogni training successivo riusa esattamente
gli stessi file, quindi nessuna immagine può migrare da validation a training tra un
esperimento e l'altro.

## Perché la deduplicazione

I tre dataset di partenza (TrashNet, Garbage Classification v2, Garbage 12 classi) si
sovrappongono e contengono le stesse foto in risoluzioni diverse. Il solo md5 non basta:
serve un hash percettivo (dhash a 64 bit, soglia 5 bit di distanza di Hamming) che
sopravvive a resize e ricompressione JPEG. Un duplicato che finisce metà in training e
metà in validation gonfia l'accuratezza di parecchi punti senza che nessuno se ne accorga.

## Selezione del modello

Il vincitore del benchmark non è il più accurato: è quello che **minimizza il costo medio
per rifiuto** secondo la matrice descritta in
[`../docs/matrice-costo-errori.md`](../docs/matrice-costo-errori.md), tenendo conto anche
della latenza sul Pi.

## Dataset

Non ridistribuito in questo repository (~8.139 immagini). Fusione deduplicata di:

| origine | licenza |
|---|---|
| TrashNet | MIT |
| Garbage Classification v2 | vedi pagina Kaggle |
| Garbage 12 classi | vedi pagina Kaggle |

Struttura attesa in `/content/dataset/`: una sottocartella per classe
(`carta_e_cartone/`, `plastica/`, `vetro_e_metallo/`).
