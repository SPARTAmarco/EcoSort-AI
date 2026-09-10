# Training

La pipeline di addestramento gira su **Google Colab** (GPU T4) e non è versionata qui:
i notebook e i moduli vivono su Drive perché richiedono il dataset completo (8.139 immagini,
~2 GB) che non può stare nel repository.

Struttura prevista di questa cartella:

```text
training/
├── prepara_dataset.py     dedup + split stratificato
├── ecosort_benchmark.py   confronto di 4 backbone con selezione a costo
├── train_finale.py        riaddestramento del vincitore su train+val
├── converti_tflite.py     export float32 / float16 / INT8 + verifica
├── ecosort_decisione.py   matrice di costo e regola bayesiana (condiviso col Pi)
└── EcoSort_Colab.ipynb    notebook guidato
```

Il funzionamento dei singoli moduli e le decisioni tecniche sono documentati in
[`../docs/pipeline.md`](../docs/pipeline.md).

## Dataset

Fusione di tre dataset pubblici, deduplicata e rimappata su 3 macro-classi:

| origine | licenza |
|---|---|
| TrashNet | MIT |
| Garbage Classification v2 | vedi pagina Kaggle |
| Garbage 12 classi | vedi pagina Kaggle |

Il dataset derivato non è ridistribuito in questo repository.
