# Pipeline di training e deployment (Colab → Raspberry Pi)

## Deploy: TFLite / LiteRT

| pacchetto | versione | Python | aarch64 | verdetto |
|---|---|---|---|---|
| `tflite-runtime` | 2.14.0 | 3.8–3.11 | sì | **non installabile** su Python 3.13 |
| `ai-edge-litert` | 2.2.0 | 3.10–3.14 | manylinux_2_27 | **in uso** |
| `onnxruntime` | 1.29.0 | 3.11–3.14 | manylinux_2_28 | alternativa valida |

Scelto TFLite via `ai-edge-litert`: XNNPACK è più veloce di ONNX Runtime su ARM per
architetture mobile, e la quantizzazione INT8 è più matura.

```bash
pip install ai-edge-litert numpy pillow --break-system-packages
```

## I moduli

| file | ruolo | dove gira |
|---|---|---|
| `ecosort_decisione.py` | matrice di costo + regola bayesiana | Colab + Pi |
| `prepara_dataset.py` | dedup (md5 + hash percettivo) + split stratificato | Colab, una volta sola |
| `ecosort_benchmark.py` | 4 backbone, stessa pipeline, selezione a costo | Colab |
| `train_finale.py` | riaddestra il vincitore su train+val | Colab |
| `converti_tflite.py` | float32 / float16 / INT8 + verifica accuratezza | Colab |
| `classifica_pi.py` | inferenza con LiteRT | Pi |

## Decisioni tecniche

**Preprocessing incorporato nel modello.** Il `.tflite` accetta direttamente l'immagine uint8
[0,255] della camera. Elimina per costruzione il bug più comune del deployment: preprocessing
divergente tra training e dispositivo.

**Output float32 anche in INT8.** Le probabilità alimentano la regola di decisione a costo;
quantizzarle a 8 bit darebbe risoluzione 1/256 e degraderebbe le soglie implicite.

**Quantizzazione verificata, non assunta.** Tutte e tre le varianti sono valutate sul test set.
INT8 viene scelto solo se perde ≤1,0 punti di accuratezza e ≤0,02 di costo.

**Training finale su train+val.** La validation serve a scegliere backbone ed epoche; dopo
averlo fatto, tenerla fuori butta via il 25% dei dati. Il test set resta intoccato.

**Epoche ottimali propagate.** Il benchmark salva `epoche_fase1` / `epoche_fase2` in
`config.json`; il training finale le riusa (senza validation non può fare early stopping).

## Ordine di esecuzione

```
prepara_dataset.py --modo pi   →  split.json
ecosort_benchmark.py           →  stato_benchmark.json, config.json
train_finale.py                →  newbest_model.keras
converti_tflite.py             →  rifiuti.tflite
```

Sul Pi: `classifica_pi.py --bench` per misurare la latenza reale.
