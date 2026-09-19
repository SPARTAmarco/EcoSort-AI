<div align="center">

# ♻️ EcoSort AI

**Sistema automatico di classificazione e smistamento dei rifiuti basato su intelligenza artificiale**

[![License: MIT](https://img.shields.io/badge/License-MIT-D4FF3A.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3B82F6.svg)](https://www.python.org/)
[![Java](https://img.shields.io/badge/Java-21-F59E0B.svg)](https://openjdk.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-00D97E.svg)](https://www.tensorflow.org/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-10B981.svg)](https://www.raspberrypi.com/)

Progetto realizzato per il **Premio GF Marilli 2026** — HackersGen & Sorint, Bergamo

</div>

---

## Cos'è

L'utente inserisce un rifiuto in una scatola dotata di camera. Il sistema lo fotografa,
lo classifica con una rete neurale addestrata su misura e una **piattaforma rotante con
sensori Hall** porta automaticamente il bidone corretto sotto l'apertura di caduta.

Nessun pulsante da premere, nessuna scelta da fare: il rifiuto entra, il bidone giusto arriva.

## Le classi

| Classe | Bidone | Colore |
|---|---|---|
| `plastica` | Giallo | ![#F59E0B](https://placehold.co/12x12/F59E0B/F59E0B.png) `#F59E0B` |
| `carta_e_cartone` | Blu | ![#3B82F6](https://placehold.co/12x12/3B82F6/3B82F6.png) `#3B82F6` |
| `vetro_e_metallo` | Verde | ![#10B981](https://placehold.co/12x12/10B981/10B981.png) `#10B981` |
| `indifferenziata` | Grigio | ![#9CA3AF](https://placehold.co/12x12/9CA3AF/9CA3AF.png) `#9CA3AF` |

`indifferenziata` non è una classe del modello: è l'**azione di fallback** scelta quando la
confidenza non basta a giustificare il rischio di un errore. Non c'è una soglia unica: la
decide la [matrice di costo](docs/matrice-costo-errori.md), rifiuto per rifiuto.

## Il modello

| | |
|---|---|
| **Architettura** | EfficientNetB0, transfer learning da ImageNet (TensorFlow / Keras) |
| **Selezione** | Vincitore di un benchmark su 4 backbone: MobileNetV3Large, EfficientNetB0, EfficientNetV2B0, ResNet50V2 |
| **Dataset** | 8.139 immagini reali: fusione di TrashNet, Garbage Classification v2 e Garbage 12 classi, deduplicate (md5 + dhash) e riclassificate in 3 macro-categorie |
| **Training** | Google Colab, GPU NVIDIA T4, in due fasi: testa di classificazione, poi fine-tuning |
| **Accuratezza** | **97,3%** sul test set: 1.093 immagini mai viste in training (96,8% nel benchmark comparativo) |
| **Decisione** | Regola bayesiana a costo atteso minimo, con probabilità calibrate (temperature scaling, T = 0,749) |
| **Coverage** | 96,2% dei rifiuti smistati; il 3,8% più incerto va in indifferenziata |
| **Errori gravi** | Vetro/metallo nel bidone di carta o plastica: **da 9 a 3** rispetto all'`argmax` |
| **Calibrazione** | ECE da 0,027 a **0,007** |
| **Input** | 224×224 RGB, preprocessing incorporato nel modello esportato |
| **Deployment** | `rifiuti.tflite` (float16, 8,3 MB) via `ai-edge-litert` sul Raspberry Pi · `newbest_model.keras` nativo nell'app desktop |

> Il modello non ottimizza l'accuratezza pura ma **minimizza il danno all'impianto di riciclo**:
> mandare vetro nel macero della carta costa 9.0, mandare carta nel vetro costa 2.0. La regola
> di decisione è bayesiana a costo atteso minimo, non un semplice `argmax`.
> Dettagli in [`docs/matrice-costo-errori.md`](docs/matrice-costo-errori.md).

## Hardware

| Componente | Modello |
|---|---|
| Calcolatore | Raspberry Pi 4B |
| Camera | CSI Camera Module V2.1 |
| Attuatore | Servo SG90 |
| Posizionamento | Piattaforma rotante con sensori Hall |

## Struttura del repository

```text
EcoSort-AI/
├── desktop-app/          App JavaFX (dimostratore da PC)
│   ├── src/main/java/    Codice sorgente Java 21
│   ├── server/           Micro-server Flask che carica il modello .keras
│   ├── pom.xml           Build Maven (shade plugin → jar unico)
│   └── build.bat         Script di build per Windows
├── raspberry-pi/         Inferenza a bordo macchina
│   ├── classifica_pi.py  LiteRT + regola di decisione a costo
│   └── raccogli_foto.py  Raccolta di foto reali con la camera del Pi
├── training/             Pipeline di addestramento (Google Colab)
│   ├── EcoSort_Colab.ipynb   Notebook guidato, dal dataset al .tflite
│   ├── prepara_dataset.py    Dedup (md5 + dhash) e split stratificato
│   ├── ecosort_benchmark.py  Confronto di 4 backbone con selezione a costo
│   ├── train_finale.py       Training definitivo su train + validation
│   ├── converti_tflite.py    Export float32/float16/INT8 con verifica
│   ├── valuta_foto_pi.py     Valutazione sulle foto scattate dalla camera
│   └── ecosort_decisione.py  Matrice di costo (condiviso col Pi)
├── tools/
│   └── prova_pc.py       Banco di prova da PC: webcam o foto, stessa logica del Pi
├── docs/                 Documentazione tecnica e decisioni di progetto
├── .env.example          Template per la chiave API Gemini
└── LICENSE               MIT
```

## Modelli pre-addestrati

I pesi **non sono versionati nel repository**: si scaricano dalla
[**Release v1.0.0**](../../releases/tag/v1.0.0).

| File | Dimensione | Uso | Dove metterlo |
|---|---|---|---|
| `newbest_model.keras` | 69,3 MB | App desktop (modalità offline) | `desktop-app/server/` |
| `rifiuti.tflite` | 8,3 MB | Raspberry Pi (float16) | `raspberry-pi/` |
| `config.json` | 1,2 KB | Soglia, etichette, calibrazione per il Pi | `raspberry-pi/` |

## Avvio rapido

### App desktop (Windows) — un doppio clic

```bat
desktop-app\avvia.bat
```

La prima volta scarica i modelli dalla Release v1.0.0, crea un ambiente Python `.venv` con
TensorFlow e compila l'app con Maven (qualche minuto). Dalla seconda volta parte subito.
Servono solo **Python 3.10–3.13**, **JDK 21** e **Maven**.

```bat
desktop-app\avvia.bat test        :: banco di prova nel browser (webcam o foto), senza JavaFX
desktop-app\avvia.bat ricompila   :: ricompila il jar dopo una modifica al codice Java
```

L'app parte in **modalità offline**: il server Flask locale carica `newbest_model.keras`
(il primo caricamento di TensorFlow richiede qualche secondo). Per la **modalità online**
(Google Gemini) serve una chiave API in `.env` (vedi `.env.example`) oppure incollata
nell'interfaccia dell'app.

### Raspberry Pi

```bash
# Su Raspberry Pi OS (Debian Trixie, Python 3.13) — senza virtualenv
sudo apt install -y python3-picamera2 python3-numpy python3-pil
pip install -r raspberry-pi/requirements-rpi.txt --break-system-packages

# Scaricare rifiuti.tflite e config.json dalla Release v1.0.0 in raspberry-pi/
cp training/ecosort_decisione.py raspberry-pi/
python3 raspberry-pi/classifica_pi.py
```

### Prova da PC senza Raspberry

```bash
pip install flask pillow numpy tensorflow
python tools/prova_pc.py      # apre il browser: webcam o trascina una foto
```

Usa lo stesso modello, lo stesso preprocessing e la stessa regola di decisione del Pi.

## Consiglio per la precisione

> Fotografare il rifiuto su uno **sfondo neutro e uniforme**. Sfondi confusi o pieni di altri
> oggetti abbassano sensibilmente l'accuratezza del riconoscimento.

## Documentazione

- [Pipeline di training e deployment](docs/pipeline.md) — dai dati grezzi al modello sul Pi
- [Matrice di costo degli errori](docs/matrice-costo-errori.md) — perché non basta l'accuratezza

## Licenza

Distribuito con licenza [MIT](LICENSE) — © 2026 Marco Bosio.

I dataset di partenza (TrashNet, Garbage Classification v2, Garbage 12 classi) restano soggetti
alle rispettive licenze originali.
