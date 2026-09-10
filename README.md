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
confidenza non è sufficiente a giustificare il rischio di un errore (vedi
[matrice di costo](docs/matrice-costo-errori.md)).

## Il modello

- **Architettura:** EfficientNetB0 con transfer learning (TensorFlow / Keras)
- **Dataset:** 8.139 immagini reali, fusione di TrashNet + Garbage Classification v2 + Garbage 12 classi, deduplicate e riclassificate in 3 macro-categorie
- **Accuratezza:** ~95% sul validation set
- **Input:** 224×224 RGB, preprocessing incorporato nel modello esportato
- **Deployment:** `.tflite` via `ai-edge-litert` sul Raspberry Pi, `.keras` nativo nell'app desktop

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
├── raspberry-pi/         Codice di inferenza a bordo macchina
├── training/             Pipeline di addestramento (Google Colab)
├── docs/                 Documentazione tecnica e decisioni di progetto
├── .env.example          Template per la chiave API Gemini
└── LICENSE               MIT
```

## Modelli pre-addestrati

I pesi **non sono versionati nel repository** (il `.keras` pesa 232 MB, oltre il limite di GitHub).
Si scaricano dalla sezione [**Releases**](../../releases):

| File | Dimensione | Uso |
|---|---|---|
| `newbest_model.keras` | ~232 MB | App desktop (modalità offline) |
| `newbest_model.tflite` | ~24 MB | Raspberry Pi |

Posizionarli rispettivamente in `desktop-app/server/` e `raspberry-pi/`.

## Avvio rapido

### App desktop (Windows)

```bash
# 1. Dipendenze Python del server locale
cd desktop-app/server
pip install -r requirements.txt

# 2. Scaricare newbest_model.keras dalle Releases in questa cartella

# 3. Build dell'applicazione JavaFX
cd ..
mvn clean package        # oppure: build.bat

# 4. Avvio
java -jar target/ecosort-ai-1.0.0.jar
```

L'app parte in **modalità offline** e avvia il server Flask in background: il primo caricamento
di TensorFlow richiede 6-7 secondi. Quando la barra di stato mostra *"Server offline pronto!"*
si può analizzare la prima immagine.

Per la **modalità online** (Google Gemini) serve una chiave API:

```bash
cp .env.example .env      # e inserire la propria chiave
```

La chiave può anche essere incollata direttamente nell'interfaccia dell'app.

### Raspberry Pi

```bash
# Su Raspberry Pi OS (Debian Trixie, Python 3.13) — senza virtualenv
sudo apt install -y python3-picamera2 python3-numpy python3-pil
pip install -r raspberry-pi/requirements-rpi.txt --break-system-packages

# Scaricare newbest_model.tflite dalle Releases in raspberry-pi/
python3 raspberry-pi/classifica_pi.py
```

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
