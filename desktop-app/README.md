# App desktop (JavaFX)

Dimostratore da PC: stessa intelligenza della macchina fisica, ma con foto caricate
dall'utente o scattate dalla webcam.

## Avvio rapido

Doppio clic su **`avvia.bat`**. Fa tutto da solo e, dalla seconda volta, salta i passi già fatti:

1. scarica `newbest_model.keras`, `rifiuti.tflite` e `config.json` dalla
   [Release v1.0.0](../../../releases/tag/v1.0.0) in `server/`
2. crea l'ambiente Python `.venv` e installa `server/requirements.txt`
3. compila il jar con Maven
4. avvia l'app

| comando | cosa fa |
|---|---|
| `avvia.bat` | prepara e apre l'app |
| `avvia.bat lite` | apre l'app con `rifiuti.tflite` invece del `.keras`: lo stesso file del Raspberry |
| `avvia.bat test` | banco di prova nel browser (`tools/prova_pc.py`): webcam o foto trascinata, stessa logica del Raspberry |
| `avvia.bat ricompila` | ricompila il jar dopo una modifica al codice Java |

Requisiti: **Python 3.10–3.13** (TensorFlow non supporta ancora versioni più nuove),
**JDK 21+**, **Maven**. Il log del server Python finisce in `ecosort-server.log`.

## Due modalità

**Offline** — un micro-server Flask (`server/server.py`) carica `newbest_model.keras`
(EfficientNetB0) con TensorFlow. Il preprocessing è incorporato nel modello: il server passa
i pixel grezzi 0–255, esattamente come il Raspberry Pi. Anche la decisione è la stessa del Pi:
legge temperatura e matrice di costo da `server/config.json` e sceglie il bidone a costo atteso
minimo, indifferenziata compresa. Se `config.json` manca, ripiega su una soglia fissa di 0,73
(la migliore trovata nel benchmark).

**Online** — chiamata alle API di Google Gemini con l'immagine in base64, ridimensionata a
300 px e compressa al 50% prima dell'invio per ridurre banda e latenza.

`server/server_lite.py` è l'alternativa senza TensorFlow completo: usa `rifiuti.tflite`
con `ai-edge-litert`, stesso preprocessing e stessa regola di decisione.

## Python usato dall'app

L'app cerca l'interprete in quest'ordine: variabile `ECOSORT_PYTHON` (la imposta
`avvia.bat`), `.venv/`, `python-embedded/` accanto al jar, installazioni di sistema.

## Build manuale

```bash
mvn clean package     # oppure build.bat su Windows
java -jar target/ecosort-ai-1.0.0.jar
```

## Chiave API

La chiave Gemini si può fornire in tre modi (in ordine di priorità):

1. variabile d'ambiente `GEMINI_API_KEY`
2. file `.env` accanto al jar (vedi `../.env.example`)
3. campo dedicato nell'interfaccia dell'app (salvata nelle Java Preferences)
