# App desktop (JavaFX)

Dimostratore da PC: stessa intelligenza della macchina fisica, ma con foto caricate
dall'utente o scattate dalla webcam.

## Due modalità

**Offline** — un micro-server Flask (`server/server.py`) carica il modello `.keras` con
TensorFlow e risponde in pochi millisecondi. Il primo avvio richiede 6-7 secondi per
l'import di TensorFlow e il caricamento dei pesi.

**Online** — chiamata alle API di Google Gemini con l'immagine in base64, ridimensionata a
300 px e compressa al 50% prima dell'invio per ridurre banda e latenza.

## Build

```bash
mvn clean package     # oppure build.bat su Windows
java -jar target/ecosort-ai-1.0.0.jar
```

Richiede JDK 21+. Il `maven-shade-plugin` produce un jar unico con tutte le dipendenze.

## Chiave API

La chiave Gemini si può fornire in tre modi (in ordine di priorità):

1. variabile d'ambiente `GEMINI_API_KEY`
2. file `.env` accanto al jar (vedi `../.env.example`)
3. campo dedicato nell'interfaccia dell'app (salvata nelle Java Preferences)

## Modello

`server/newbest_model.keras` non è nel repository: scaricarlo dalla sezione Releases.
