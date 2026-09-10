# Raspberry Pi

Codice che gira a bordo della macchina: scatto, classificazione, rotazione della piattaforma.

## Ambiente

- Raspberry Pi 4B — Raspberry Pi OS (Debian Trixie), Python 3.13
- CSI Camera Module V2.1 via `picamera2`
- Servo SG90 + piattaforma rotante con sensori Hall

## Installazione

Sul Pi l'ambiente virtuale non è attivo: installare a livello di sistema.

```bash
sudo apt install -y python3-picamera2 python3-numpy python3-pil
pip install -r requirements-rpi.txt --break-system-packages
```

Scaricare `newbest_model.tflite` dalla sezione Releases e posizionarlo in questa cartella.

## Uso

```bash
python3 classifica_pi.py          # ciclo normale
python3 classifica_pi.py --bench  # misura la latenza di inferenza
```

La classe vincente viene tradotta in un'azione tramite la regola di decisione a costo
descritta in [`../docs/matrice-costo-errori.md`](../docs/matrice-costo-errori.md):
sotto la soglia implicita il rifiuto va in **indifferenziata**.
