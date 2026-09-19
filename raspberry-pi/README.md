# Raspberry Pi

Inferenza a bordo macchina: scatto, classificazione, scelta del bidone.

## Ambiente

- Raspberry Pi 4B — Raspberry Pi OS (Debian Trixie), Python 3.13
- CSI Camera Module V2.1 via `picamera2`
- Servo SG90 + piattaforma rotante con sensori Hall

## File da avere nella stessa cartella sul Pi

```text
classifica_pi.py        questo repository
ecosort_decisione.py    da ../training/ (modulo condiviso)
rifiuti.tflite          Release v1.0.0 (float16, 8,3 MB)
config.json             Release v1.0.0 (soglia 0,73, etichette, temperatura)
```

## Installazione

Sul Pi l'ambiente virtuale non è attivo: si installa a livello di sistema.

```bash
sudo apt install -y python3-picamera2 python3-numpy python3-pil
pip install ai-edge-litert numpy pillow --break-system-packages
```

> Non usare `tflite-runtime`: è fermo alla 2.14.0 (ottobre 2023) e ha wheel solo fino a
> Python 3.11, quindi sul Pi con Python 3.13 non si installa. Il successore ufficiale è
> `ai-edge-litert`, con wheel manylinux aarch64 per cp310–cp314.

## Uso

```bash
python3 classifica_pi.py --bench            # misura la latenza reale di inferenza
python3 classifica_pi.py --foto prova.jpg   # classifica un file esistente
python3 classifica_pi.py                    # INVIO per scattare con la camera
```

## Come decide

Il modello restituisce le tre probabilità; la scelta del bidone **non** è un `argmax`.
`ecosort_decisione.py` calcola il costo atteso di ognuna delle quattro azioni
(blu, giallo, verde, grigio) e sceglie quella che lo minimizza: se nessuna destinazione
"vera" conviene, il rifiuto va in **indifferenziata**. La soglia non è un numero scelto a
occhio, emerge dalla matrice di costo — dettagli in
[`../docs/matrice-costo-errori.md`](../docs/matrice-costo-errori.md).

Il preprocessing è incorporato nel `.tflite`: lo script passa l'immagine uint8 [0,255]
direttamente al modello, quindi è impossibile che il preprocessing sul Pi diverga da
quello usato in training.
