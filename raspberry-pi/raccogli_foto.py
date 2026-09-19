#!/usr/bin/env python3
"""
EcoSort AI — Raccolta guidata delle foto di test con la camera del Pi.

Da eseguire SUL RASPBERRY, con la camera gia montata nella scatola e la luce
accesa: le foto devono nascere nelle stesse identiche condizioni in cui il
sistema lavorera. Se le scatti a mano su un tavolo, non stai piu misurando il
domain shift, stai creando un secondo dataset web.

    python3 raccogli_foto.py --classe plastica
    python3 raccogli_foto.py --lista
    python3 raccogli_foto.py --classe vetro_e_metallo --raffica 3

INVIO scatta, 'q' + INVIO passa alla classe successiva o esce.

Le foto finiscono in foto_pi/<classe>/<classe>_<progressivo>.jpg, che e'
esattamente la struttura che prepara_dataset.py si aspetta in --foto-pi.

CONSIGLI PER UN TEST SET ONESTO
-------------------------------
- Oggetti DIVERSI, non lo stesso oggetto 70 volte. 20-25 oggetti per classe,
  3 scatti ciascuno (posizioni e rotazioni diverse) sono meglio di 70 scatti
  della stessa bottiglia: quelli misurano una cosa sola.
- Includi i casi difficili che vedrai davvero: la bottiglia schiacciata, il
  cartone del latte (che e' carta ma sembra plastica), il barattolo con
  l'etichetta di carta, il vetro trasparente su fondo chiaro.
- Non ritoccare e non scartare le foto venute male per motivi di
  illuminazione: se succede nella scatola, deve stare nel test set.
- Queste foto non entreranno MAI nel training. Servono solo a misurare.
"""
import argparse, os, sys, time

CLASSI = ['carta_e_cartone', 'plastica', 'vetro_e_metallo']
RISOLUZIONE = (1280, 960)   # la stessa di classifica_pi.py
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foto_pi')


def conta():
    tot = 0
    print(f"\nCartella: {BASE}")
    for c in CLASSI:
        d = os.path.join(BASE, c)
        n = len([f for f in os.listdir(d) if f.lower().endswith('.jpg')]) if os.path.isdir(d) else 0
        tot += n
        barra = '#' * min(40, n // 2)
        stato = 'ok' if n >= 60 else ('scarse' if n >= 30 else 'troppo poche')
        print(f"  {c:>18}: {n:4d}  {barra:<40} {stato}")
    print(f"  {'TOTALE':>18}: {tot:4d}   (obiettivo: 200+, il piu bilanciate possibile)\n")
    return tot


def apri_camera():
    try:
        from picamera2 import Picamera2
    except ImportError:
        sys.exit("picamera2 non trovato. Sul Pi:\n"
                 "  sudo apt install -y python3-picamera2")
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": RISOLUZIONE}))
    cam.start()
    time.sleep(2)          # l'esposizione automatica ha bisogno di assestarsi
    return cam


def prossimo_indice(cartella, classe):
    esistenti = [f for f in os.listdir(cartella) if f.startswith(classe) and f.endswith('.jpg')]
    numeri = []
    for f in esistenti:
        try:
            numeri.append(int(f.rsplit('_', 1)[1].split('.')[0]))
        except (IndexError, ValueError):
            pass
    return max(numeri) + 1 if numeri else 1


def raccogli(cam, classe, raffica):
    cartella = os.path.join(BASE, classe)
    os.makedirs(cartella, exist_ok=True)
    i = prossimo_indice(cartella, classe)
    print(f"\n=== {classe.upper()} ===")
    print("Metti l'oggetto nella scatola e premi INVIO per scattare.")
    print("'q' + INVIO per passare oltre.\n")
    scattate = 0
    while True:
        try:
            if input(f"[{classe}] scatto #{i} > ").strip().lower() == 'q':
                break
        except (EOFError, KeyboardInterrupt):
            break
        for k in range(raffica):
            percorso = os.path.join(cartella, f"{classe}_{i:04d}.jpg")
            cam.capture_file(percorso)
            kb = os.path.getsize(percorso) / 1024
            print(f"    salvata {os.path.basename(percorso)}  ({kb:.0f} KB)")
            i += 1
            scattate += 1
            if raffica > 1 and k < raffica - 1:
                time.sleep(0.4)   # tempo di spostare o ruotare l'oggetto
    return scattate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classe', choices=CLASSI,
                    help='classe da fotografare; senza questo le percorre tutte')
    ap.add_argument('--raffica', type=int, default=1,
                    help='scatti per ogni INVIO (utile per ruotare l oggetto fra uno e l altro)')
    ap.add_argument('--lista', action='store_true', help='mostra solo i conteggi ed esce')
    a = ap.parse_args()

    os.makedirs(BASE, exist_ok=True)
    if a.lista:
        conta()
        return

    conta()
    cam = apri_camera()
    totale = 0
    try:
        for c in ([a.classe] if a.classe else CLASSI):
            totale += raccogli(cam, c, max(1, a.raffica))
    finally:
        cam.stop()

    print(f"\n{totale} foto nuove in questa sessione.")
    conta()
    print("Quando hai finito, comprimi e porta lo zip su Drive:")
    print(f"  cd {os.path.dirname(BASE)} && zip -r foto_pi.zip foto_pi")


if __name__ == '__main__':
    main()
