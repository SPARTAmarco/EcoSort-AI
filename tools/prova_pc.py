#!/usr/bin/env python3
"""
EcoSort AI — Banco di prova sul PC.

Apre una paginetta locale nel browser dove puoi:
  - inquadrare un oggetto con la webcam e scattare
  - trascinare dentro una foto qualsiasi

e vedere la classificazione ESATTAMENTE come la calcola il Raspberry:
stesso preprocessing, stessa temperatura, stessa regola di decisione a costo.
Non e' una demo semplificata: e' lo stesso identico percorso del Pi.

    python prova_pc.py
    python prova_pc.py --modello C:\\percorso\\rifiuti.tflite --config C:\\percorso\\config.json
    python prova_pc.py --cartella C:\\foto_da_provare     (modalita batch, senza browser)

Se non passi --modello lo cerca da solo in: cartella dello script, cartella
corrente, Downloads, Desktop, e nelle sottocartelle EcoSortAI/ecosort.

SERVE: flask, pillow, numpy e un runtime TFLite (tensorflow va benissimo).
    pip install flask pillow numpy tensorflow
"""
import argparse, base64, io, json, os, sys, time, webbrowser
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- ricerca file
NOMI_MODELLO = ['rifiuti.tflite', 'rifiuti_float16.tflite', 'rifiuti_float32.tflite',
                'newbest_model.tflite', 'newbest_model.keras']


def cartelle_candidate(extra=None):
    casa = os.path.expanduser('~')
    basi = [os.path.dirname(os.path.abspath(__file__)), os.getcwd(),
            os.path.join(casa, 'Downloads'), os.path.join(casa, 'Desktop'),
            os.path.join(casa, 'Documents')]
    if extra:
        basi.insert(0, extra)
    fuori = []
    for b in basi:
        fuori.append(b)
        for sub in ('EcoSortAI', 'ecosort', 'EcoSort', 'modello'):
            fuori.append(os.path.join(b, sub))
    return [c for c in fuori if os.path.isdir(c)]


def trova(nomi, esplicito=None):
    if esplicito:
        if not os.path.exists(esplicito):
            sys.exit(f"Non trovo {esplicito}")
        return esplicito
    for c in cartelle_candidate():
        for n in nomi:
            p = os.path.join(c, n)
            if os.path.exists(p):
                return p
    return None


# ---------------------------------------------------------------- il modello
class Modello:
    """Incapsula .tflite o .keras dietro la stessa interfaccia."""

    def __init__(self, percorso):
        self.percorso = percorso
        if percorso.endswith('.tflite'):
            Interpreter = self._runtime()
            self.interp = Interpreter(model_path=percorso, num_threads=4)
            self.interp.allocate_tensors()
            self.inp = self.interp.get_input_details()[0]
            self.out = self.interp.get_output_details()[0]
            self.dtype = self.inp['dtype']
            self.size = (int(self.inp['shape'][2]), int(self.inp['shape'][1]))  # (w, h)
            self.tipo = 'tflite'
        else:
            try:
                from tensorflow import keras
            except ImportError:
                sys.exit("Per un modello .keras serve tensorflow:\n  pip install tensorflow")
            self.m = keras.models.load_model(percorso)
            f = self.m.input_shape
            self.size = (int(f[2]), int(f[1]))
            self.dtype = np.float32
            self.tipo = 'keras'

    @staticmethod
    def _runtime():
        for mod, attr in (('ai_edge_litert.interpreter', 'Interpreter'),
                          ('tflite_runtime.interpreter', 'Interpreter')):
            try:
                return getattr(__import__(mod, fromlist=[attr]), attr)
            except ImportError:
                pass
        try:
            import tensorflow as tf
            return tf.lite.Interpreter
        except ImportError:
            sys.exit("Nessun runtime TFLite trovato. Installa:\n"
                     "  pip install tensorflow\n"
                     "(oppure ai-edge-litert, se il tuo sistema ha il wheel)")

    def probabilita(self, img):
        """img = PIL.Image -> array (3,) di probabilita grezze."""
        im = img.convert('RGB').resize(self.size, Image.BILINEAR)
        x = np.asarray(im)[None, ...].astype(self.dtype)
        if self.tipo == 'tflite':
            self.interp.set_tensor(self.inp['index'], x)
            self.interp.invoke()
            return self.interp.get_tensor(self.out['index'])[0].astype(np.float64)
        return self.m.predict(x.astype('float32'), verbose=0)[0].astype(np.float64)


# ---------------------------------------------------------------- la decisione
def applica_temperatura(p, T):
    if T is None or abs(T - 1.0) < 1e-6:
        return p
    z = np.log(np.clip(p, 1e-12, 1.0)) / float(T)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def decidi(p, C):
    """Costo atteso di ogni azione = somma_classi p[c] * C[c][a]. Vince il minimo."""
    atteso = p @ C
    return int(np.argmin(atteso)), atteso


# ---------------------------------------------------------------- pagina web
PAGINA = r"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EcoSort AI — banco di prova</title><style>
*{box-sizing:border-box}
body{margin:0;background:#0A0F0D;color:#E8EFE9;font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1020px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
h1 b{color:#D4FF3A}
.meta{color:#8a968c;font-size:13px;margin-bottom:20px}
.griglia{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.griglia{grid-template-columns:1fr}}
.card{background:#111815;border:1px solid #1e2a24;border-radius:14px;padding:16px}
video,#anteprima{width:100%;border-radius:10px;background:#000;display:block;aspect-ratio:4/3;object-fit:cover}
.row{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
button{background:#D4FF3A;color:#0A0F0D;border:0;border-radius:9px;padding:10px 16px;
  font-weight:650;font-size:14px;cursor:pointer;font-family:inherit}
button.ghost{background:#1e2a24;color:#E8EFE9}
button:disabled{opacity:.45;cursor:default}
#drop{border:2px dashed #2a3a32;border-radius:10px;padding:22px;text-align:center;color:#8a968c;
  cursor:pointer;transition:.15s}
#drop.on{border-color:#D4FF3A;color:#D4FF3A;background:#141c18}
.verdetto{font-size:26px;font-weight:700;letter-spacing:-.02em;margin:2px 0 2px}
.sotto{color:#8a968c;font-size:13px}
.barra{height:26px;border-radius:6px;background:#1a231e;overflow:hidden;margin:5px 0 2px}
.barra i{display:block;height:100%;border-radius:6px}
.lab{display:flex;justify-content:space-between;font-size:13px;color:#b9c4bb}
.pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:600}
.diverso{background:#3a2a12;color:#F59E0B;padding:9px 12px;border-radius:9px;font-size:13px;margin-top:12px}
.uguale{color:#8a968c;font-size:12px;margin-top:12px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
td{padding:3px 0;color:#8a968c}td:last-child{text-align:right;color:#E8EFE9;font-variant-numeric:tabular-nums}
</style></head><body><div class="wrap">
<h1>EcoSort <b>AI</b> — banco di prova</h1>
<div class="meta" id="info">…</div>
<div class="griglia">
  <div class="card">
    <video id="cam" autoplay playsinline muted></video>
    <img id="anteprima" style="display:none">
    <div class="row">
      <button id="scatta">Scatta e classifica</button>
      <button id="ricam" class="ghost" style="display:none">Torna alla webcam</button>
    </div>
    <div class="row"><div id="drop" style="flex:1">…oppure trascina qui una foto (o clicca)</div></div>
    <input type="file" id="file" accept="image/*" hidden>
  </div>
  <div class="card" id="risultato"><div class="sotto">Nessuna immagine ancora.</div></div>
</div></div>
<script>
const COL={carta_e_cartone:'#3B82F6',plastica:'#F59E0B',vetro_e_metallo:'#10B981',indifferenziata:'#9CA3AF'};
const NOME={carta_e_cartone:'Carta e cartone',plastica:'Plastica',vetro_e_metallo:'Vetro e metallo',indifferenziata:'Indifferenziata'};
const BIDONE={carta_e_cartone:'bidone BLU',plastica:'bidone GIALLO',vetro_e_metallo:'bidone VERDE',indifferenziata:'bidone GRIGIO'};
const cam=document.getElementById('cam'),ant=document.getElementById('anteprima');
fetch('/info').then(r=>r.json()).then(d=>{document.getElementById('info').textContent=
  `${d.modello} · ingresso ${d.size[0]}×${d.size[1]} · T = ${d.T.toFixed(3)} · regola di decisione a costo`});
navigator.mediaDevices?.getUserMedia({video:{width:1280,height:960}}).then(s=>cam.srcObject=s)
  .catch(()=>{document.getElementById('scatta').disabled=true;
    document.getElementById('info').textContent+=' · webcam non disponibile, usa il trascinamento'});
function mostraFoto(src){cam.style.display='none';ant.style.display='block';ant.src=src;
  document.getElementById('ricam').style.display='inline-block'}
document.getElementById('ricam').onclick=()=>{ant.style.display='none';cam.style.display='block';
  document.getElementById('ricam').style.display='none'};
document.getElementById('scatta').onclick=()=>{
  const c=document.createElement('canvas');c.width=cam.videoWidth;c.height=cam.videoHeight;
  c.getContext('2d').drawImage(cam,0,0);const d=c.toDataURL('image/jpeg',.92);mostraFoto(d);invia(d)};
const drop=document.getElementById('drop'),file=document.getElementById('file');
drop.onclick=()=>file.click();
file.onchange=e=>{if(e.target.files[0])leggi(e.target.files[0])};
drop.ondragover=e=>{e.preventDefault();drop.classList.add('on')};
drop.ondragleave=()=>drop.classList.remove('on');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('on');if(e.dataTransfer.files[0])leggi(e.dataTransfer.files[0])};
function leggi(f){const r=new FileReader();r.onload=()=>{mostraFoto(r.result);invia(r.result)};r.readAsDataURL(f)}
function invia(dataUrl){
  document.getElementById('risultato').innerHTML='<div class="sotto">classifico…</div>';
  fetch('/classifica',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({img:dataUrl})}).then(r=>r.json()).then(render)}
function render(d){
  if(d.errore){document.getElementById('risultato').innerHTML='<div class="sotto">'+d.errore+'</div>';return}
  const c=COL[d.azione],barre=d.classi.map((k,i)=>
    `<div class="lab"><span>${NOME[k]}</span><span>${(d.probs[i]*100).toFixed(1)}%</span></div>
     <div class="barra"><i style="width:${Math.max(1,d.probs[i]*100)}%;background:${COL[k]}"></i></div>`).join('');
  const nota=d.azione===d.argmax?
    `<div class="uguale">La regola a costo conferma la classe piu probabile.</div>`:
    `<div class="diverso"><b>La regola a costo ha cambiato la decisione.</b><br>
      Classe piu probabile: ${NOME[d.argmax]} (${(d.p_argmax*100).toFixed(1)}%), ma il rischio
      atteso e piu basso mandandolo in ${NOME[d.azione].toLowerCase()}.</div>`;
  document.getElementById('risultato').innerHTML=
   `<span class="pill" style="background:${c}22;color:${c}">${BIDONE[d.azione]}</span>
    <div class="verdetto" style="color:${c}">${NOME[d.azione]}</div>
    <div class="sotto">${d.azione==='indifferenziata'
        ? 'classe piu probabile al '+(d.confidenza*100).toFixed(1)+'%, sotto la soglia di rischio'
        : 'confidenza '+(d.confidenza*100).toFixed(1)+'%'} · ${d.ms.toFixed(0)} ms</div>
    <div style="margin-top:14px">${barre}</div>${nota}
    <table><tr><td>costo atteso della scelta</td><td>${d.costo_scelta.toFixed(3)}</td></tr>
    <tr><td>costo se seguissi solo l'argmax</td><td>${d.costo_argmax.toFixed(3)}</td></tr>
    <tr><td>probabilita prima della calibrazione</td><td>${(d.p_grezza*100).toFixed(1)}%</td></tr></table>`}
</script></body></html>"""


# ---------------------------------------------------------------- modalita batch
def batch(mod, cfg, cartella):
    C = np.array(cfg['matrice_costo'], dtype=np.float64)
    T = float(cfg.get('temperatura', 1.0))
    classi, azioni = cfg['labels'], cfg['azioni']
    est = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    file = sorted(f for f in os.listdir(cartella) if f.lower().endswith(est))
    if not file:
        sys.exit(f"Nessuna immagine in {cartella}")
    print(f"\n{'file':<38}{'decisione':<18}{'conf':>6} {'argmax':<18}{'ms':>6}")
    print('-' * 92)
    cambi = 0
    for f in file:
        t0 = time.perf_counter()
        p0 = mod.probabilita(Image.open(os.path.join(cartella, f)))
        ms = (time.perf_counter() - t0) * 1000
        p = applica_temperatura(p0, T)
        a, _ = decidi(p, C)
        am = int(p.argmax())
        cambi += (azioni[a] != classi[am])
        print(f"{f[:36]:<38}{azioni[a]:<18}{p[a] if a < 3 else p.max():>6.1%} "
              f"{classi[am]:<18}{ms:>6.0f}")
    print('-' * 92)
    print(f"{len(file)} immagini, la regola a costo ha cambiato la decisione {cambi} volte")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modello')
    ap.add_argument('--config')
    ap.add_argument('--cartella', help='modalita batch: classifica tutte le foto e stampa una tabella')
    ap.add_argument('--porta', type=int, default=7860)
    a = ap.parse_args()

    percorso = trova(NOMI_MODELLO, a.modello)
    if not percorso:
        sys.exit("Non trovo nessun modello. Passa il percorso:\n"
                 "  python prova_pc.py --modello C:\\percorso\\rifiuti.tflite")
    cfg_path = trova(['config.json'], a.config) or os.path.join(os.path.dirname(percorso), 'config.json')
    if not os.path.exists(cfg_path):
        sys.exit(f"Non trovo config.json (cercato anche in {os.path.dirname(percorso)})")

    cfg = json.load(open(cfg_path, encoding='utf-8'))
    mod = Modello(percorso)
    T = float(cfg.get('temperatura', 1.0))
    C = np.array(cfg['matrice_costo'], dtype=np.float64)
    print(f"Modello : {percorso}")
    print(f"Config  : {cfg_path}")
    print(f"Ingresso: {mod.size[0]}x{mod.size[1]}  dtype {np.dtype(mod.dtype).name}  "
          f"quantizzazione {cfg.get('quantizzazione','?')}")
    print(f"Temperatura {T:.3f} — regola di decisione a costo attiva\n")

    if a.cartella:
        batch(mod, cfg, a.cartella)
        return

    try:
        from flask import Flask, request, jsonify
    except ImportError:
        sys.exit("Serve flask per la pagina web:\n  pip install flask\n"
                 "(oppure usa --cartella per la modalita senza browser)")

    app = Flask(__name__)
    log = __import__('logging').getLogger('werkzeug')
    log.setLevel(__import__('logging').ERROR)

    @app.get('/')
    def home():
        return PAGINA

    @app.get('/info')
    def info():
        return jsonify(modello=os.path.basename(percorso), size=list(mod.size), T=T)

    @app.post('/classifica')
    def classifica():
        try:
            dati = request.json['img'].split(',', 1)[1]
            img = Image.open(io.BytesIO(base64.b64decode(dati)))
            t0 = time.perf_counter()
            p0 = mod.probabilita(img)
            ms = (time.perf_counter() - t0) * 1000
            p = applica_temperatura(p0, T)
            az, attesi = decidi(p, C)
            am = int(p.argmax())
            return jsonify(
                classi=cfg['labels'], probs=[float(v) for v in p],
                azione=cfg['azioni'][az], argmax=cfg['labels'][am],
                p_argmax=float(p[am]), p_grezza=float(p0[am]),
                confidenza=float(p[az]) if az < 3 else float(p.max()),
                costo_scelta=float(attesi[az]), costo_argmax=float(attesi[am]), ms=ms)
        except Exception as e:
            return jsonify(errore=f"{type(e).__name__}: {e}")

    url = f"http://127.0.0.1:{a.porta}"
    print(f"Apri {url}  (Ctrl+C per chiudere)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    app.run(host='127.0.0.1', port=a.porta, debug=False)


if __name__ == '__main__':
    main()
