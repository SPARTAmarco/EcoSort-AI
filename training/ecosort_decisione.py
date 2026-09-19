"""
EcoSort AI — Matrice di costo, calibrazione e regola di decisione bayesiana.

Modulo condiviso tra:
  - il benchmark su Colab (selezione del modello)
  - lo script di inferenza sul Raspberry Pi

L'idea centrale: il sistema non deve massimizzare l'accuratezza, deve
MINIMIZZARE IL DANNO all'impianto di riciclo. Sono due obiettivi diversi.

NOVITA v2: temperature scaling. La regola bayesiana si regge sulle
probabilita, e una rete addestrata con cross-entropy e' quasi sempre
SOVRA-CONFIDENTE: dice 0.97 quando ha ragione l'88% delle volte. Con
probabilita gonfiate la regola agisce troppo spesso invece di mandare in
indifferenziata, e il costo peggiora senza che l'accuratezza cambi di un
punto. Un solo parametro T stimato sulla validation lo corregge.
"""
import numpy as np

CLASSI = ['carta_e_cartone', 'plastica', 'vetro_e_metallo']
AZIONI = ['carta_e_cartone', 'plastica', 'vetro_e_metallo', 'indifferenziata']
BIDONI = {0: 'BLU', 1: 'GIALLO', 2: 'VERDE', 3: 'GRIGIO'}

# ============================================================================
# MATRICE DI COSTO   righe = rifiuto REALE, colonne = bidone di DESTINAZIONE
# Scala 0-10, dove 10 = danno massimo al processo di riciclo.
# ============================================================================
#
# GIUSTIFICAZIONE DI OGNI VALORE (serve alla giuria, non solo al codice):
#
#  carta -> plastica  = 3.0  Scarto nel flusso plastica. I separatori ottici
#                            la individuano, ma abbassa la resa dell'impianto.
#  carta -> vetro     = 2.0  In fornace a 1500 gradi la carta brucia e sparisce.
#                            E' il piu innocuo degli errori incrociati.
#  plastica -> carta  = 4.0  Contamina il macero: la plastica fusa nel pulper
#                            crea grumi che degradano la qualita della fibra.
#  plastica -> vetro  = 5.0  Bruciando lascia residui carboniosi che formano
#                            bolle e difetti nel vetro colato.
#  vetro -> carta     = 9.0  IL PEGGIORE. Frammenti di vetro nel macero: taglienti
#                            per gli operatori, abrasivi per gli impianti, e le
#                            schegge fini sono praticamente irrecuperabili.
#  vetro -> plastica  = 8.0  Il vetro e' abrasivo: distrugge lame dei mulini e
#                            viti degli estrusori. Danno economico diretto.
#
#  qualsiasi -> indifferenziata = 1.0-1.5
#                            Nessun danno agli impianti: solo materiale perso e
#                            costo di smaltimento. E' il senso stesso del fallback.
#                            NON puo' valere 0, altrimenti la regola ottimale
#                            diventa "manda tutto in grigio" e il sistema e' inutile.
#                            Per il vetro vale 1.5 perche' e' riciclabile
#                            all'infinito: buttarlo e' uno spreco maggiore.
#
COSTO = np.array([
    # -> carta  -> plastica  -> vetro  -> indifferenziata
    [    0.0,       3.0,        2.0,        1.0  ],   # reale: carta_e_cartone
    [    4.0,       0.0,        5.0,        1.0  ],   # reale: plastica
    [    9.0,       8.0,        0.0,        1.5  ],   # reale: vetro_e_metallo
], dtype=np.float64)


# ============================================================================
# CALIBRAZIONE (temperature scaling)
# ============================================================================
def applica_temperatura(probs, T):
    """
    Ricalibra le probabilita con un solo parametro.

    Il modello esporta gia il softmax, non i logit. Non e' un problema:
        log(p) = logit - logsumexp(logit)
    quindi softmax(log(p)/T) == softmax(logit/T), perche' la costante sparisce
    nel softmax. Si puo' quindi calibrare a valle del modello, senza toccare
    il grafo e senza rischi in fase di conversione TFLite.

    T > 1  ammorbidisce (il modello era troppo sicuro)   <- caso tipico
    T < 1  irrigidisce
    T = 1  nessuna modifica
    """
    if T is None or abs(T - 1.0) < 1e-6:
        return np.asarray(probs, dtype=np.float64)
    z = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)) / float(T)
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(probs, y):
    """Negative log-likelihood: la metrica che la temperatura minimizza."""
    p = np.asarray(probs, dtype=np.float64)[np.arange(len(y)), np.asarray(y)]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def stima_temperatura(probs, y, lo=0.25, hi=5.0, passi=60, raffinamenti=3):
    """
    Cerca la T che minimizza la NLL sulla VALIDATION (mai sul test).
    Ricerca a griglia con raffinamento locale: nessuna dipendenza da scipy,
    e la funzione e' convessa in log(T), quindi la griglia basta e avanza.
    """
    probs, y = np.asarray(probs, dtype=np.float64), np.asarray(y)
    best_T, best_v = 1.0, nll(probs, y)
    for _ in range(raffinamenti):
        griglia = np.linspace(lo, hi, passi)
        for T in griglia:
            v = nll(applica_temperatura(probs, T), y)
            if v < best_v:
                best_T, best_v = float(T), v
        passo = (hi - lo) / passi
        lo, hi = max(0.05, best_T - passo * 2), best_T + passo * 2
    return best_T, best_v


def errore_calibrazione(probs, y, bins=15):
    """
    ECE — Expected Calibration Error. Divide le predizioni in fasce di
    confidenza e misura quanto la confidenza media si scosta dall'accuratezza
    reale in ogni fascia. 0 = perfettamente calibrato.
    Serve a dimostrare (numeri alla mano) che la calibrazione ha funzionato.
    """
    probs, y = np.asarray(probs, dtype=np.float64), np.asarray(y)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    corretti = (pred == y).astype(np.float64)
    bordi = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf > bordi[i]) & (conf <= bordi[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(corretti[m].mean() - conf[m].mean())
    return float(ece)


# ============================================================================
# REGOLA DI DECISIONE
# ============================================================================
def azione_ottima(probs, costo=COSTO):
    """
    Regola di decisione bayesiana: sceglie l'azione che minimizza il COSTO
    ATTESO, non quella piu probabile.

        azione* = argmin_a  SUM_c  P(c|x) * Costo[c][a]

    Sostituisce sia argmax sia la soglia fissa del 70%: l'opzione
    "indifferenziata" e' una delle azioni possibili, quindi la soglia emerge
    da sola dalla matrice invece di essere un numero scelto a occhio.

    probs: array (n, 3) di probabilita softmax (meglio se gia calibrate)
    ritorna: (azioni (n,), costi_attesi (n, 4))
    """
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    costi_attesi = probs @ costo          # (n,3) @ (3,4) -> (n,4)
    return costi_attesi.argmin(axis=1), costi_attesi


def costo_medio(y_true, azioni, costo=COSTO):
    """Costo medio per rifiuto: LA metrica di selezione del modello."""
    y_true = np.asarray(y_true)
    azioni = np.asarray(azioni)
    return float(costo[y_true, azioni].mean())


def costo_medio_argmax(y_true, probs, costo=COSTO):
    """Costo se si usasse il semplice argmax: serve per il confronto."""
    return costo_medio(y_true, np.asarray(probs).argmax(axis=1), costo)


def costo_medio_soglia(y_true, probs, soglia=0.70, costo=COSTO):
    """Costo con la vecchia regola a soglia fissa: serve per il confronto."""
    probs = np.asarray(probs)
    pred = probs.argmax(axis=1)
    azioni = np.where(probs.max(axis=1) >= soglia, pred, 3)
    return costo_medio(y_true, azioni, costo)


def soglia_migliore(y_true, probs, costo=COSTO):
    """
    Cerca la soglia fissa ottimale a posteriori. Non serve per il deployment:
    serve come termine di confronto. Attenzione: e' tarata sugli stessi dati
    su cui viene misurata, quindi il suo costo e' una stima ottimistica. Nel
    benchmark reale e' risultata di poco migliore sul costo medio, ma la regola
    a costo non ha parametri tarati sui dati e dimezza gli errori gravi,
    perche' una soglia unica non puo' distinguere il vetro dalla carta.
    """
    probs = np.asarray(probs)
    migliore = (0.0, float('inf'))
    for s in np.linspace(0.0, 0.99, 100):
        c = costo_medio_soglia(y_true, probs, float(s), costo)
        if c < migliore[1]:
            migliore = (float(s), c)
    return migliore


def gravi(y_true, azioni):
    """Errori gravi: vetro_e_metallo finito nel bidone blu o giallo."""
    return int(((np.asarray(y_true) == 2) & np.isin(np.asarray(azioni), [0, 1])).sum())


def soglie_implicite(costo=COSTO):
    """
    Per ogni coppia (classe vincente, classe rivale) calcola la probabilita
    minima oltre la quale conviene agire invece di mandare in indifferenziata.

    Con probs = p sulla classe c e (1-p) sul rivale r:
        costo(azione c)      = (1-p) * Costo[r][c]
        costo(indifferenziata) = p * Costo[c][3] + (1-p) * Costo[r][3]
    Uguagliando:   p* = A / (A + B),  A = Costo[r][c] - Costo[r][3],  B = Costo[c][3]

    Mostra che una soglia unica al 70% e' una semplificazione grossolana:
    il vetro va accettato con meno confidenza della carta, perche' mandare
    vetro in grigio spreca, mentre mandare vetro nel blu e' un disastro.
    """
    righe = []
    for c in range(3):
        for r in range(3):
            if c == r:
                continue
            A = costo[r][c] - costo[r][3]
            B = costo[c][3]
            p = A / (A + B) if (A + B) > 0 else 0.0
            righe.append((CLASSI[c], CLASSI[r], float(np.clip(p, 0.0, 1.0))))
    return righe


def pesi_training(costo=COSTO, class_weight_bilanciato=None):
    """
    Pesi per il training: quanto costa MEDIAMENTE sbagliare ogni classe reale.
    Il vetro pesa di piu, quindi il modello impara a non sbagliarlo.
    Opzionalmente moltiplicati per i pesi 'balanced' di sklearn.
    """
    costo_medio_riga = np.array([
        costo[c, [a for a in range(3) if a != c]].mean() for c in range(3)
    ])
    pesi = costo_medio_riga / costo_medio_riga.mean()
    if class_weight_bilanciato is not None:
        bil = np.array([class_weight_bilanciato[i] for i in range(3)])
        pesi = pesi * bil
        pesi = pesi / pesi.mean()
    return pesi


def riepilogo(y_true, probs_grezze, T=None, etichetta=''):
    """
    Confronto compatto fra le quattro regole possibili, con e senza
    calibrazione. E' la tabella da mostrare alla giuria.
    """
    y_true = np.asarray(y_true)
    p_cal = applica_temperatura(probs_grezze, T) if T else np.asarray(probs_grezze)
    az, _ = azione_ottima(p_cal)
    s_best, c_best = soglia_migliore(y_true, p_cal)
    r = dict(
        etichetta=etichetta,
        accuracy=float((p_cal.argmax(1) == y_true).mean()),
        temperatura=float(T) if T else 1.0,
        ece_prima=errore_calibrazione(probs_grezze, y_true),
        ece_dopo=errore_calibrazione(p_cal, y_true),
        costo_argmax=costo_medio_argmax(y_true, p_cal),
        costo_soglia70=costo_medio_soglia(y_true, p_cal, 0.70),
        costo_soglia_ottima=c_best,
        soglia_ottima=s_best,
        costo_decisione=costo_medio(y_true, az),
        coverage=float((az != 3).mean()),
        gravi_argmax=gravi(y_true, p_cal.argmax(1)),
        gravi_decisione=gravi(y_true, az),
    )
    return r


def stampa_matrice(costo=COSTO):
    print("MATRICE DI COSTO (riga = reale, colonna = destinazione)")
    print(f"{'':>18}" + "".join(f"{a[:12]:>14}" for a in AZIONI))
    for c in range(3):
        print(f"{CLASSI[c]:>18}" + "".join(f"{costo[c][a]:>14.1f}" for a in range(4)))


if __name__ == '__main__':
    stampa_matrice()
    print("\nPESI DI TRAINING (costo medio di sbagliare ogni classe):")
    for c, p in zip(CLASSI, pesi_training()):
        print(f"  {c:>18} : {p:.3f}")
    print("\nSOGLIE IMPLICITE (p minima per agire invece di mandare in grigio):")
    for c, r, p in soglie_implicite():
        print(f"  {c:>18}  vs {r:<18} : {p*100:5.1f}%")

    # dimostrazione del temperature scaling su probabilita sintetiche
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logit = np.eye(3)[y] * 1.6 + rng.normal(0, 1.0, (2000, 3))
    p = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
    p = p ** 2.0                      # rete deliberatamente sovra-confidente
    p /= p.sum(1, keepdims=True)
    T, _ = stima_temperatura(p, y)
    print(f"\nESEMPIO — temperatura stimata: {T:.3f}")
    print(f"  ECE prima {errore_calibrazione(p, y):.4f} -> dopo "
          f"{errore_calibrazione(applica_temperatura(p, T), y):.4f}")
    print(f"  costo decisione prima {costo_medio(y, azione_ottima(p)[0]):.4f} -> dopo "
          f"{costo_medio(y, azione_ottima(applica_temperatura(p, T))[0]):.4f}")
