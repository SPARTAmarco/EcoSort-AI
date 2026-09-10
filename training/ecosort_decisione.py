"""
EcoSort AI — Matrice di costo e regola di decisione bayesiana.

Modulo condiviso tra:
  - il benchmark su Colab (selezione del modello)
  - lo script di inferenza sul Raspberry Pi

L'idea centrale: il sistema non deve massimizzare l'accuratezza, deve
MINIMIZZARE IL DANNO all'impianto di riciclo. Sono due obiettivi diversi.
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


def azione_ottima(probs, costo=COSTO):
    """
    Regola di decisione bayesiana: sceglie l'azione che minimizza il COSTO
    ATTESO, non quella piu probabile.

        azione* = argmin_a  SUM_c  P(c|x) * Costo[c][a]

    Sostituisce sia argmax sia la soglia fissa del 70%: l'opzione
    "indifferenziata" e' una delle azioni possibili, quindi la soglia emerge
    da sola dalla matrice invece di essere un numero scelto a occhio.

    probs: array (n, 3) di probabilita softmax
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
