# Matrice di costo degli errori e regola di decisione

Il sistema non ottimizza l'accuratezza: minimizza il **danno all'impianto di riciclo**.

## Matrice di costo (scala 0-10)

Righe = rifiuto reale, colonne = bidone di destinazione.

| reale \ destinazione | carta (BLU) | plastica (GIALLO) | vetro (VERDE) | indifferenziata (GRIGIO) |
|---|---|---|---|---|
| carta_e_cartone | 0 | 3.0 | 2.0 | 1.0 |
| plastica | 4.0 | 0 | 5.0 | 1.0 |
| vetro_e_metallo | **9.0** | **8.0** | 0 | 1.5 |

## Giustificazione dei valori

- **carta → vetro (2.0)** — in fornace a 1500 °C la carta brucia e sparisce. Il più innocuo degli errori incrociati.
- **carta → plastica (3.0)** — scarto nel flusso plastica; i separatori ottici la individuano ma abbassa la resa.
- **plastica → carta (4.0)** — nel pulper la plastica fusa crea grumi che degradano la fibra.
- **plastica → vetro (5.0)** — bruciando lascia residui carboniosi: bolle e difetti nel vetro colato.
- **vetro → plastica (8.0)** — il vetro è abrasivo: distrugge lame dei mulini e viti degli estrusori.
- **vetro → carta (9.0)** — il peggiore: frammenti taglienti nel macero, pericolosi per gli operatori e irrecuperabili una volta ridotti in schegge.
- **qualsiasi → indifferenziata (1.0–1.5)** — nessun danno agli impianti, solo materiale perso. Non può valere 0, altrimenti la regola ottimale diventa "manda tutto in grigio". Per il vetro vale 1.5 perché è riciclabile all'infinito.

**Asimmetria chiave:** carta→vetro costa 2.0, vetro→carta costa 9.0. Un fattore 4,5×.
Un modello ottimizzato sull'accuratezza tratta i due errori come identici.

## Regola di decisione bayesiana

Sostituisce sia `argmax` sia la soglia fissa:

```
azione* = argmin_a  Σ_c  P(c|x) · Costo[c][a]
```

L'opzione "indifferenziata" è una delle 4 azioni, quindi la soglia **emerge dalla matrice**
invece di essere scelta a occhio.

### Soglie implicite risultanti

| classe vincente | rivale | confidenza minima per agire |
|---|---|---|
| vetro_e_metallo | carta_e_cartone | 40,0 % |
| plastica | carta_e_cartone | 66,7 % |
| vetro_e_metallo | plastica | 72,7 % |
| carta_e_cartone | plastica | 75,0 % |
| plastica | vetro_e_metallo | 86,7 % |
| carta_e_cartone | vetro_e_metallo | 88,2 % |

Range 40 %–88 % contro il 70 % fisso: la soglia unica è una semplificazione grossolana.

## Pesi di training (bilanciamento × costo)

- carta_e_cartone: 0,484
- plastica: 0,871
- vetro_e_metallo: 1,645

## Validazione su dati sintetici (modello al 91,2 %)

| regola | costo medio | coverage | errori gravi (vetro nel bidone sbagliato) |
|---|---|---|---|
| argmax | 0,4685 | 100 % | 191 |
| soglia 0,70 | 0,3622 | 81,9 % | — |
| decisione a costo | 0,3942 | 73,7 % | **26** |

Errori gravi ridotti dell'86 %.
