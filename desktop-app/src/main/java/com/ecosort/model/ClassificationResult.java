package com.ecosort.model;

public class ClassificationResult {

    public enum Categoria {
        PLASTICA        ("Plastica",         "#F59E0B", "Bidone GIALLO"),
        CARTA_E_CARTONE ("Carta e Cartone",  "#3B82F6", "Bidone BLU"),
        VETRO_E_METALLO ("Vetro e Metallo",  "#10B981", "Bidone VERDE/GIALLO"),
        INDIFFERENZIATA ("Indifferenziata",  "#9CA3AF", "Bidone GRIGIO"),
        ERRORE          ("Errore",           "#EF4444", "—");

        public final String label;
        public final String hex;
        public final String bidone;

        Categoria(String label, String hex, String bidone) {
            this.label  = label;
            this.hex    = hex;
            this.bidone = bidone;
        }

        public static Categoria fromString(String s) {
            if (s == null) return INDIFFERENZIATA;
            return switch (s.toLowerCase().trim()) {
                case "plastica"         -> PLASTICA;
                case "carta_e_cartone"  -> CARTA_E_CARTONE;
                case "vetro_e_metallo"  -> VETRO_E_METALLO;
                default                 -> INDIFFERENZIATA;
            };
        }
    }

    public enum Modalita { ONLINE_GEMINI, OFFLINE_TFLITE }

    private final Categoria categoria;
    private final double    confidenza;
    private final String    motivo;
    private final Modalita  modalita;
    private final long      tempoMs;
    private final boolean   sottoSoglia;
    private final String    motore;      // es. "Keras · TensorFlow", "TFLite · LiteRT"

    public ClassificationResult(Categoria categoria, double confidenza,
                                String motivo, Modalita modalita,
                                long tempoMs, boolean sottoSoglia) {
        this(categoria, confidenza, motivo, modalita, tempoMs, sottoSoglia, null);
    }

    public ClassificationResult(Categoria categoria, double confidenza,
                                String motivo, Modalita modalita,
                                long tempoMs, boolean sottoSoglia, String motore) {
        this.motore     = motore;
        this.categoria  = categoria;
        this.confidenza = confidenza;
        this.motivo     = motivo;
        this.modalita   = modalita;
        this.tempoMs    = tempoMs;
        this.sottoSoglia = sottoSoglia;
    }

    public Categoria getCategoria()   { return categoria; }
    public double    getConfidenza()  { return confidenza; }
    public String    getMotivo()      { return motivo; }
    public Modalita  getModalita()    { return modalita; }
    public long      getTempoMs()     { return tempoMs; }
    public boolean   isSottoSoglia()  { return sottoSoglia; }
    public String    getMotore()      { return motore; }
    public int       getConfPct()     { return (int) Math.round(confidenza * 100); }
}