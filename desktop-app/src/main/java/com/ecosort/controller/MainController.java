package com.ecosort.controller;

import com.ecosort.model.ClassificationResult;
import com.ecosort.model.ClassificationResult.Modalita;
import com.ecosort.service.GeminiService;
import com.ecosort.service.LocalAIService;
import javafx.animation.*;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.shape.Circle;
import javafx.stage.FileChooser;
import javafx.util.Duration;
import com.github.sarxos.webcam.Webcam;

import java.io.File;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;
import java.util.prefs.Preferences;

public class MainController implements Initializable {

    // ── Top bar ──────────────────────────────────────────────────
    @FXML
    private Button btnOffline;
    @FXML
    private Button btnOnline;

    // ── Drop zone ────────────────────────────────────────────────
    @FXML
    private StackPane dropZone;
    @FXML
    private VBox dropPlaceholder;
    @FXML
    private ImageView previewImage;
    @FXML
    private Label labelNomeFile;

    // ── Camera UI ────────────────────────────────────────────────
    @FXML
    private VBox cameraContainer;
    @FXML
    private ImageView cameraFeed;
    @FXML
    private Button btnScatta;
    @FXML
    private Button btnAnnullaCamera;

    // ── Bottoni ──────────────────────────────────────────────────
    @FXML
    private Button btnAnalizza;
    @FXML
    private Button btnCamera;
    @FXML
    private Button btnPulisci;

    // ── Settings panel ───────────────────────────────────────────
    @FXML
    private VBox rowApiKey;
    @FXML
    private TextField fieldApiKey;
    @FXML
    private Label labelKeyStatus;
    @FXML
    private VBox rowModello;

    // ── Result panel ─────────────────────────────────────────────
    @FXML
    private VBox panelRisultato;
    @FXML
    private Label labelCategoria;
    @FXML
    private Label labelConfidenzaPct;
    @FXML
    private ProgressBar barraConfidenza;
    @FXML
    private Label labelTempo;
    @FXML
    private Label labelModalitaUsata;
    @FXML
    private VBox rowMotivo;
    @FXML
    private Label labelMotivo;
    @FXML
    private Rectangle colorBar;

    // ── Loading overlay ──────────────────────────────────────────
    @FXML
    private StackPane loadingOverlay;
    @FXML
    private Label labelLoading;
    @FXML
    private ProgressBar loadingProgressBar;
    @FXML
    private Circle loadingSpinner;

    // ── Status bar ───────────────────────────────────────────────
    @FXML
    private Label statusBar;

    // ── State ────────────────────────────────────────────────────
    private File fileSelezionato;
    private boolean modalitaOnline = false;
    private LocalAIService localAI;
    private GeminiService geminiService;
    private Webcam webcam;
    private volatile boolean cameraAttiva = false;
    private Timeline loadingTimeline;
    private RotateTransition spinnerTransition;
    private boolean serverPronto = false;
    private String erroreServer = null;

    private static final String PREF_API_KEY = "gemini_api_key";
    private static final String PREF_MODALITA = "modalita_online";
    private final Preferences prefs = Preferences.userNodeForPackage(MainController.class);

    @Override
    public void initialize(URL url, ResourceBundle rb) {
        localAI = new LocalAIService();

        // Forza la colorBar a riempire la larghezza del pannello dei risultati in modo
        // dinamico e sicuro
        if (colorBar != null && panelRisultato != null) {
            colorBar.widthProperty().bind(panelRisultato.widthProperty());
        }

        // Ripristina preferenze salvate
        if (fieldApiKey != null) {
            String initialKey = prefs.get(PREF_API_KEY, "");
            if (initialKey.isEmpty()) {
                try {
                    File keyFile = new File("gemini_key.txt");
                    if (keyFile.exists()) {
                        initialKey = java.nio.file.Files.readString(keyFile.toPath()).trim();
                    }
                } catch (Exception ignored) {
                }
            }
            fieldApiKey.setText(initialKey);

            // Aggiorna stato key ogni volta che cambia
            fieldApiKey.textProperty().addListener((obs, oldVal, newVal) -> {
                prefs.put(PREF_API_KEY, newVal.trim());
                aggiornaLabelKey(newVal.trim());
                geminiService = null; // Reinizializza con la nuova chiave alla prossima chiamata
            });
        }

        // Forza sempre la modalità OFFLINE come predefinita all'avvio dell'applicazione
        modalitaOnline = false;
        prefs.putBoolean(PREF_MODALITA, false);

        aggiornaUiModo();

        // Avvia il server locale in modalità offline
        if (!modalitaOnline) {
            avviaServerBackground();
        } else {
            setStatus("Pronto — trascina un'immagine o clicca nell'area di caricamento.");
        }
    }

    // ── Toggle modo ──────────────────────────────────────────────

    @FXML
    private void onModoOffline() {
        if (!modalitaOnline)
            return;
        modalitaOnline = false;
        prefs.putBoolean(PREF_MODALITA, false);
        aggiornaUiModo();
        avviaServerBackground();
    }

    @FXML
    private void onModoOnline() {
        if (modalitaOnline)
            return;
        modalitaOnline = true;
        prefs.putBoolean(PREF_MODALITA, true);
        aggiornaUiModo();
        String key = getApiKey();
        if (key.isEmpty()) {
            setStatus("Modalità Online — inserisci la tua API key Gemini.");
        } else {
            setStatus("Modalità Online — API key configurata.");
        }
    }

    private void aggiornaUiModo() {
        if (modalitaOnline) {
            if (btnOnline != null)
                btnOnline.getStyleClass().setAll("button", "modo-btn-active");
            if (btnOffline != null)
                btnOffline.getStyleClass().setAll("button", "modo-btn-inactive");
            if (rowApiKey != null) {
                rowApiKey.setVisible(true);
                rowApiKey.setManaged(true);
            }
            if (rowModello != null) {
                rowModello.setVisible(false);
                rowModello.setManaged(false);
            }
            aggiornaLabelKey(getApiKey());
        } else {
            if (btnOffline != null)
                btnOffline.getStyleClass().setAll("button", "modo-btn-active");
            if (btnOnline != null)
                btnOnline.getStyleClass().setAll("button", "modo-btn-inactive");
            if (rowApiKey != null) {
                rowApiKey.setVisible(false);
                rowApiKey.setManaged(false);
            }
            if (rowModello != null) {
                rowModello.setVisible(true);
                rowModello.setManaged(true);
            }
        }
    }

    private void aggiornaLabelKey(String key) {
        if (labelKeyStatus == null)
            return;
        if (key.isEmpty()) {
            labelKeyStatus.setText("✗ Mancante");
            labelKeyStatus.getStyleClass().setAll("label", "key-missing");
        } else {
            labelKeyStatus.setText("✓ Salvata");
            labelKeyStatus.getStyleClass().setAll("label", "key-ok");
        }
    }

    // ── Avvio server Python ──────────────────────────────────────

    private void avviaServerBackground() {
        serverPronto = false;
        erroreServer = null;
        setStatus("⏳ Avvio server per analisi offline...");
        Task<Void> task = new Task<>() {
            @Override
            protected Void call() throws Exception {
                localAI.avviaServer();
                localAI.attendiPronto(30000);
                return null;
            }
        };
        task.setOnSucceeded(e -> {
            serverPronto = true;
            erroreServer = null;
            setStatus("✅ Server pronto — carica un'immagine.");
        });
        task.setOnFailed(e -> {
            serverPronto = false;
            erroreServer = task.getException() != null
                    ? task.getException().getMessage()
                    : "Errore sconosciuto";
            setStatus("⚠️ Server non disponibile: " + erroreServer);
        });
        Thread t = new Thread(task);
        t.setDaemon(true);
        t.start();
    }

    // ── Drag & Drop ──────────────────────────────────────────────

    @FXML
    private void onDragOver(DragEvent e) {
        if (e.getDragboard().hasFiles()) {
            List<File> files = e.getDragboard().getFiles();
            if (!files.isEmpty() && isImmagine(files.get(0))) {
                e.acceptTransferModes(TransferMode.COPY);
                dropZone.getStyleClass().add("drop-zone-hover");
            }
        }
        e.consume();
    }

    @FXML
    private void onDragExited(DragEvent e) {
        dropZone.getStyleClass().remove("drop-zone-hover");
        e.consume();
    }

    @FXML
    private void onDragDropped(DragEvent e) {
        Dragboard db = e.getDragboard();
        if (db.hasFiles()) {
            List<File> files = db.getFiles();
            if (!files.isEmpty() && isImmagine(files.get(0))) {
                caricaImmagine(files.get(0));
                e.setDropCompleted(true);
            } else {
                setStatus("⚠️ Formato non supportato. Usa JPG, PNG, WEBP o BMP.");
            }
        }
        dropZone.getStyleClass().remove("drop-zone-hover");
        e.consume();
    }

    @FXML
    private void onDropZoneClick() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Seleziona immagine");

        // Imposta come cartella predefinita "Downloads"
        String userHome = System.getProperty("user.home");
        if (userHome != null) {
            File downloadsDir = new File(userHome, "Downloads");
            if (downloadsDir.exists() && downloadsDir.isDirectory()) {
                fc.setInitialDirectory(downloadsDir);
            }
        }

        fc.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Immagini",
                        "*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.gif"),
                new FileChooser.ExtensionFilter("Tutti i file", "*.*"));
        File f = fc.showOpenDialog(dropZone.getScene().getWindow());
        if (f != null)
            caricaImmagine(f);
    }

    private void caricaImmagine(File f) {
        fileSelezionato = f;
        try {
            Image img = new Image(f.toURI().toString(), 460, 340, true, true);
            previewImage.setImage(img);
            previewImage.setVisible(true);
            dropPlaceholder.setVisible(false);
            dropPlaceholder.setManaged(false);
            labelNomeFile.setText(f.getName());
            btnAnalizza.setDisable(false);
            setStatus("Immagine caricata: " + f.getName());
        } catch (Exception ex) {
            setStatus("❌ Errore caricamento immagine: " + ex.getMessage());
        }
    }

    // ── Analizza ─────────────────────────────────────────────────

    @FXML
    private void onAnalizza() {
        if (fileSelezionato == null) {
            setStatus("⚠️ Nessuna immagine selezionata.");
            return;
        }

        if (modalitaOnline) {
            String apiKey = getApiKey();
            if (apiKey.isEmpty()) {
                setStatus("⚠️ Inserisci la API key Gemini nelle impostazioni.");
                if (fieldApiKey != null) {
                    fieldApiKey.requestFocus();
                }
                return;
            }
        } else {
            if (!serverPronto) {
                setStatus("⚠️ Il server locale non è pronto. Attendi il caricamento.");
                return;
            }
        }

        resetRisultato();
        mostraLoading(true, "Analisi in corso...");
        btnAnalizza.setDisable(true);

        // Cattura riferimento al file PRIMA di lanciare il thread
        final File fileDaAnalizzare = fileSelezionato;

        Task<ClassificationResult> task = new Task<>() {
            @Override
            protected ClassificationResult call() throws Exception {
                if (modalitaOnline) {
                    if (geminiService == null) {
                        geminiService = new GeminiService(getApiKey());
                    }
                    return geminiService.classifica(fileDaAnalizzare);
                } else {
                    return localAI.classifica(fileDaAnalizzare);
                }
            }
        };

        task.setOnSucceeded(e -> {
            mostraLoading(false, null);
            btnAnalizza.setDisable(false);
            mostraRisultato(task.getValue());
        });
        task.setOnFailed(e -> {
            mostraLoading(false, null);
            btnAnalizza.setDisable(false);
            String msg = task.getException() != null
                    ? task.getException().getMessage()
                    : "Errore sconosciuto";
            setStatus("❌ " + msg);
            mostraErroreRisultato(msg);
        });

        Thread t = new Thread(task);
        t.setDaemon(true);
        t.start();
    }

    // ── Risultato ────────────────────────────────────────────────

    private void mostraRisultato(ClassificationResult r) {
        ClassificationResult.Categoria cat = r.getCategoria();
        int pct = r.getConfPct();

        labelCategoria.setText(cat.label);
        // Percentuale softmax reale (mai valori inventati)
        labelConfidenzaPct.setText(pct > 0 ? pct + "%" : "\u2014");
        labelTempo.setText(r.getTempoMs() + " ms");
        labelModalitaUsata.setText(
                r.getModalita() == Modalita.ONLINE_GEMINI ? "Gemini 2.5 Flash" : "TFLite locale");

        try {
            colorBar.setFill(Color.web(cat.hex));
            labelCategoria.setStyle("-fx-text-fill: " + cat.hex + ";");
        } catch (Exception ignored) {
        }

        animaBarraConfidenza(r.getConfidenza());

        if (r.isSottoSoglia()) {
            // Confidenza troppo bassa: mostra spiegazione onesta
            labelMotivo.setText(
                    "\u26a0\ufe0f  Confidenza insufficiente (" + pct + "%) \u2014 l\u0027immagine non corrisponde "
                            + "chiaramente ad alcuna categoria. "
                            + "Prova con una foto pi\u00f9 nitida o inquadra meglio il rifiuto.");
            rowMotivo.setVisible(true);
            rowMotivo.setManaged(true);
        } else {
            // Elimina la parte analisi AI per lasciare il pannello pulito e compatto
            rowMotivo.setVisible(false);
            rowMotivo.setManaged(false);
        }

        panelRisultato.setVisible(true);
        panelRisultato.setManaged(true);

        String statusMsg = r.isSottoSoglia()
                ? "\u26a0\ufe0f Confidenza bassa (" + pct + "%) \u2014 classificato come Indifferenziata"
                : "\u2705 " + cat.label + " \u2014 " + pct + "% in " + r.getTempoMs() + " ms";
        setStatus(statusMsg);
    }

    private void mostraErroreRisultato(String msg) {
        labelCategoria.setText("Errore");
        labelCategoria.setStyle("-fx-text-fill: #EF4444;");
        labelConfidenzaPct.setText("—");
        labelTempo.setText("—");
        labelModalitaUsata.setText("—");
        labelMotivo.setText(msg);
        rowMotivo.setVisible(true);
        rowMotivo.setManaged(true);
        colorBar.setFill(Color.web("#EF4444"));
        barraConfidenza.setProgress(0);
        panelRisultato.setVisible(true);
        panelRisultato.setManaged(true);
    }

    private void resetRisultato() {
        panelRisultato.setVisible(false);
        panelRisultato.setManaged(false);
        barraConfidenza.setProgress(0);
        labelCategoria.setText("—");
        labelCategoria.setStyle("");
        labelConfidenzaPct.setText("—");
        labelTempo.setText("—");
        labelModalitaUsata.setText("—");
        rowMotivo.setVisible(false);
        rowMotivo.setManaged(false);
        colorBar.setFill(Color.web("#1A2A1A"));
    }

    private void animaBarraConfidenza(double target) {
        Timeline tl = new Timeline(
                new KeyFrame(Duration.ZERO,
                        new KeyValue(barraConfidenza.progressProperty(), 0)),
                new KeyFrame(Duration.millis(700),
                        new KeyValue(barraConfidenza.progressProperty(), target,
                                Interpolator.EASE_OUT)));
        tl.play();
    }

    // ── Pulisci ──────────────────────────────────────────────────

    @FXML
    private void onPulisci() {
        fileSelezionato = null;
        previewImage.setImage(null);
        previewImage.setVisible(false);
        dropPlaceholder.setVisible(true);
        dropPlaceholder.setManaged(true);
        labelNomeFile.setText("");
        btnAnalizza.setDisable(true);
        resetRisultato();
        if (modalitaOnline) {
            setStatus("Pronto — trascina un'immagine o clicca nell'area di caricamento.");
        } else {
            if (serverPronto) {
                setStatus("✅ Server pronto — carica un'immagine.");
            } else if (erroreServer != null) {
                setStatus("⚠️ Server non disponibile: " + erroreServer);
            } else {
                setStatus("⏳ Avvio server per analisi offline...");
            }
        }
    }

    // ── Loading ──────────────────────────────────────────────────

    private void mostraLoading(boolean mostra, String testo) {
        if (mostra) {
            if (testo != null)
                labelLoading.setText(testo);
            loadingProgressBar.setProgress(0.0);

            if (loadingTimeline != null)
                loadingTimeline.stop();
            loadingTimeline = new Timeline(
                    new KeyFrame(Duration.ZERO, new KeyValue(loadingProgressBar.progressProperty(), 0.0)),
                    new KeyFrame(Duration.millis(600),
                            new KeyValue(loadingProgressBar.progressProperty(), 0.85, Interpolator.EASE_OUT)),
                    new KeyFrame(Duration.millis(5000),
                            new KeyValue(loadingProgressBar.progressProperty(), 0.95, Interpolator.LINEAR)));
            loadingTimeline.play();

            if (spinnerTransition != null)
                spinnerTransition.stop();
            spinnerTransition = new RotateTransition(Duration.millis(1200), loadingSpinner);
            spinnerTransition.setByAngle(360);
            spinnerTransition.setCycleCount(Animation.INDEFINITE);
            spinnerTransition.setInterpolator(Interpolator.LINEAR);
            spinnerTransition.play();

            loadingOverlay.setOpacity(1.0);
            loadingOverlay.setVisible(true);
            loadingOverlay.setManaged(true);
        } else {
            if (loadingTimeline != null)
                loadingTimeline.stop();
            if (spinnerTransition != null) {
                spinnerTransition.stop();
                loadingSpinner.setRotate(0);
            }

            if (loadingProgressBar.getProgress() > 0 && loadingProgressBar.getProgress() < 1.0) {
                Timeline finishTimeline = new Timeline(
                        new KeyFrame(Duration.millis(200),
                                new KeyValue(loadingProgressBar.progressProperty(), 1.0, Interpolator.EASE_BOTH)));
                finishTimeline.setOnFinished(ev -> {
                    FadeTransition fade = new FadeTransition(Duration.millis(150), loadingOverlay);
                    fade.setFromValue(1.0);
                    fade.setToValue(0.0);
                    fade.setOnFinished(fEv -> {
                        loadingOverlay.setVisible(false);
                        loadingOverlay.setManaged(false);
                        loadingOverlay.setOpacity(1.0);
                    });
                    fade.play();
                });
                finishTimeline.play();
            } else {
                loadingOverlay.setVisible(false);
                loadingOverlay.setManaged(false);
            }
        }
    }

    // ── Helpers ──────────────────────────────────────────────────

    private void setStatus(String msg) {
        Platform.runLater(() -> statusBar.setText(msg));
    }

    private boolean isImmagine(File f) {
        String n = f.getName().toLowerCase();
        return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png")
                || n.endsWith(".bmp") || n.endsWith(".webp") || n.endsWith(".gif");
    }

    public void shutdown() {
        cameraAttiva = false;
        chiudiWebcam();
        localAI.fermaServer();
    }

    // ── Camera capture actions ───────────────────────────────────

    @FXML
    private void onCameraClick() {
        if (cameraAttiva || (webcam != null && webcam.isOpen())) {
            return;
        }

        // Se c'è un'analisi precedente, nascondila
        resetRisultato();

        // Mostra il contenitore della camera e nascondi la preview/placeholder
        cameraContainer.setVisible(true);
        cameraContainer.setManaged(true);
        dropPlaceholder.setVisible(false);
        dropPlaceholder.setManaged(false);
        previewImage.setVisible(false);

        setStatus("⏳ Avvio fotocamera in corso...");
        btnCamera.setDisable(true);
        btnScatta.setDisable(true);
        btnAnnullaCamera.setDisable(true);

        Task<Void> avvioCameraTask = new Task<>() {
            @Override
            protected Void call() throws Exception {
                webcam = Webcam.getDefault();
                if (webcam == null) {
                    throw new RuntimeException("Nessuna fotocamera rilevata nel sistema.");
                }
                webcam.open();
                return null;
            }
        };

        avvioCameraTask.setOnSucceeded(e -> {
            cameraAttiva = true;
            btnCamera.setDisable(false);
            btnScatta.setDisable(false);
            btnAnnullaCamera.setDisable(false);
            setStatus("📸 Fotocamera attiva — inquadra e clicca Cattura.");
            avviaWebcamStream();
        });

        avvioCameraTask.setOnFailed(e -> {
            btnCamera.setDisable(false);
            onAnnullaCamera();
            setStatus("❌ Errore avvio fotocamera: " + avvioCameraTask.getException().getMessage());
        });

        Thread t = new Thread(avvioCameraTask);
        t.setDaemon(true);
        t.start();
    }

    private void avviaWebcamStream() {
        Thread streamThread = new Thread(() -> {
            while (cameraAttiva && webcam != null && webcam.isOpen()) {
                try {
                    java.awt.image.BufferedImage img = webcam.getImage();
                    if (img != null) {
                        Image fxImg = convertiBufferedImage(img);
                        Platform.runLater(() -> cameraFeed.setImage(fxImg));
                    }
                    Thread.sleep(40); // circa 25 FPS
                } catch (InterruptedException ex) {
                    break;
                } catch (Exception ex) {
                    System.err.println("Errore stream camera: " + ex.getMessage());
                }
            }
        });
        streamThread.setDaemon(true);
        streamThread.start();
    }

    private Image convertiBufferedImage(java.awt.image.BufferedImage bimg) {
        try {
            java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
            javax.imageio.ImageIO.write(bimg, "png", out);
            out.flush();
            return new Image(new java.io.ByteArrayInputStream(out.toByteArray()));
        } catch (java.io.IOException e) {
            return null;
        }
    }

    @FXML
    private void onScattaFoto() {
        if (webcam == null || !webcam.isOpen())
            return;

        cameraAttiva = false;
        try {
            java.awt.image.BufferedImage bimg = webcam.getImage();
            if (bimg != null) {
                // Crea file temporaneo per salvare l'immagine
                File tempFile = File.createTempFile("ecosort_captured_", ".jpg");
                tempFile.deleteOnExit();
                javax.imageio.ImageIO.write(bimg, "jpg", tempFile);

                // Chiudi webcam
                chiudiWebcam();

                // Pulisci l'immagine del feed
                cameraFeed.setImage(null);
                cameraContainer.setVisible(false);
                cameraContainer.setManaged(false);

                // Carica la foto come se fosse stata aperta tramite FileChooser
                caricaImmagine(tempFile);
                setStatus("✅ Foto catturata con successo!");
            }
        } catch (Exception ex) {
            setStatus("❌ Errore cattura foto: " + ex.getMessage());
            onAnnullaCamera();
        }
    }

    @FXML
    private void onAnnullaCamera() {
        cameraAttiva = false;
        chiudiWebcam();

        cameraFeed.setImage(null);
        cameraContainer.setVisible(false);
        cameraContainer.setManaged(false);

        // Ripristina stato precedente
        if (fileSelezionato != null) {
            previewImage.setVisible(true);
            dropPlaceholder.setVisible(false);
            dropPlaceholder.setManaged(false);
            setStatus("Cancellato. Ripristinata immagine caricata in precedenza.");
        } else {
            previewImage.setVisible(false);
            dropPlaceholder.setVisible(true);
            dropPlaceholder.setManaged(true);
            if (modalitaOnline) {
                setStatus("Cancellato. Pronto per caricare un'immagine o usare la camera.");
            } else {
                if (serverPronto) {
                    setStatus("Cancellato. Server pronto — carica un'immagine o usa la camera.");
                } else if (erroreServer != null) {
                    setStatus("Cancellato. Server  non disponibile: " + erroreServer);
                } else {
                    setStatus("Cancellato. In attesa del server per analisi offline...");
                }
            }
        }
    }

    private void chiudiWebcam() {
        if (webcam != null) {
            try {
                webcam.close();
            } catch (Exception e) {
                System.err.println("Errore durante la chiusura della fotocamera: " + e.getMessage());
            }
            webcam = null;
        }
    }

    private String getApiKey() {
        // 1) Controlla se la chiave è inserita direttamente nell'interfaccia
        // (fieldApiKey)
        if (fieldApiKey != null && !fieldApiKey.getText().trim().isEmpty()) {
            return fieldApiKey.getText().trim();
        }

        // 2) Controlla variabile d'ambiente di sistema
        String envKey = System.getenv("GEMINI_API_KEY");
        if (envKey != null && !envKey.trim().isEmpty()) {
            return envKey.trim();
        }

        // 3) Proviamo a leggere da un file .env (cercando GEMINI_API_KEY=chiave)
        try {
            File envFile = new File(".env");
            if (envFile.exists()) {
                java.util.List<String> lines = java.nio.file.Files.readAllLines(envFile.toPath());
                for (String line : lines) {
                    String trimmed = line.trim();
                    if (trimmed.startsWith("GEMINI_API_KEY=") || trimmed.startsWith("GEMINI_KEY=")) {
                        String value = trimmed.substring(trimmed.indexOf("=") + 1).trim();
                        // Rimuove eventuali virgolette singole o doppie circostanti
                        value = value.replaceAll("^[\"']|[\"']$", "");
                        if (!value.isEmpty()) {
                            return value;
                        }
                    }
                }
            }
        } catch (Exception ignored) {
        }

        // 4) Fallback: proviamo a leggere da gemini_key.txt (chiave grezza sul primo
        // rigo)
        try {
            File keyFile = new File("gemini_key.txt");
            if (keyFile.exists()) {
                return java.nio.file.Files.readString(keyFile.toPath()).trim();
            }
        } catch (Exception ignored) {
        }

        // 5) Fallback finale: leggi dalle preferenze salvate nel registro di Windows
        return prefs.get(PREF_API_KEY, "");
    }
}
