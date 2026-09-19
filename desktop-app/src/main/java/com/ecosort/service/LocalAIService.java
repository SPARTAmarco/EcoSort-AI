package com.ecosort.service;

import com.ecosort.model.ClassificationResult;
import com.ecosort.model.ClassificationResult.Categoria;
import com.ecosort.model.ClassificationResult.Modalita;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.time.Duration;
import java.util.Base64;

/**
 * Servizio che comunica con il server Flask Python locale su http://127.0.0.1:5891
 */
public class LocalAIService {

    private static final String BASE_URL = "http://127.0.0.1:5891";
    private final HttpClient client;
    private Process serverProcess;

    public LocalAIService() {
        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    /**
     * Avvia server.py come processo in background.
     * Cerca python.exe prima in python-embedded/ relativa al JAR, poi nel PATH.
     */
    public void avviaServer() throws IOException {
        if (serverProcess != null && serverProcess.isAlive()) {
            return; // già in esecuzione
        }

        // Determina il percorso del JAR/working directory
        File jarDir;
        try {
            jarDir = new File(
                    LocalAIService.class.getProtectionDomain()
                            .getCodeSource().getLocation().toURI()
            ).getParentFile();
        } catch (Exception e) {
            jarDir = new File(".").getAbsoluteFile();
        }

        // Cerca python embedded prima
        String pythonExe = trovaPython(jarDir);

        // Trova server.py — prova diversi percorsi in ordine
        File serverPy = trovareServerPy(jarDir);
        if (serverPy == null) {
            throw new IOException(
                "server.py non trovato. Assicurati che la cartella 'server/' " +
                "sia vicina al JAR oppure alla directory di lavoro corrente (" +
                new File(".").getAbsolutePath() + ")"
            );
        }

        ProcessBuilder pb = new ProcessBuilder(pythonExe, serverPy.getAbsolutePath());
        pb.directory(serverPy.getParentFile());
        pb.redirectErrorStream(true);

        // Reindirizza output su un file di log nella CWD
        File logFile = new File("ecosort-server.log").getAbsoluteFile();
        pb.redirectOutput(logFile);

        serverProcess = pb.start();
        System.out.println("[LocalAI] Server Python avviato (PID=" + serverProcess.pid() + ")");
    }

    /**
     * Cerca server.py in varie posizioni possibili.
     */
    private File trovareServerPy(File jarDir) {
        // Candidati in ordine di preferenza
        String[] candidati = {
            "server/server.py",           // relativo alla CWD (mvn javafx:run)
            "../server/server.py",        // un livello su
        };
        // Prima prova relativo alla CWD
        for (String rel : candidati) {
            File f = new File(rel).getAbsoluteFile();
            if (f.exists()) {
                System.out.println("[LocalAI] Trovato server.py: " + f.getAbsolutePath());
                return f;
            }
        }
        // Poi prova relativo al JAR
        if (jarDir != null) {
            File f = new File(jarDir, "server/server.py");
            if (f.exists()) {
                System.out.println("[LocalAI] Trovato server.py (jar-rel): " + f.getAbsolutePath());
                return f;
            }
            // Risali di qualche livello (es. dentro target/)
            File parent = jarDir.getParentFile();
            while (parent != null) {
                f = new File(parent, "server/server.py");
                if (f.exists()) {
                    System.out.println("[LocalAI] Trovato server.py (parent): " + f.getAbsolutePath());
                    return f;
                }
                parent = parent.getParentFile();
            }
        }
        return null;
    }

    /**
     * Cerca python.exe: prima in python-embedded/ vicino al JAR, poi nel PATH.
     */
    private String trovaPython(File jarDir) throws IOException {
        // 0) Variabile d'ambiente esplicita (la imposta avvia.bat)
        String daEnv = System.getenv("ECOSORT_PYTHON");
        if (daEnv != null && !daEnv.isBlank() && testPythonExe(daEnv)) {
            System.out.println("[LocalAI] Usando ECOSORT_PYTHON: " + daEnv);
            return daEnv;
        }

        // 0b) Ambiente virtuale .venv creato da avvia.bat (CWD, cartella del JAR o la sua madre)
        File jarParent = jarDir != null ? jarDir.getParentFile() : null;
        for (File base : new File[]{new File("."), jarDir, jarParent}) {
            if (base == null) continue;
            for (String rel : new String[]{".venv/Scripts/python.exe", ".venv/bin/python"}) {
                File venvExe = new File(base, rel).getAbsoluteFile();
                if (venvExe.exists() && testPythonExe(venvExe.getAbsolutePath())) {
                    System.out.println("[LocalAI] Usando .venv: " + venvExe.getAbsolutePath());
                    return venvExe.getAbsolutePath();
                }
            }
        }

        // 1) Python embedded nella cartella del JAR
        for (File base : new File[]{jarDir, new File(".")}) {
            if (base == null) continue;
            File embeddedExe = new File(base, "python-embedded/python.exe");
            if (embeddedExe.exists() && testPythonExe(embeddedExe.getAbsolutePath())) {
                System.out.println("[LocalAI] Usando Python embedded: " + embeddedExe.getAbsolutePath());
                return embeddedExe.getAbsolutePath();
            }
        }

        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            // 2) Cerca il percorso Python reale per evitare il Microsoft Store alias stub (WindowsApps)
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData != null) {
                File pythonDir = new File(localAppData, "Programs/Python");
                if (pythonDir.exists() && pythonDir.isDirectory()) {
                    File[] subDirs = pythonDir.listFiles();
                    if (subDirs != null) {
                        for (File sub : subDirs) {
                            File exe = new File(sub, "python.exe");
                            if (exe.exists() && testPythonExe(exe.getAbsolutePath())) {
                                System.out.println("[LocalAI] Trovato Python reale in AppData: " + exe.getAbsolutePath());
                                return exe.getAbsolutePath();
                            }
                        }
                    }
                }
            }

            // Cerca in Program Files
            for (String baseDir : new String[]{"C:\\Program Files", "C:\\Program Files (x86)"}) {
                File pDir = new File(baseDir);
                if (pDir.exists()) {
                    File[] subDirs = pDir.listFiles();
                    if (subDirs != null) {
                        for (File sub : subDirs) {
                            if (sub.getName().toLowerCase().startsWith("python")) {
                                File exe = new File(sub, "python.exe");
                                if (exe.exists() && testPythonExe(exe.getAbsolutePath())) {
                                    System.out.println("[LocalAI] Trovato Python reale in Program Files: " + exe.getAbsolutePath());
                                    return exe.getAbsolutePath();
                                }
                            }
                        }
                    }
                }
            }

            // 3) Fallback su Windows launcher e di sistema
            for (String cmd : new String[]{"py", "python"}) {
                if (testPythonExe(cmd)) {
                    System.out.println("[LocalAI] Usando Python di sistema: " + cmd);
                    return cmd;
                }
            }
        } else {
            if (testPythonExe("python3")) return "python3";
            if (testPythonExe("python")) return "python";
        }
        
        throw new IOException("Python non è installato o non trovato sul sistema. Per usare la Modalità Offline devi installare Python.");
    }

    /**
     * Verifica se l'eseguibile Python specificato è funzionante ed esegue codice.
     */
    private boolean testPythonExe(String path) {
        try {
            Process p = new ProcessBuilder(path, "-c", "import sys; print(sys.version_info[:2])")
                    .redirectErrorStream(true)
                    .start();
            boolean finished = p.waitFor(3, java.util.concurrent.TimeUnit.SECONDS);
            if (!finished) {
                p.destroyForcibly();
                return false;
            }
            if (p.exitValue() != 0) {
                return false;
            }
            byte[] bytes = p.getInputStream().readAllBytes();
            String out = new String(bytes).trim();
            if (out.isEmpty() || out.toLowerCase().contains("unable to create") || out.toLowerCase().contains("error")) {
                return false;
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Verifica se un comando è disponibile nel PATH.
     */
    private boolean comandoEsiste(String cmd) {
        try {
            Process p = new ProcessBuilder(cmd, "--version")
                    .redirectErrorStream(true).start();
            p.waitFor();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Fa polling su /ping finché il server risponde o scade il timeout.
     *
     * @param timeoutMs timeout massimo in ms
     * @throws IOException se il server non risponde entro il timeout
     */
    public void attendiPronto(int timeoutMs) throws IOException, InterruptedException {
        long scadenza = System.currentTimeMillis() + timeoutMs;
        int intervallo = 300;

        while (System.currentTimeMillis() < scadenza) {
            if (serverProcess != null && !serverProcess.isAlive()) {
                throw new IOException("Il server Python si è fermato inaspettatamente. Controlla ecosort-server.log");
            }
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(BASE_URL + "/ping"))
                        .timeout(Duration.ofSeconds(2))
                        .GET()
                        .build();
                HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() == 200) {
                    System.out.println("[LocalAI] Server pronto: " + resp.body());
                    return;
                }
            } catch (Exception ignored) {
                // Server non ancora pronto
            }
            Thread.sleep(intervallo);
        }
        throw new IOException("Timeout: il server Python non ha risposto entro " + timeoutMs + "ms");
    }

    /**
     * Classifica un'immagine usando il server Flask locale.
     *
     * @param imageFile file immagine da classificare
     * @return ClassificationResult con i risultati
     */
    public ClassificationResult classifica(File imageFile) throws IOException, InterruptedException {
        if (!imageFile.exists()) {
            throw new IOException("File non trovato: " + imageFile.getAbsolutePath());
        }

        long t0 = System.currentTimeMillis();

        // Leggi e converti in base64
        byte[] bytes = Files.readAllBytes(imageFile.toPath());
        String base64 = Base64.getEncoder().encodeToString(bytes);

        // Costruisci JSON
        JsonObject payload = new JsonObject();
        payload.addProperty("image", base64);
        String jsonBody = payload.toString();

        // Manda POST a /classifica
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(BASE_URL + "/classifica"))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        HttpResponse<String> resp;
        try {
            resp = client.send(req, HttpResponse.BodyHandlers.ofString());
        } catch (java.net.ConnectException e) {
            throw new IOException("Il server offline non è pronto o non sta funzionando. Assicurati di avere Python installato.");
        } catch (IOException e) {
            if (e.getMessage() != null && e.getMessage().contains("refused")) {
                throw new IOException("Il server offline non è pronto o non sta funzionando. Assicurati di avere Python installato.");
            }
            throw e;
        }
        long tempoMs = System.currentTimeMillis() - t0;

        if (resp.statusCode() != 200) {
            throw new IOException("Server ha risposto con HTTP " + resp.statusCode() + ": " + resp.body());
        }

        // Parse risposta JSON
        JsonObject json = JsonParser.parseString(resp.body()).getAsJsonObject();

        if (json.has("errore")) {
            throw new IOException("Errore server: " + json.get("errore").getAsString());
        }

        // Usa il tempo riportato dal server se disponibile
        if (json.has("tempo_ms")) {
            tempoMs = (long) json.get("tempo_ms").getAsDouble();
        }

        // Usa direttamente la decisione del server (che applica softmax + soglia)
        String categoriaStr = json.has("categoria")
                ? json.get("categoria").getAsString()
                : "indifferenziata";

        // La confidenza è la probabilità softmax VERA della classe vincente
        double confidenza = json.has("confidenza")
                ? json.get("confidenza").getAsDouble()
                : 0.0;

        // Se il server ha segnalato "sotto_soglia", forza INDIFFERENZIATA
        boolean sottoSoglia = json.has("sotto_soglia") && json.get("sotto_soglia").getAsBoolean();
        if (sottoSoglia) {
            categoriaStr = "indifferenziata";
        }

        Categoria categoria = Categoria.fromString(categoriaStr);

        return new ClassificationResult(
                categoria,
                confidenza,
                null,
                Modalita.OFFLINE_TFLITE,
                tempoMs,
                sottoSoglia
        );
    }

    /**
     * Ferma il processo Python se in esecuzione.
     */
    public void fermaServer() {
        if (serverProcess != null && serverProcess.isAlive()) {
            serverProcess.destroy();
            System.out.println("[LocalAI] Server Python fermato");
        }
    }

    /**
     * Verifica se il server è in esecuzione.
     */
    public boolean isServerInEsecuzione() {
        return serverProcess != null && serverProcess.isAlive();
    }
}
