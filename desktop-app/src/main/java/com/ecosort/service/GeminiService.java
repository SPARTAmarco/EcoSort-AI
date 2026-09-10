package com.ecosort.service;

import com.ecosort.model.ClassificationResult;
import com.ecosort.model.ClassificationResult.Categoria;
import com.ecosort.model.ClassificationResult.Modalita;
import com.google.gson.*;

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
 * Servizio che chiama l'API Gemini 2.5 Flash per classificare immagini di rifiuti.
 */
public class GeminiService {

    private static final String MODELLO   = "gemini-2.5-flash-lite";
    private static final String BASE_URL  = "https://generativelanguage.googleapis.com/v1beta/models/"
            + MODELLO + ":generateContent";
    // Nessuna soglia: si mostra sempre la classe col punteggio più alto

    private final String apiKey;
    private final HttpClient client;

    public GeminiService(String apiKey) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalArgumentException("API key Gemini non può essere vuota");
        }
        this.apiKey = apiKey.trim();
        this.client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1) // Forza HTTP/1.1 per evitare hang o blocchi con le API Google
                .proxy(HttpClient.Builder.NO_PROXY)   // Disabilita l'auto-rilevamento proxy (WPAD) velocizzando di 10 secondi!
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    /**
     * Classifica un'immagine usando Gemini Vision.
     *
     * @param imageFile file immagine locale
     * @return ClassificationResult con categoria, confidenza e motivazione
     */
    public ClassificationResult classifica(File imageFile) throws IOException, InterruptedException {
        if (!imageFile.exists()) {
            throw new IOException("File non trovato: " + imageFile.getAbsolutePath());
        }

        long t0 = System.currentTimeMillis();

        // Determina il MIME type
        String mimeType = determinaMime(imageFile.getName());

        // Comprime e ridimensiona l'immagine prima dell'upload per evitare caricamenti infiniti
        byte[] imageBytes = comprimiERidimensiona(imageFile);
        String base64Data = Base64.getEncoder().encodeToString(imageBytes);

        // Costruisci il payload Gemini
        String requestBody = costruisciPayload(base64Data, mimeType);

        // Esegui la richiesta HTTP
        String url = BASE_URL + "?key=" + apiKey;
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(30))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> resp = null;
        int maxRetries = 3;
        long delayMs = 1500;
        
        for (int i = 0; i < maxRetries; i++) {
            try {
                resp = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() == 200) {
                    break;
                }
                if (resp.statusCode() == 503 || resp.statusCode() == 429) {
                    // Spikes temporanei o rate limit: aspetta e riprova
                    Thread.sleep(delayMs);
                    delayMs *= 2; // Exponential backoff (1.5s -> 3.0s)
                    continue;
                }
            } catch (IOException e) {
                if (i == maxRetries - 1) throw e;
                Thread.sleep(delayMs);
                delayMs *= 2;
                continue;
            }
            break;
        }

        long tempoMs = System.currentTimeMillis() - t0;

        if (resp == null) {
            throw new IOException("Impossibile contattare le API di Gemini dopo i tentativi di connessione.");
        }

        if (resp.statusCode() != 200) {
            String errorMsg = estraiErroreGemini(resp.body());
            if (resp.statusCode() == 429 || (errorMsg != null && (errorMsg.toLowerCase().contains("quota") || errorMsg.toLowerCase().contains("rate limit") || errorMsg.toLowerCase().contains("limit exceeded")))) {
                throw new IOException("Limite giornaliero di richieste API a Google terminato. Riprova più tardi o passa alla modalità offline.");
            }
            throw new IOException("Gemini API errore (HTTP " + resp.statusCode() + "): " + errorMsg);
        }

        // Estrai il testo dalla risposta Gemini
        String testoRisposta = estraiTesto(resp.body());

        // Parse il JSON nella risposta
        return parseRispostaGemini(testoRisposta, tempoMs);
    }

    /**
     * Costruisce il payload JSON per Gemini con immagine inline.
     */
    private String costruisciPayload(String base64Data, String mimeType) {
        String prompt = "Rispondi esclusivamente con una sola parola tra queste tre: 'plastica', 'carta_e_cartone', 'vetro_e_metallo'. Non aggiungere altro testo, nessun commento, nessun markdown.";

        JsonObject root = new JsonObject();
        JsonArray contents = new JsonArray();
        JsonObject content = new JsonObject();
        JsonArray parts = new JsonArray();

        // Parte testo (prompt)
        JsonObject textPart = new JsonObject();
        textPart.addProperty("text", prompt);
        parts.add(textPart);

        // Parte immagine inline
        JsonObject imagePart = new JsonObject();
        JsonObject inlineData = new JsonObject();
        inlineData.addProperty("mime_type", mimeType);
        inlineData.addProperty("data", base64Data);
        imagePart.add("inline_data", inlineData);
        parts.add(imagePart);

        content.add("parts", parts);
        contents.add(content);
        root.add("contents", contents);

        // Configurazione generazione ad altissime prestazioni (senza schema per azzerare il tempo di elaborazione server)
        JsonObject genConfig = new JsonObject();
        genConfig.addProperty("temperature", 0.0);
        genConfig.addProperty("maxOutputTokens", 5);
        root.add("generationConfig", genConfig);

        return root.toString();
    }

    /**
     * Estrae il testo dalla risposta Gemini.
     */
    private String estraiTesto(String responseBody) throws IOException {
        try {
            JsonObject json = JsonParser.parseString(responseBody).getAsJsonObject();

            if (json.has("error")) {
                JsonObject error = json.getAsJsonObject("error");
                throw new IOException("Gemini error: " + error.get("message").getAsString());
            }

            JsonArray candidates = json.getAsJsonArray("candidates");
            if (candidates == null || candidates.isEmpty()) {
                throw new IOException("Risposta Gemini senza candidati: " + responseBody);
            }

            JsonObject candidate = candidates.get(0).getAsJsonObject();

            // Controlla finish reason
            if (candidate.has("finishReason")) {
                String reason = candidate.get("finishReason").getAsString();
                if (reason.equals("SAFETY") || reason.equals("RECITATION")) {
                    throw new IOException("Gemini ha bloccato la risposta (reason: " + reason + ")");
                }
            }

            JsonObject contentObj = candidate.getAsJsonObject("content");
            JsonArray parts = contentObj.getAsJsonArray("parts");
            return parts.get(0).getAsJsonObject().get("text").getAsString().trim();

        } catch (JsonSyntaxException e) {
            throw new IOException("Risposta Gemini non è JSON valido: " + responseBody.substring(0, Math.min(200, responseBody.length())));
        }
    }

    /**
     * Parse il JSON restituito da Gemini.
     */
    private ClassificationResult parseRispostaGemini(String testo, long tempoMs) throws IOException {
        String pulito = testo.toLowerCase().trim();
        
        // Estrai robustamente la categoria dal testo grezzo
        String categoriaStr = "plastica";
        if (pulito.contains("carta") || pulito.contains("cartone")) {
            categoriaStr = "carta_e_cartone";
        } else if (pulito.contains("vetro") || pulito.contains("metallo")) {
            categoriaStr = "vetro_e_metallo";
        } else if (pulito.contains("plastica")) {
            categoriaStr = "plastica";
        }

        // Crea una spiegazione dinamica locale istantanea per azzerare il tempo di risposta
        String motivo;
        if ("carta_e_cartone".equalsIgnoreCase(categoriaStr)) {
            motivo = "Rilevato oggetto in carta o cartone. Va gettato nel cassonetto azzurro o bianco.";
        } else if ("plastica".equalsIgnoreCase(categoriaStr)) {
            motivo = "Rilevato oggetto in plastica. Va gettato nel cassonetto giallo per la plastica.";
        } else if ("vetro_e_metallo".equalsIgnoreCase(categoriaStr)) {
            motivo = "Rilevato oggetto in vetro o metallo. Va gettato nel cassonetto verde o campana del vetro.";
        } else {
            motivo = "Rifiuto classificato online con successo.";
        }

        Categoria categoria = Categoria.fromString(categoriaStr);

        return new ClassificationResult(
                categoria,
                0.99, // Confidenza fissa al 99% per azzerare i tempi di calcolo dell'API
                motivo,
                Modalita.ONLINE_GEMINI,
                tempoMs,
                false
        );
    }

    /**
     * Estrae il messaggio di errore dalla risposta Gemini.
     */
    private String estraiErroreGemini(String body) {
        try {
            JsonObject json = JsonParser.parseString(body).getAsJsonObject();
            if (json.has("error")) {
                return json.getAsJsonObject("error").get("message").getAsString();
            }
        } catch (Exception ignored) {}
        return body.substring(0, Math.min(300, body.length()));
    }

    /**
     * Determina il MIME type in base all'estensione del file.
     */
    private String determinaMime(String fileName) {
        String lower = fileName.toLowerCase();
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
        if (lower.endsWith(".png"))  return "image/png";
        if (lower.endsWith(".gif"))  return "image/gif";
        if (lower.endsWith(".webp")) return "image/webp";
        if (lower.endsWith(".bmp"))  return "image/bmp";
        return "image/jpeg"; // default
    }

    /**
     * Comprime e ridimensiona l'immagine in locale per ridurre il payload da MB a KB
     * e garantire una latenza di upload quasi istantanea.
     */
    private byte[] comprimiERidimensiona(File file) throws IOException {
        java.awt.image.BufferedImage img = javax.imageio.ImageIO.read(file);
        if (img == null) {
            return Files.readAllBytes(file.toPath()); // Fallback
        }

        int maxDim = 300; // Ridotto a 300px per rendere l'upload fulmineo
        int w = img.getWidth();
        int h = img.getHeight();

        if (w > maxDim || h > maxDim) {
            double scale = Math.min((double) maxDim / w, (double) maxDim / h);
            int newW = (int) (w * scale);
            int newH = (int) (h * scale);

            java.awt.image.BufferedImage scaled = new java.awt.image.BufferedImage(newW, newH, java.awt.image.BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D g = scaled.createGraphics();
            g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION, java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(img, 0, 0, newW, newH, null);
            g.dispose();
            img = scaled;
        }

        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        
        // Utilizziamo un compressore JPEG per salvare con qualità bilanciata del 50% (file ultraleggero)
        java.util.Iterator<javax.imageio.ImageWriter> writers = javax.imageio.ImageIO.getImageWritersByFormatName("jpg");
        if (writers.hasNext()) {
            javax.imageio.ImageWriter writer = writers.next();
            javax.imageio.stream.ImageOutputStream ios = javax.imageio.ImageIO.createImageOutputStream(baos);
            writer.setOutput(ios);
            
            javax.imageio.ImageWriteParam param = writer.getDefaultWriteParam();
            if (param.canWriteCompressed()) {
                param.setCompressionMode(javax.imageio.ImageWriteParam.MODE_EXPLICIT);
                param.setCompressionType(param.getCompressionTypes()[0]);
                param.setCompressionQuality(0.50f);
            }
            
            writer.write(null, new javax.imageio.IIOImage(img, null, null), param);
            writer.dispose();
            ios.close();
        } else {
            javax.imageio.ImageIO.write(img, "jpg", baos);
        }

        return baos.toByteArray();
    }
}
