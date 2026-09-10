package com.ecosort;

import com.ecosort.controller.MainController;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class EcoSortApp extends Application {

    @Override
    public void start(Stage stage) throws Exception {
        FXMLLoader loader = new FXMLLoader(
            getClass().getResource("/com/ecosort/fxml/MainView.fxml")
        );
        Scene scene = new Scene(loader.load(), 1120, 740);
        scene.getStylesheets().add(
            getClass().getResource("/com/ecosort/css/style.css").toExternalForm()
        );
        stage.setTitle("EcoSort AI");
        stage.setScene(scene);
        stage.setMinWidth(900);
        stage.setMinHeight(640);

        // Ferma il server Python alla chiusura
        MainController controller = loader.getController();
        stage.setOnCloseRequest(e -> controller.shutdown());

        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
