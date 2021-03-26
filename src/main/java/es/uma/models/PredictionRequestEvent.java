package es.uma.models;

import java.util.ArrayList;

public class PredictionRequestEvent {
    public String id;
    public String timestamp;
    public String name;
    public String version;
    public String executionId;
    public String operation;
    public Model model;
    public ArrayList<Origins> origins;
    public Output output;
    public Source modelsource;
}
