package es.uma.algorithms;

import es.uma.data.SimpleDataSet;
import es.uma.models.PredictionRequestEvent;

import java.io.IOException;
import java.util.ArrayList;

public class TrainedModel {

    // Variables for the network and dataset
    private SimpleDataSet ds;
    private RecurrentNeuralNetwork net;
    private AlgorithmConfiguration algorithmConfiguration;
    private PredictionRequestEvent requestEvent;
    private String modelID;
    private static Integer DAY = 24*60/4;

    public TrainedModel(AlgorithmConfiguration algorithmConfiguration) {
        this.setModelID("");
        this.setAlgorithmConfiguration(algorithmConfiguration);
        setDs(new SimpleDataSet(getAlgorithmConfiguration().getDatapath()+"/",
                getAlgorithmConfiguration().getDatafile(), true));

        try {
            getDs().generateFiles(getAlgorithmConfiguration().getPeriod());
        } catch (IOException e) {
            e.printStackTrace();
        }
        getDs().setTrainingSize(getAlgorithmConfiguration().getTraining());

        setNet(new RecurrentNeuralNetwork(getAlgorithmConfiguration().getLayers()));

        getNet().build(40);
    }

    public ArrayList<ArrayList<Double>> getPrediction(){
        ArrayList<ArrayList<Double>> a = null;
        Integer number = null;
        switch(getAlgorithmConfiguration().getPrediction()){
            case LARGE: number = 30*DAY; break;
            case MEDIUM: number = 7*DAY; break;
            case SHORT: number = DAY;
        }
        switch (getAlgorithmConfiguration().getPredictionPeriod()){
            case D1: number = number / DAY; break;
            case M60: number = number / 4; break;
            case M30: number = number / 2;
        }

        getNet().setParameters(getAlgorithmConfiguration().getWeights());
        try {
            a = getNet().predict(getDs().getTestData(), number);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return a;
    }

    public void train(){
        try {
            getNet().train(getDs().getTrainingData());
            getNet().test(getDs().getTestData(), false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public SimpleDataSet getDs() {
        return ds;
    }

    public void setDs(SimpleDataSet ds) {
        this.ds = ds;
    }

    public RecurrentNeuralNetwork getNet() {
        return net;
    }

    public void setNet(RecurrentNeuralNetwork net) {
        this.net = net;
    }

    public AlgorithmConfiguration getAlgorithmConfiguration() {
        return algorithmConfiguration;
    }

    public void setAlgorithmConfiguration(AlgorithmConfiguration algorithmConfiguration) {
        this.algorithmConfiguration = algorithmConfiguration;
    }

    public Double getMSE(){
        getNet().setParameters(getAlgorithmConfiguration().getWeights());
        try {
            getNet().test(getDs().getTestData(), false);
        } catch (Exception e) {
            e.printStackTrace();
        }
        Double mse =  getNet().getMSE();
        if(Double.isNaN(mse)){
            mse = 1000000.0;
        }
        return mse;
    }

    public Double getConfidenceLevel(){
        if(getNet().getCL() < 0){
            try {
                getNet().test(getDs().getTestData(), false);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return getNet().getCL();
    }

    public PredictionRequestEvent getRequestEvent() {
        return requestEvent;
    }

    public void setRequestEvent(PredictionRequestEvent requestEvent) {
        this.requestEvent = requestEvent;
    }

    public String getModelID() {
        if(modelID == ""){
            modelID = ""+requestEvent.hashCode();
        }
        return modelID;
    }

    public void setModelID(String modelID) {
        this.modelID = modelID;
    }
}
