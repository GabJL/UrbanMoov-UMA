package es.uma.test;

import es.uma.algorithms.RecurrentNeuralNetwork;
import es.uma.data.SimpleDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.ArrayList;

public class testSimpleRNN {

    public static String datapath = "data/ble/";
    public static String datafile = "ble.csv";

    public static void main(String [] args) throws IOException, InterruptedException {
        SimpleDataSet ds = new SimpleDataSet(datapath, datafile, true);

        ds.generateFiles(4);
        ds.setTrainingSize(0.75);

        DataSetIterator trainingData = ds.getTrainingData();
        DataSetIterator testData = ds.getTestData();

        int [] layers = {8, 64, 32, 24, 8};

        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(layers);

        net.build(40);
        // net.writeParameters();
        System.out.println("Number of parameters: " + net.getNumberOfParameters());
        //double [] w = new double[net.getNumberOfParameters()];
        //Random r = new Random();
        //for(int i = 0; i < w.length;i++)
        //    w[i] = r.nextDouble()*4 - 2;
        //net.setParameters(w);
        //net.writeParameters();
        net.train(trainingData);

        System.out.println("Training set:");
        net.test(trainingData);
        System.out.println("MSE = " + net.getMSE());
        System.out.println("MAE = " + net.getMAE());

        System.out.println("Test set:");
        net.test(testData);
        System.out.println("MSE = " + net.getMSE());
        System.out.println("MAE = " + net.getMAE());
        ArrayList<ArrayList<Double>> p = net.predict(testData, 10);
        for(ArrayList<Double> a: p){
            for(Double i: a){
                System.out.print(i + " ");
            }
            System.out.println();
        }
        //net.writeParameters();
    }
}
