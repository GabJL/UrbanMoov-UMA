package es.uma.test;

import es.uma.algorithms.RecurrentNeuralNetwork;
import es.uma.data.SimpleDataSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.ArrayList;

public class testSimpleRNN {

    public static String datapath = "data/ble/";
    public static String datafile = "ble.csv";

    public static void main(String [] args) throws IOException, InterruptedException {
        SimpleDataSet ds = new SimpleDataSet(datapath, datafile, true);

        ds.generateFiles(16);
        ds.setTrainingSize(0.75);

        DataSetIterator trainingData = ds.getTrainingData();
        DataSetIterator testData = ds.getTestData();
//        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
//                normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
//        trainingData.setPreProcessor(normalizer);
//        testData.setPreProcessor(normalizer);
        int [] layers = {8, 64, 32, 24, 8};

        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(layers);

        net.build(40);
        // net.writeParameters();
        System.out.println("Number of parameters: " + net.getNumberOfParameters());
        //double [] w = new double[net.getNumberOfParameters()];
        //Random r = new Random();
        //for(int i = 0; i < w.length;i++)
        //    w[i] = r.nextDouble()*4 - 2;
        net.train(trainingData);
        double [] d = net.getParameters();
        int cont = 0;
        for(double v: d) {
            if(cont < 2500)
                System.out.print(v+" ");
            cont++;
        }
        System.out.println();
        //net.writeParameters();

        INDArray output = null;
        while (testData.hasNext()) {
            DataSet batch = testData.next(1);
            System.out.print("I: ");
            for(int j = 0; j < net.getLayers()[net.getLayers().length-1]; j++){
                System.out.print(((batch.getFeatures().get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(batch.getFeatures().size(2) - 1))).getDouble(j)) + " ");
            }
            System.out.println();
            System.out.print("O: ");
            for(int j = 0; j < net.getLayers()[net.getLayers().length-1]; j++){
                System.out.print((batch.getLabels().get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(batch.getLabels().size(2) - 1))).getDouble(j) + " ");
            }

            System.out.println();
            System.out.print("P1: ");
            output = net.net.output(batch.getFeatures());
            for(int j = 0; j < net.getLayers()[net.getLayers().length-1]; j++){
                System.out.print(((output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(output.size(2) - 1))).getDouble(j)) + " ");
            }
            System.out.println();
            System.out.println(output);
            System.out.print("NEXT1: ");
            output = net.net.rnnTimeStep(output);
            for(int j = 0; j < net.getLayers()[net.getLayers().length-1]; j++){
                System.out.print(((output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(output.size(2) - 1))).getDouble(j)) + " ");
            }
            System.out.println();
            System.out.println(output);
            System.out.print("NEXT2: ");
            output = net.net.rnnTimeStep(batch.getFeatures());
            for(int j = 0; j < net.getLayers()[net.getLayers().length-1]; j++){
                System.out.print(((output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(output.size(2) - 1))).getDouble(j)) + " ");
            }
/*            System.out.println();
            System.out.print("V2: ");
            normalizer.revert(batch);
            for(int j = 0; j < net.getLayers()[net.getLayers().length-1]; j++){
                System.out.print(((batch.getFeatures().get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(batch.getFeatures().size(2) - 1))).getDouble(j)) + " ");
            }
 */
            System.out.println();
            System.out.println(output);
        }
        System.out.println(output);
        System.out.println("10 más");
        for (int i = 0; i < 10; i++){
            output = net.net.output(output);
            for(int j = 0; j < net.getLayers()[net.getLayers().length-1]; j++){
                System.out.print(((output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0))).getDouble(j)) + " ");
//                    System.out.print(((output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(output.size(2) - 1))).getDouble(j)) + " ");
            }
            output = net.net.rnnTimeStep(output);
            System.out.println();
        }
        testData.reset();
/*
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
 */
        //net.writeParameters();
    }
}
