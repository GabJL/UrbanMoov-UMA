package es.uma.algorithms;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;

public class RecurrentNeuralNetwork {

    private static int default_seed = 12345;
    private static double default_learningrate = 0.005;
    private static int default_nb_epoch = 1000;

    private int [] layers;
    private int seed;
    private double learningrate;
    public MultiLayerNetwork net;

    private double MSE;
    private double MAE;
    private double MaxE;
    private double CL;

    public RecurrentNeuralNetwork(int [] layers, int seed, double learningrate){
        this.setLayers(layers);
        this.seed = seed;
        this.learningrate = learningrate;
        this.net = null;
        this.MSE = -1.;
        this.MAE = -1.;
        this.MaxE = -1.;
        this.CL = -1.;
    }

    public RecurrentNeuralNetwork(int [] layers, double learningrate){
        this(layers, default_seed, learningrate);
    }

    public RecurrentNeuralNetwork(int [] layers, int seed){
        this(layers, seed, default_learningrate);
    }

    public RecurrentNeuralNetwork(int [] layers){
        this(layers, default_seed, default_learningrate);
    }

    public void build(int report_step){
        // some common parameters
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.seed(seed);
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        builder.weightInit(WeightInit.XAVIER);
        builder.updater(new Adam(learningrate));

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        for(int i = 0; i < getLayers().length-2; i++) {
            LSTM.Builder hiddenLayerBuilder = new LSTM.Builder();
            hiddenLayerBuilder.nIn(getLayers()[i]);
            hiddenLayerBuilder.nOut(getLayers()[i + 1]);
            // adopted activation function from LSTMCharModellingExample
            // seems to work well with RNNs
            hiddenLayerBuilder.activation(Activation.TANH);
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }
        // we need to use RnnOutputLayer for our RNN
        //RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT);
        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE);
        // softmax normalizes the output neurons, the sum of all outputs is 1
        // this is required for our sampleFromDistribution-function
        //outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.activation(Activation.IDENTITY);
        outputLayerBuilder.nIn(getLayers()[getLayers().length-2]);
        outputLayerBuilder.nOut(getLayers()[getLayers().length-1]);
        listBuilder.layer(getLayers().length-2, outputLayerBuilder.build());

        // create network
        MultiLayerConfiguration conf = listBuilder.build();
        net = new MultiLayerNetwork(conf);
        net.init();
        if(report_step > 0)
            net.setListeners(new ScoreIterationListener(report_step));
    }

    public void build(){
        build(-1);
    }

    public int getNumberOfParameters(){

        Map<String, INDArray> paramTable = net.paramTable(true);
        Set<String> keys = paramTable.keySet();
        int params = 0;

        for (String key : keys) {
            INDArray values = paramTable.get(key);
            long [] v = values.shape();
            params += (v[0]*v[1]);
        }
        return params;
    }

    public void setParameters(double [] weights){

        Map<String, INDArray> paramTable = net.paramTable(true);
        Set<String> keys = paramTable.keySet();
        int init = 0;
        int end = 0;

        for (String key : keys) {
            INDArray values = paramTable.get(key);
            long [] v = values.shape();
            end += (v[0]*v[1]);
            double [] w = Arrays.copyOfRange(weights, init, end);
            net.setParam(key, Nd4j.create(w, values.shape()));
            init = end;
        }
    }

    public double []  getParameters(){
        double [] weights = new double[getNumberOfParameters()];
        Map<String, INDArray> paramTable = net.paramTable(true);
        Set<String> keys = paramTable.keySet();
        int init = 0;

        for (String key : keys) {
            INDArray values = Nd4j.toFlattened(paramTable.get(key));
            for(double d: values.toDoubleVector()){
                weights[init] = d;
                init++;
            }
        }
        return weights;
    }


    public void writeParameters(){
        Map<String, INDArray> paramTable = net.paramTable(true);
        Set<String> keys = paramTable.keySet();

        for (String key : keys) {
            INDArray values = paramTable.get(key);
            System.out.print(key + " ");//print keys
            System.out.println(Arrays.toString(values.shape()));//print shape of INDArray
            System.out.println(values);
        }
    }

    public void train(DataSetIterator ds, int nb_epoch){
        net.fit(ds, nb_epoch);
        this.MSE = -1.;
        this.MAE = -1.;
        this.MaxE = -1.;
        this.CL = -1.;
    }

    public void train(DataSetIterator ds){
        train(ds, default_nb_epoch);
    }

    public void test(DataSetIterator ds){
        test(ds, true);
    }

    public void test(DataSetIterator ds, Boolean round){
        MSE = 0.;
        MAE = 0.;
        MaxE = 0.;
        double MaxC = 0.;
        int count = 0;

        while (ds.hasNext()) {
            DataSet batch = ds.next(1);
            INDArray output = net.output(batch.getFeatures());
            for(int j = 0; j < getLayers()[getLayers().length-1]; j++) {
                if(round) {
                    long pre = Math.round(
                            (output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(output.size(2) - 1))).getDouble(0)
                    );
                    long cor = Math.round(
                            (batch.getLabels().get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(batch.getLabels().size(2) - 1))).getDouble(0)
                    );
                    //System.out.println(pre + " vs " + cor + " -> " + (cor-pre));
                    MSE += Math.pow((cor - pre), 2);
                    MAE += Math.abs((cor - pre));
                    if (Math.abs((cor - pre)) > MaxE) MaxE = 100 - Math.abs((cor - pre));
                    double corD = (batch.getLabels().get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(batch.getLabels().size(2) - 1))).getDouble(0);
                    if(cor > MaxC) MaxC = corD;
                } else {
                    Double pre = (output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(output.size(2) - 1))).getDouble(0);
                    Double cor = (batch.getLabels().get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(batch.getLabels().size(2) - 1))).getDouble(0);
                    //System.out.println(pre + " vs " + cor + " -> " + (cor-pre));
                    MSE += Math.pow((cor - pre), 2);
                    MAE += Math.abs((cor - pre));
                    if(cor > MaxC) MaxC = cor;
                    if (Math.abs((cor - pre)) > MaxE) MaxE = 100 - Math.abs((cor - pre));
                }
                count++;
            }
        }
        ds.reset();
        //if(count == 0) System.out.println("It doesn't be 0!!!");
        MSE = MSE/count;
        MAE = MAE/count;
        CL = (MaxC - MSE)/MaxC;
        CL = Math.max(CL, 0.6);

    }

    public ArrayList<ArrayList<Double>> predict(DataSetIterator ds, int number){
        INDArray output = null;
        while (ds.hasNext()) {
            DataSet batch = ds.next(1);
            output = net.output(batch.getFeatures());
        }
        ArrayList<ArrayList<Double>> prediction = new ArrayList<>();
        for (int i = 0; i < number; i++){
            ArrayList<Double> list = new ArrayList<>();
            output = net.rnnTimeStep(output);
            for(int j = 0; j < getLayers()[getLayers().length-1]; j++){
                //long pre = Math.round(
                double pre = (output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(output.size(2) - 1))).getDouble(j);
                //);
                if(pre < 0) pre = 0;
                list.add(pre);
            }
            prediction.add(list);
        }
        ds.reset();
        return prediction;
    }

    public double getMSE() {
        return MSE;
    }

    public double getMAE() {
        return MAE;
    }

    public double getMaxE() {
        return MaxE;
    }

    public int[] getLayers() {
        return layers;
    }

    public void setLayers(int[] layers) {
        this.layers = layers;
    }

    public double getCL() {
        return CL;
    }

    public void setCL(double CL) {
        this.CL = CL;
    }
}
