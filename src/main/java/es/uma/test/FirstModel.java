package es.uma.test;

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class FirstModel {

    public static String datapath = "data/traffic/";
    public static String datafile = "traffic.csv";
    public static String dir = "cache/";

    public static void main(String [] args) throws IOException, InterruptedException {
        Integer NB_EPOCHS = 20;

        genFiles();
        List<DataSetIterator> data = getData();

        MultiLayerNetwork net = getNetwork();

        net.fit(data.get(0), NB_EPOCHS);

        System.out.println("ROC: " + evaluate(net, data.get(1)));
    }

    public static Double evaluate(MultiLayerNetwork net, DataSetIterator  testData){
         ROC roc = new ROC(100);

         Double MSE = 0., MAE = 0.;
         Integer count = 0;

         while (testData.hasNext()) {
             DataSet batch = testData.next(1);
             INDArray output = net.output(batch.getFeatures());
             long pre = Math.round(
                     (output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(output.size(2)-1))).getDouble(0)
                                    );
             long cor = Math.round(
                     (batch.getLabels().get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(batch.getLabels().size(2)-1))).getDouble(0)
                );
             //System.out.println(pre + " vs " + cor + " -> " + (cor-pre));
             MSE += Math.pow((cor-pre), 2);
             MAE += Math.abs((cor-pre));

             count++;
             roc.evalTimeSeries(batch.getLabels(), output);
         }
         System.out.println("MSE = " + MSE/count);
         System.out.println("MAE = " + MAE/count);
         return roc.calculateAUC();
    }

    public static void genFiles() throws IOException {
        int max_time = 4;
        File f = new File(datapath+dir+"i0.csv");
        if(f.exists()) return;

        BufferedReader bf = new BufferedReader(new FileReader(datapath+datafile));
        String titles = bf.readLine();
        String [] s = new String[max_time+1];
        for (int i = max_time; i >= 0; i--){
            s[i] = bf.readLine();
        }
        int pos = 0;
        while(s[0] != null && !s[0].equals("")){
            PrintWriter pI = new PrintWriter(new FileWriter(datapath+dir+"i"+pos+".csv"));
            PrintWriter pO = new PrintWriter(new FileWriter(datapath+dir+"o"+pos+".csv"));
            for(int i = 4; i > 0; i--){
                pI.println(s[i]); s[i] = s[i-1];
            }
            String [] ss = s[0].split(",");
            pO.println(ss[ss.length-1]);
            pI.close();
            pO.close();
            s[0] = bf.readLine();
            pos++;
        }


    }

    public static List<DataSetIterator > getData() throws IOException, InterruptedException {
        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        String delimiter = ",";
        int NB_EXAMPLES = 2973;
        int NB_TRAIN_EXAMPLES = 2200;
        int NB_TEST_EXAMPLES = NB_EXAMPLES - NB_TRAIN_EXAMPLES;

        // Load training data

        CSVSequenceRecordReader trainFeatures = new CSVSequenceRecordReader(numLinesToSkip, delimiter);
        trainFeatures.initialize( new NumberedFileInputSplit( datapath+dir+"i%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

        CSVSequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(datapath+dir+"o%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

        DataSetIterator  trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
                32, 0, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        // Load testing data
        CSVSequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
        testFeatures.initialize(new NumberedFileInputSplit(datapath+dir+"i%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

        CSVSequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(datapath+dir+"o%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES  + NB_TEST_EXAMPLES - 1));

        DataSetIterator  testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
                32, 0, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        List<DataSetIterator > res = new ArrayList<DataSetIterator >();
        res.add(trainData); // training data
        res.add(testData); // test data

        return res;

    }

    public static MultiLayerNetwork getNetwork(){
        // Set neural network parameters
        Integer NB_INPUTS = 8;
        Integer RANDOM_SEED = 1234;
        Double LEARNING_RATE = 0.005;
        Integer HIDDEN_LAYER_CONT = 3;
        Integer NUM_OUTPUTS = 1;

        // some common parameters
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.seed(RANDOM_SEED);
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        builder.weightInit(WeightInit.XAVIER);
        builder.updater(new Adam(LEARNING_RATE));
//        builder.biasInit(0);
//        builder.miniBatch(false);

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();

        // first difference, for rnns we need to use LSTM.Builder
        LSTM.Builder initialLayer = new LSTM.Builder();
        initialLayer.nIn(NB_INPUTS);
        Integer layer_size = NB_INPUTS*4;
        initialLayer.nOut(layer_size);
        initialLayer.activation(Activation.TANH);
        listBuilder.layer(0, initialLayer.build());
        for (int i = 1; i <= HIDDEN_LAYER_CONT; i++) {
            LSTM.Builder hiddenLayerBuilder = new LSTM.Builder();
            hiddenLayerBuilder.nIn(layer_size);
            layer_size -= 8;
            hiddenLayerBuilder.nOut(layer_size);
            // adopted activation function from LSTMCharModellingExample
            // seems to work well with RNNs
            hiddenLayerBuilder.activation(Activation.TANH);
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }

        // we need to use RnnOutputLayer for our RNN
        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE);
        //RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT);
        // softmax normalizes the output neurons, the sum of all outputs is 1
        // this is required for our sampleFromDistribution-function
        outputLayerBuilder.activation(Activation.IDENTITY);
        //outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.nIn(layer_size);
        outputLayerBuilder.nOut(NUM_OUTPUTS);
        listBuilder.layer(HIDDEN_LAYER_CONT+1, outputLayerBuilder.build());

        // create network
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(25));

        return net;
    }

}
