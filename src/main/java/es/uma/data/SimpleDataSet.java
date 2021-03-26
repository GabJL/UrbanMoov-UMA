package es.uma.data;

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.*;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;

public class SimpleDataSet {

    private DataSetIterator trainingData;
    private DataSetIterator testData;
    private String path;
    private String file;
    private int nbExamples;
    private int period; // Previous time steps needed to calculated current one
    private boolean headers;
    private double trainingSize;

    public SimpleDataSet(String path, String file, boolean headers){
        trainingData = testData = null;
        this.path = path;
        this.file = file;
        this.period = -1;
        this.headers = headers;
        this.nbExamples = -1;
        this.setTrainingSize(0.9);

    }
    public SimpleDataSet(String path, String file){
        this(path, file, false);
    }

    public void generateFiles(int period, boolean regenerate) throws IOException {
        if(this.period != period || regenerate){
            File f = new File(path+"cache/");
            if(f.exists()) {
                Files.walk(FileSystems.getDefault().getPath(path + "cache/"))
                        .sorted(Comparator.reverseOrder())
                        .map(Path::toFile)
                        .forEach(File::delete);
            }
            f.mkdir();
        } else return;
        this.period = period;
        BufferedReader bf = new BufferedReader(new FileReader(path+file));
        String titles = bf.readLine();
        String [] s = new String[period+1];
        for (int i = period; i >= 0; i--){
            s[i] = bf.readLine();
        }
        int pos = 0;
        while(s[0] != null && !s[0].equals("")){
            PrintWriter pI = new PrintWriter(new FileWriter(path+"cache/i"+pos+".csv"));
            PrintWriter pO = new PrintWriter(new FileWriter(path+"cache/o"+pos+".csv"));
            for(int i = period; i > 0; i--){
                pI.println(s[i]); s[i] = s[i-1];
            }
            /// TEST
            /// TEST : Original
            // String [] ss = s[0].split(",");
            // pO.println(ss[ss.length-1]);
            /// TEST : test
            pO.println(s[0]);

            pI.close();
            pO.close();
            s[0] = bf.readLine();
            pos++;
        }
        nbExamples = pos;
        bf.close();
    }

    public void generateFiles(int period) throws IOException {
        generateFiles(period, false);
    }

    public double getTrainingSize() {
        return trainingSize;
    }

    public void setTrainingSize(double trainingSize) {
        if(trainingSize != this.trainingSize){
            trainingData = testData = null;
        }
        this.trainingSize = trainingSize;
    }

    public DataSetIterator getTrainingData() throws IOException, InterruptedException {
        if(trainingData == null) {
            int numLinesToSkip = 0;
            String delimiter = ",";
            int NB_TRAIN_EXAMPLES = (int) Math.round(nbExamples*trainingSize);

            // Load training data

            CSVSequenceRecordReader trainFeatures = new CSVSequenceRecordReader(numLinesToSkip, delimiter);
            trainFeatures.initialize( new NumberedFileInputSplit( path+"cache/i%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

            CSVSequenceRecordReader trainLabels = new CSVSequenceRecordReader();
            trainLabels.initialize(new NumberedFileInputSplit(path+"cache/o%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

            trainingData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
                    32, 0, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        }
        return trainingData;
    }

    public DataSetIterator getTestData() throws IOException, InterruptedException {
        if(testData == null){
            int numLinesToSkip = 0;
            String delimiter = ",";
            int NB_TRAIN_EXAMPLES = (int) Math.round(nbExamples*trainingSize);
            int NB_TEST_EXAMPLES = nbExamples - NB_TRAIN_EXAMPLES;

            CSVSequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
            testFeatures.initialize(new NumberedFileInputSplit(path+"cache/i%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_TEST_EXAMPLES - 1));

            CSVSequenceRecordReader testLabels = new CSVSequenceRecordReader();
            testLabels.initialize(new NumberedFileInputSplit(path+"cache/o%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES  + NB_TEST_EXAMPLES - 1));

            testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
                    32, 0, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        }
        return testData;
    }
}
