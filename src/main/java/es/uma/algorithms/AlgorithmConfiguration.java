package es.uma.algorithms;

public class AlgorithmConfiguration {
    // Prediction Range
    public enum Prediction {
        SHORT,
        MEDIUM,
        LARGE
    };

    // Prediction period
    public enum Period {
        M15,
        M30,
        M60,
        D1
    };

    // General::Net
    private String datapath;
    private String datafile;
    private int [] layers;
    private double [] weights;
    private int period;
    private Period predictionPeriod;
    private double training;
    // General::Algorithm
    private Integer maxIterations = 10;
    private Prediction prediction;
    // Parallel
    private static Integer numberOfThreads;
    // Multiobjective
    private Integer archiveSize;
    // cGA
    private Integer popSize;
    private Double crossoverProbability;
    private Double  crossoverDistributionIndex;
    // PSO
    private Integer swarmSize;
    private Double mutationProbability ;
    private Integer maxIterationsMutation;
    private Double perturbation;

    public AlgorithmConfiguration(){
        setDatafile("test.csv");
        setDatapath("temporal");
        setLayers(new int[]{8, 16, 8, 1});
        setPeriod(8);
        setTraining(0.75);
        setMaxIterations(10);
        setPrediction(Prediction.SHORT);
        setPredictionPeriod(Period.M15);
        setWeights(null);
        // Multiobjective
        setArchiveSize(20);
        // cGA
        setPopSize(100);
        setCrossoverDistributionIndex(20.0);
        setCrossoverProbability(0.9);
        // PSO
        setSwarmSize(10);
        setMutationProbability(0.1);
        setMaxIterationsMutation(10);
        setPerturbation(0.5);
        // Parallel
        setNumberOfThreads(getSwarmSize() /2);
    }

    public static Integer getNumberOfThreads() {
        return numberOfThreads;
    }

    public static void setNumberOfThreads(Integer numberOfThreads) {
        AlgorithmConfiguration.numberOfThreads = numberOfThreads;
    }


    public String getDatapath() {
        return datapath;
    }

    public void setDatapath(String datapath) {
        this.datapath = datapath;
    }

    public String getDatafile() {
        return datafile;
    }

    public void setDatafile(String datafile) {
        this.datafile = datafile;
    }

    public int[] getLayers() {
        return layers;
    }

    public void setLayers(int[] layers) {
        this.layers = layers;
    }

    public double[] getWeights(){ return weights; }

    public void setWeights(double [] weights) { this.weights = weights; }

    public Integer getMaxIterations() {
        return maxIterations;
    }

    public void setMaxIterations(Integer maxIterations) {
        this.maxIterations = maxIterations;
    }

    public Integer getArchiveSize() {
        return archiveSize;
    }

    public void setArchiveSize(Integer archiveSize) {
        this.archiveSize = archiveSize;
    }

    public Integer getSwarmSize() {
        return swarmSize;
    }

    public void setSwarmSize(Integer swarmSize) {
        this.swarmSize = swarmSize;
    }

    public Double getMutationProbability() {
        return mutationProbability;
    }

    public void setMutationProbability(Double mutationProbability) {
        this.mutationProbability = mutationProbability;
    }

    public Integer getMaxIterationsMutation() {
        return maxIterationsMutation;
    }

    public void setMaxIterationsMutation(Integer maxIterationsMutation) {
        this.maxIterationsMutation = maxIterationsMutation;
    }

    public Double getPerturbation() {
        return perturbation;
    }

    public void setPerturbation(Double perturbation) {
        this.perturbation = perturbation;
    }

    public Prediction getPrediction() {
        return prediction;
    }

    public void setPrediction(Prediction prediction) {
        this.prediction = prediction;
    }

    public void setPrediction(String p){
        if(p == null) p = "";
        if(p.equals("MEDIUM")) prediction = Prediction.MEDIUM;
        else if(p.equals("SHORT")) prediction = Prediction.SHORT;
        else prediction = Prediction.LARGE;
    }
    public int getPeriod() { return period; }

    public void setPeriod(int period) {
        this.period = period;
    }

    public Period getPredictionPeriod() { return predictionPeriod; }

    public void setPredictionPeriod(Period period) {
        this.predictionPeriod = period;
    }

    public void setPredictionPeriod(String p){
        if(p == null) p = "";
        if(p.equals("30M")) predictionPeriod = Period.M30;
        else if(p.equals("60M")) predictionPeriod = Period.M60;
        else if(p.equals("1D")) predictionPeriod = Period.D1;
        else predictionPeriod = Period.M15;
    }

    public double getTraining() {
        return training;
    }

    public void setTraining(double training) {
        this.training = training;
    }

    public Integer getPopSize() {
        return popSize;
    }

    public void setPopSize(Integer popSize) {
        this.popSize = popSize;
    }
    public Double getCrossoverProbability() {
        return crossoverProbability;
    }

    public void setCrossoverProbability(Double crossoverProbability) {
        this.crossoverProbability = crossoverProbability;
    }

    public Double getCrossoverDistributionIndex() {
        return crossoverDistributionIndex;
    }

    public void setCrossoverDistributionIndex(Double crossoverDistributionIndex) {
        this.crossoverDistributionIndex = crossoverDistributionIndex;
    }

}
