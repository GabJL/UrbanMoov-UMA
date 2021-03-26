package es.uma.problem;

import es.uma.algorithms.AlgorithmConfiguration;
import org.uma.jmetal.solution.DoubleSolution;

import java.util.ArrayList;
import java.util.List;

public class TrainingWeightsProblem_Mono extends TrainingWeightsProblem {
    public TrainingWeightsProblem_Mono(AlgorithmConfiguration algorithmConfiguration) {
        super(algorithmConfiguration);
        setNumberOfVariables(getNet().getNumberOfParameters());
        setNumberOfObjectives(2);
        setName("TrainingWeightsProblemMono-objective");

        List<Double> lowerLimit = new ArrayList<>(getNumberOfVariables()) ;
        List<Double> upperLimit = new ArrayList<>(getNumberOfVariables()) ;

        for (int i = 0; i < getNumberOfVariables(); i++) {
            lowerLimit.add(-2.0);
            upperLimit.add(2.0);
        }

        setLowerLimit(lowerLimit);
        setUpperLimit(upperLimit);
    }

    /** Evaluate() method */
    public void evaluate(DoubleSolution solution){
        double[] fx = new double[getNumberOfObjectives()];
        double[] x = new double[getNumberOfVariables()];
        for (int i = 0; i < solution.getNumberOfVariables(); i++) {
            x[i] = solution.getVariableValue(i) ;
        }

        getNet().setParameters(x);
        try {
            getNet().test(getDs().getTrainingData());
            fx[0] = getNet().getMAE();
            fx[1] = getNet().getMaxE();
            if(Double.isNaN(fx[0])){
                fx[0] = 1000000;
                fx[1] = 1000000;
            }
        } catch (Exception e) {
            fx[0] = 1000000;
            fx[1] = 1000000;
        }
        solution.setObjective(0, fx[0]);
        solution.setObjective(1, fx[1]);
    }

    public void test(DoubleSolution solution){
        double[] fx = new double[getNumberOfObjectives()];
        double[] x = new double[getNumberOfVariables()];
        for (int i = 0; i < solution.getNumberOfVariables(); i++) {
            x[i] = solution.getVariableValue(i) ;
        }

        getNet().setParameters(x);
        try {
            getNet().test(getDs().getTestData());
        } catch (Exception e) {
            //e.printStackTrace();
        }
        fx[0] = getNet().getMAE();
        fx[1] = getNet().getMAE();
        if(Double.isNaN(fx[0])){
            fx[0] = 1000000;
            fx[1] = 1000000;
        }
        solution.setObjective(0, fx[0]);
        solution.setObjective(1, fx[1]);
    }
}