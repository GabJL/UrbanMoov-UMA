package es.uma.algorithms;

import es.uma.problem.TrainingWeightsProblem_Multi;
import org.uma.jmetal.algorithm.multiobjective.omopso.OMOPSO;
import org.uma.jmetal.operator.impl.mutation.NonUniformMutation;
import org.uma.jmetal.operator.impl.mutation.UniformMutation;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.evaluator.SolutionListEvaluator;
import org.uma.jmetal.util.evaluator.impl.MultithreadedSolutionListEvaluator;


public class PSO_Multi_Seq extends AbstractAlgorithm {

    public PSO_Multi_Seq(AlgorithmConfiguration algorithmConfiguration){
        super(algorithmConfiguration);
    }

    public void run()  {
        SolutionListEvaluator<DoubleSolution> evaluation;
        UniformMutation mutation;
        NonUniformMutation mutation2;
        OMOPSO algorithm;

        setProblem(new TrainingWeightsProblem_Multi(getAlgorithmConfiguration()));
        evaluation = new MultithreadedSolutionListEvaluator<>(1, getProblem());
        mutation = new UniformMutation( getAlgorithmConfiguration().getMutationProbability(),
                                        getAlgorithmConfiguration().getPerturbation());
        mutation2 = new NonUniformMutation( getAlgorithmConfiguration().getMutationProbability(),
                                            getAlgorithmConfiguration().getPerturbation(),
                                            getAlgorithmConfiguration().getMaxIterationsMutation());


        algorithm = new OMOPSO(getProblem(), evaluation, getAlgorithmConfiguration().getSwarmSize(),
                                getAlgorithmConfiguration().getMaxIterations(),
                                getAlgorithmConfiguration().getArchiveSize(), mutation, mutation2);

        algorithm.run();

        setSols(algorithm.getResult());
    }

}

