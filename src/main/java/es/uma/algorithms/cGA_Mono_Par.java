package es.uma.algorithms;

import es.uma.problem.TrainingWeightsProblem_Mono;
import org.uma.jmetal.algorithm.multiobjective.mocell.MOCell;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.operator.impl.crossover.SBXCrossover;
import org.uma.jmetal.operator.impl.mutation.UniformMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.archive.BoundedArchive;
import org.uma.jmetal.util.archive.impl.CrowdingDistanceArchive;
import org.uma.jmetal.util.evaluator.SolutionListEvaluator;
import org.uma.jmetal.util.evaluator.impl.MultithreadedSolutionListEvaluator;
import org.uma.jmetal.util.neighborhood.Neighborhood;
import org.uma.jmetal.util.neighborhood.impl.C9;


public class cGA_Mono_Par extends AbstractAlgorithm {

    public cGA_Mono_Par(AlgorithmConfiguration algorithmConfiguration){
        super(algorithmConfiguration);
    }

    public void run()  {
        SolutionListEvaluator<DoubleSolution> evaluation;
        BoundedArchive<DoubleSolution> archive;
        Neighborhood<DoubleSolution> neighborhood;
        CrossoverOperator crossover;
        UniformMutation mutation;
        SelectionOperator selectionOperator;
        MOCell algorithm;

        setProblem(new TrainingWeightsProblem_Mono(getAlgorithmConfiguration()));
        evaluation = new MultithreadedSolutionListEvaluator<>(getAlgorithmConfiguration().getNumberOfThreads(), getProblem());
        mutation = new UniformMutation( getAlgorithmConfiguration().getMutationProbability(),
                                        getAlgorithmConfiguration().getPerturbation());
        archive = new CrowdingDistanceArchive(getAlgorithmConfiguration().getArchiveSize());
        neighborhood = new C9<>(10,10);
        crossover = new SBXCrossover(getAlgorithmConfiguration().getCrossoverProbability(),
                                    getAlgorithmConfiguration().getCrossoverDistributionIndex());
        selectionOperator = new BinaryTournamentSelection();

        algorithm = new MOCell(getProblem(), getAlgorithmConfiguration().getMaxIterations(),
                             getAlgorithmConfiguration().getPopSize(), archive, neighborhood, crossover, mutation,
                             selectionOperator, evaluation);

        algorithm.run();

        setSols(algorithm.getResult());
    }

}

