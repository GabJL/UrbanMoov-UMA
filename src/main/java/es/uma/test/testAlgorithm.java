package es.uma.test;

import com.google.gson.Gson;
import es.uma.algorithms.*;
import es.uma.main.databaseAccess;
import es.uma.main.queueCommunication;
import es.uma.models.*;
import org.bson.Document;

import java.time.Duration;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.time.temporal.TemporalAccessor;
import java.util.ArrayList;

public class testAlgorithm {
    public static void main(String [] argv){
        runAlgorithm();
    }

    private static void runAlgorithm() {
        String datapath = "data/ble1";
        String datafile = "ble1.csv";
        Integer inputs = 1;
        executeAlgorithm(datapath, datafile, inputs);
    }

    private static void executeAlgorithm(String dp, String fn, Integer inputs) {
        AlgorithmConfiguration ac = new AlgorithmConfiguration();
        ac.setDatapath(dp);
        ac.setDatafile(fn);
        ac.setLayers(new int[]{inputs, inputs*8, inputs * 4, inputs * 2, inputs});
        AbstractAlgorithm algorithm = null;
        algorithm = new PSO_Multi_Par(ac);
        // algorithm = new PSO_Multi_Par(ac);
        // algorithm = new PSO_Multi_Seq(ac);
        // algorithm = new PSO_Mono_Par(ac);
        // algorithm = new PSO_Mono_Seq(ac);
        // algorithm = new cGA_Multi_Par(ac);
        // algorithm = new cGA_Multi_Par(ac);
        // algorithm = new cGA_Mono_Par(ac);
        // algorithm = new cGA_Mono_Seq(ac);
        // algorithm = new ACO_Multi_Par(ac);
        // algorithm = new ACO_Multi_Par(ac);
        // algorithm = new ACO_Mono_Par(ac);
        // algorithm = new ACO_Mono_Seq(ac);
        Instant start = Instant.now();
        algorithm.run();
        Instant finish = Instant.now();
        long timeElapsed = Duration.between(start, finish).toMillis();
        System.out.println(algorithm.getMSE()+ " " + timeElapsed);
    }

}
