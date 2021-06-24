package es.uma.main;

import com.google.gson.Gson;
import es.uma.algorithms.*;
import es.uma.auxiliar.CSVBuilder;
import es.uma.data.SimpleDataSet;
import es.uma.models.Error;
import es.uma.models.*;
import org.bson.Document;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.time.temporal.TemporalAccessor;
import java.util.ArrayList;

public class mainLoop {
    public static void main(String [] argv){
        /// TEST: Values for testing
        /* queueCommunication queue = new queueCommunication(
                "uma-client", // System.getenv("UM_QUEUE_USER"),
                "cfKCWpAK3dsS9hrk", //System.getenv("UM_QUEUE_PASS"),
                "127.0.0.1", //System.getenv("UM_QUEUE_HOST"),
                "uma", //System.getenv("UM_QUEUE_VHOST"),
                15672 //Integer.parseInt(System.getenv("UM_QUEUE_PORT"))
        ); */
        queueCommunication queue = new queueCommunication(
                System.getenv("UM_QUEUE_USER"),
                System.getenv("UM_QUEUE_PASS"),
                System.getenv("UM_QUEUE_HOST"),
                System.getenv("UM_QUEUE_VHOST"),
                Integer.parseInt(System.getenv("UM_QUEUE_PORT"))
        );
        if(queue == null){
            System.err.println("Cannot connect to RabbitMQ queue");
            return;
        }
        while(true){
            IncomingMessage message = queue.getMessage();
            if(message != null) {
                String result = manageMessage(message.message);
                queue.sendMessage(result, message.deliveryTag);
            } else {
                System.err.println("Receiving null message from queue.");
            }
        }
    }

    private static String manageMessage(String message) {
        // Convert message to object
        Gson gson = new Gson();
        PredictionRequestEvent requestEvent = gson.fromJson(message, PredictionRequestEvent.class);
        if(requestEvent == null){
            System.err.println("request event format is not valid");
            return null;
        }

        if(requestEvent.model == null){
            return createError(400, "Model information is required", requestEvent);
        }

        if(requestEvent.model.attributes == null){
            return createError(400, "Attribute list is required", requestEvent);
        }

        databaseAccess db = getDB(requestEvent.modelsource);
        if(db == null){
            return createError(400, "Model datasource is required", requestEvent);
        }

        if(requestEvent.operation == null ||
                (!requestEvent.operation.equals("PREDICT") && !requestEvent.operation.equals("IMPROVE"))) 
            requestEvent.operation = "TRAIN";

        TrainedModel ml = null;
        if(requestEvent.operation.equals("TRAIN")){
            // Connect to database with event and get data
            ArrayList<ArrayList<Document>> data = connectDB(requestEvent);
            if(data == null || data.isEmpty()){
                return createError(404, "Invalid datasources or no datasources in request event", requestEvent);
            }
            /// Real Call
            // generate CSV
            CSVBuilder csv = new CSVBuilder("temporal", "test.csv", data, requestEvent.model.attributes,
                                                requestEvent.model.period, requestEvent.model.from, requestEvent.model.to);
            csv.writeFile();
            // Call algorithm
            //ArrayList<ArrayList<Integer>> resAlg =
            ml = executeAlgorithm(requestEvent, "temporal", "test.csv", csv.getTitles().size()-1);
            ml.setRequestEvent(requestEvent);
        }else {
            ml = loadModel(requestEvent, db);
            if(ml == null){
                return createError(400, "Error building existing model", requestEvent);
            }
            if(requestEvent.operation.equals("PREDICT")){
                // Connect to database with event and get data
                ArrayList<ArrayList<Document>> data = connectDB(requestEvent);
                Boolean nuevos = true;
                if(data == null || data.isEmpty()) {
                    data = connectDB(ml.getRequestEvent());
                    nuevos = false;
                    if (data == null || data.isEmpty()) {
                        System.out.println("Invalid datasources or no datasources in request event");
                        return null;
                    }
                }

                /// Real Call
                // generate CSV
                CSVBuilder csv = new CSVBuilder("temporal", "test.csv", data, ml.getRequestEvent().model.attributes,
                        requestEvent.model.period, requestEvent.model.from, requestEvent.model.to);
                csv.writeFile();
                // Calculate result
                ArrayList<Document> result;
                if(nuevos) ml.train();
                ArrayList< ArrayList <Double>> resAlg = ml.getPrediction();
                /**/ // DESDE AQUI
                /*
                System.out.println("Prediciendo");
                for(ArrayList<Double> a: resAlg){
                    for(Double d: a)
                        System.out.print(d + " ");
                    System.out.println();
                }
                System.out.println("Fin prediciendo");
                */
                /**/ // HASTA AQUÏ

                // Generate Results
                result = analyzeResultsAlgorithm(resAlg, data, requestEvent, csv.getTitles());

                if(result == null || result.isEmpty()){
                    return createError(400, "Error data", requestEvent);
                }

                // Write Database
                writeDatabase(requestEvent.output, result);
            } else {
                ml.train();
            }
        }
        writeModel(ml, db, requestEvent.modelsource.collection);
        // Create event
        String modelID = ml.getModelID();
        Double cL = ml.getConfidenceLevel();
        String res = createExecutedEvent(requestEvent, true, null, modelID, cL);
        return res;
    }

    private static void writeModel(TrainedModel ml, databaseAccess db, String collection) {
        // /**/ System.out.println("test: Escribiendo modelo ...");
        AlgorithmModel am = new AlgorithmModel();
        am.request = ml.getRequestEvent();
        am.layers = ml.getNet().getLayers();
        am.modelID = ml.getModelID();
        am.weights = ml.getNet().getParameters();
        Gson parser = new Gson();
        Document doc = Document.parse(parser.toJson(am));
        // /**/ System.out.println("test: Escribiendo modelo ...");
        db.setModel(collection, doc);
    }

    private static TrainedModel loadModel(PredictionRequestEvent requestEvent, databaseAccess db) {
        // /**/ System.out.println("Loading model 1");
        if(requestEvent.model.modelID == null) return null;
        // /**/ System.out.println("Loading model 2");
        Document doc = db.getModel(requestEvent.modelsource.collection, requestEvent.model.modelID);
        if(doc == null) return null;
        // /**/ System.out.println("Loading model 3");
        AlgorithmModel am = null;
        try{
            Gson g = new Gson();
            am = g.fromJson(doc.toJson(), AlgorithmModel.class);
        } catch(Exception e){
            return null;
        }
        // /**/ System.out.println("Loading model 4");
        PredictionRequestEvent re_original = am.request;
        AlgorithmConfiguration ac = generateConfiguration(requestEvent, "temporal", "test.csv",0);
        ac.setWeights(am.weights);
        ac.setLayers(am.layers);
        // /**/ System.out.println("Loading model 5");
        if(requestEvent.operation.equals("IMPROVE")){
            // Connect to database with event and get data
            ArrayList<ArrayList<Document>> data = connectDB(requestEvent);
            if(data == null || data.isEmpty()){
                System.out.println("Invalid datasources or no datasources in request event");
                return null;
            }
            // generate CSV
            CSVBuilder csv = new CSVBuilder("temporal", "test.csv", data, requestEvent.model.attributes,
                    requestEvent.model.period, requestEvent.model.from, requestEvent.model.to);
            csv.writeFile();
        } else {
            // /**/ System.out.println("Loading model 6");
            // Connect to database with event and get data
            ArrayList<ArrayList<Document>> data = connectDB(requestEvent);
            if(data == null || data.isEmpty()) {
                data = connectDB(re_original);
                if (data == null || data.isEmpty()) {
                    System.out.println("Invalid datasources or no datasources in request event");
                    return null;
                }
            }
            // /**/ System.out.println("Loading model 7");
            // generate CSV
            CSVBuilder csv = new CSVBuilder("temporal", "test.csv", data, re_original.model.attributes,
                    requestEvent.model.period, requestEvent.model.from, requestEvent.model.to);
            csv.writeFile();
            /**/ // System.out.println("Loading model 8");

        }
        // /**/ System.out.println("Loading model 9");
        TrainedModel model = new TrainedModel(ac);
        model.setModelID(requestEvent.model.modelID);
        model.setRequestEvent(re_original);
        // /**/ System.out.println("Loading model 10");
        return model;
    }

    private static String createError(int status, String msg, PredictionRequestEvent requestEvent){
        Error e = new Error();
        e.status = status;
        e.msg = msg;
        String res = createExecutedEvent(requestEvent, false, e, null, null);
        return res;
    }

    private static AlgorithmConfiguration generateConfiguration(PredictionRequestEvent requestEvent, String dp,
                                                                String fn, Integer inputs){

        // /**/ System.out.println("Numero de inputs: " + inputs);
        AlgorithmConfiguration ac = new AlgorithmConfiguration();
        ac.setDatapath(dp);
        ac.setDatafile(fn);
        ac.setLayers(new int[]{inputs, inputs*8, inputs * 4, inputs * 2, inputs});
        ac.setWeights(null);
        ac.setPrediction(requestEvent.model.horizon);
        ac.setPredictionPeriod(requestEvent.model.period);
        return ac;
    }

    private static TrainedModel executeAlgorithm(PredictionRequestEvent requestEvent, String dp,
                                                                  String fn, Integer inputs) {
        AlgorithmConfiguration ac = generateConfiguration(requestEvent, dp, fn, inputs);

        AbstractAlgorithm algorithm = null;

        if (requestEvent.model.type == null) {
            algorithm = new PSO_Multi_Par(ac);
        } else {
            if (requestEvent.model.type.equals("PSOMOPAR")) {
                algorithm = new PSO_Multi_Par(ac);
            } else if (requestEvent.model.type.equals("PSOMOSEQ")) {
                algorithm = new PSO_Multi_Seq(ac);
            } else if (requestEvent.model.type.equals("PSOSOPAR")) {
                algorithm = new PSO_Mono_Par(ac);
            } else if (requestEvent.model.type.equals("PSOSOSEQ")) {
                algorithm = new PSO_Mono_Seq(ac);
            } else if (requestEvent.model.type.equals("CGAMOPAR")) {
                algorithm = new cGA_Multi_Par(ac);
            } else if (requestEvent.model.type.equals("CGAMOSEQ")) {
                algorithm = new cGA_Multi_Seq(ac);
            } else if (requestEvent.model.type.equals("CGASOPAR")) {
                algorithm = new cGA_Mono_Par(ac);
            } else if (requestEvent.model.type.equals("CGASOSEQ")) {
                algorithm = new cGA_Mono_Seq(ac);
            } else if (requestEvent.model.type.equals("ACOMOPAR")) {
                algorithm = new ACO_Multi_Par(ac);
            } else if (requestEvent.model.type.equals("ACOMOSEQ")) {
                algorithm = new ACO_Multi_Seq(ac);
            } else if (requestEvent.model.type.equals("ACOSOPAR")) {
                algorithm = new ACO_Mono_Par(ac);
            } else if (requestEvent.model.type.equals("ACOSOSEQ")) {
                algorithm = new ACO_Mono_Seq(ac);
            } else {
                algorithm = new PSO_Mono_Par(ac);
            }
        }

        algorithm.run();

        /**/ // Aditional model
        TrainedModel ml = new TrainedModel(ac), ml1 = algorithm.getModel(), ml2 = new TrainedModel(ac);
        try {
            ml.setNet(newModel(ac.getLayers()));
            ml.getAlgorithmConfiguration().setWeights(ml.getNet().getParameters());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        ml2.setRequestEvent(requestEvent);
        ml2.train();
        if (ml.getConfidenceLevel() >= ml1.getConfidenceLevel() && ml.getConfidenceLevel() >= ml2.getConfidenceLevel()) {
            System.out.println("New model selected");
            return ml;
        } else if (ml2.getConfidenceLevel() >= ml1.getConfidenceLevel()){
            System.out.println("Old model selected");
            return ml2;
        }
        System.out.println("Alg model selected");
        return ml1;

        //return algorithm.getPrediction();
    }

    private static RecurrentNeuralNetwork newModel(int [] layers) throws IOException, InterruptedException {
        SimpleDataSet ds = new SimpleDataSet("temporal/", "test.csv", true);

        ds.generateFiles(16);
        ds.setTrainingSize(0.75);

        DataSetIterator trainingData = ds.getTrainingData();
        DataSetIterator testData = ds.getTestData();

        RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(layers);

        net.build(4000);
        net.train(trainingData);
        return net;
    }

    private static String createExecutedEvent(PredictionRequestEvent r, boolean ack, Error e, String modelID,
                                              Double confL) {
        PredictionExecutionEvent p = new PredictionExecutionEvent();
        p.id = r.id;
        p.timestamp = "" + Instant.now().getEpochSecond();
        p.name = "PredictionExecutedEvent";
        p.version = "1";
        p.ack = ack;
        p.etlId = r.executionId;
        p.executionId = r.executionId;
        p.userId = r.executionId;
        p.modelID = modelID;
        p.error = e;
        p.ConfidenceLevel = confL;
        Gson gson = new Gson();
        String json = gson.toJson(p);
        return json;
    }

    private static databaseAccess getDB(Source sc){
        if(sc == null){
            return null;
        }
        if(sc.collection != null && sc.host != null && sc.name != null && sc.port != null) {
            //  mirar si hay user y pass
            //  Llamar al constructor apropiado
            databaseAccess db = null;
            if(sc.username == null || sc.password == null) {
                db = new databaseAccess(sc.host, sc.port, sc.name);
            } else {
                db = new databaseAccess(sc.host, sc.port, sc.name, sc.username, sc.password);
            }
            return db;
        }
        return null;
    }

    private static void writeDatabase(Output output, ArrayList<Document> result) {
        // /**/ System.out.println("test: writing predictions");
        if(output != null){
            databaseAccess db = getDB(output.source);
            if(db == null){
                System.err.println("Incorrent Datasource");
            } else {
                // /**/ System.out.println("test: writing predictions data");
                db.setData(output.source.collection, result);
            }
        }
    }

    private static ArrayList<Document> analyzeResultsAlgorithm(ArrayList<ArrayList<Double>> res,
                                                               ArrayList<ArrayList<Document>> data,
                                                               PredictionRequestEvent requestEvent,
                                                               ArrayList<String> titles) {

        int time = 15*60;

        switch (requestEvent.model.period) {
            case "15M":
                time = 15 * 60;
                break;
            case "30M":
                time = 30 * 60;
                break;
            case "60M":
                time = 60 * 60;
                break;
            case "1D":
                time = 24 * 60 * 60;
                break;
        }
        /**/ // System.out.println("periodo: " + requestEvent.model.period + " time: " + time + " values: " + data.size());
        if(requestEvent.model.usecase != null && requestEvent.model.usecase.equals("PARKING")) {
            res = postprocessing(res);
            titles.add("global");
        }

        Instant date = null, max_date = null;
        ArrayList<Document> result = new ArrayList<>();
        if(requestEvent.model.to != null){
            max_date = Instant.from(DateTimeFormatter.ISO_INSTANT.parse(CSVBuilder.convertDate(requestEvent.model.to)));
        } else {
            for(ArrayList<Document> ad: data){
                for(Document d: ad){
                    Document aux = (Document) d.get("TimeInstant");
                    String s = CSVBuilder.convertDate(aux.get("value", String.class));
                    // /**/ System.out.println("test Fecha:" + s);
                    TemporalAccessor ta = DateTimeFormatter.ISO_INSTANT.parse(s);
                    date = Instant.from(ta);
                    if(max_date == null){
                        max_date = date;
                    } else if(date.compareTo(max_date) > 0){
                        max_date = date;
                    }
                    // /**/ System.out.println("Max date: " + DateTimeFormatter.ISO_INSTANT.format(max_date));
                }
            }
        }
        for(ArrayList<Double> ai: res){
            max_date = max_date.plusSeconds(time);
            int index = 1;
            Document doc = new Document();
            doc.put("TimeInstant",DateTimeFormatter.ISO_INSTANT.format(max_date).replace('T',' ').replace("Z",".0"));
            // /**/ System.out.println("TimeInstant " + DateTimeFormatter.ISO_INSTANT.format(max_date));
            // /**/ System.out.println("ai " + ai.toString());
            for(Double i: ai){
                doc.put(titles.get(index), i);
                index++;
            }
            result.add(doc);
        }

        return result;
    }

    private static ArrayList<ArrayList<Double>> postprocessing(ArrayList<ArrayList<Double>> res) {
        ArrayList<ArrayList<Double> > result = new ArrayList<>();
        for(ArrayList<Double> a: res){
            double suma = 0.0;
            int cont = 0;
            ArrayList<Double> aux = new ArrayList<>();
            for(Double d: a){
                if(d < 0.5) aux.add(0.0);
                else        { aux.add(1.0); suma++; }
                cont++;
            }
            aux.add(suma/cont);
            result.add(aux);
        }
        return result;
    }

    private static ArrayList<Document> analyzeResults(ArrayList<ArrayList<Document>> data,
                                                      PredictionRequestEvent requestEvent) {
        Instant date = null, max_date = null;
        ArrayList<Document> result = new ArrayList<>();
        for(ArrayList<Document> ad: data){
            for(Document d: ad){
                String s = d.get("TimeInstant", String.class);
                TemporalAccessor ta = DateTimeFormatter.ISO_INSTANT.parse(s);
                date = Instant.from(ta);
                if(max_date == null){
                    max_date = date;
                } else if(date.compareTo(max_date) > 0){
                    max_date = date;
                }
            }
        }
        Integer d = 0;
        Integer o = 0;
        for(ArrayList<Document> ad: data){
            Instant max_date1 = max_date;
            for(Document doc: ad) {
                Document doc1 = new Document();
                doc.put("Datasource", requestEvent.origins.get(o).datasources.get(d).source.collection);
                max_date1 = max_date.plusSeconds(240);
                doc.put("TimeInstant",DateTimeFormatter.ISO_INSTANT.format(max_date1));
                try {
                    Integer i = d.getInteger("per");
                    i += 2;
                    doc1.put("per", i);
                } catch(Exception e){
                    doc1.put("per", 0);
                }
                result.add(doc1);
            }
            d++;
            if(d >= requestEvent.origins.get(o).datasources.size()){
                d = 0;
                o++;
            }

        }

        return result;
    }

    private static ArrayList<ArrayList<Document>> connectDB(PredictionRequestEvent requestEvent) {
        ArrayList<ArrayList<Document>> data = new ArrayList<>();
        // Comprobar atributos
        if(requestEvent.origins == null){
            return data;
        }
        for(Origins o: requestEvent.origins){
            if(o.datasources != null){
                for(DataSource s: o.datasources) {
                    if (s != null) {
                        databaseAccess db = getDB(s.source);
                        if (db == null) {
                            System.err.println("Incorrent Datasource");
                        } else {
                            data.add(db.getData(s.source.collection, requestEvent.model.from, requestEvent.model.to));
                        }
                    }
                }
            }
        }
        return data;
    }
}
