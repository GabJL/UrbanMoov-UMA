package es.uma.main;

import com.google.gson.Gson;
import es.uma.algorithms.*;
import es.uma.auxiliar.CSVBuilder;
import es.uma.models.Error;
import es.uma.models.*;
import org.bson.Document;

import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.time.temporal.TemporalAccessor;
import java.util.ArrayList;

public class mainLoop {
    public static void main(String [] argv){
        /// TEST: Values for testing
        /* queueCommunication queue = new queueCommunication(
                "invitado", // System.getenv("UM_QUEUE_USER"),
                "invitado", //System.getenv("UM_QUEUE_PASS"),
                "127.0.0.1", //System.getenv("UM_QUEUE_HOST"),
                "/", //System.getenv("UM_QUEUE_VHOST"),
                5672 //Integer.parseInt(System.getenv("UM_QUEUE_PORT"))
        );*/
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
            CSVBuilder csv = new CSVBuilder("temporal", "test.csv", data, requestEvent.model.attributes, requestEvent.model.period);
            csv.writeFile();
            // Call algorithm
            //ArrayList<ArrayList<Integer>> resAlg =
            ml = executeAlgorithm(requestEvent, "temporal", "test.csv", data.size());
        }else {
            ml = loadModel(requestEvent, db);
            if(ml == null){
                return createError(400, "Error building existing model", requestEvent);
            }
            if(requestEvent.operation.equals("PREDICT")){
                // Connect to database with event and get data
                ArrayList<ArrayList<Document>> data = connectDB(ml.getRequestEvent());
                if(data == null || data.isEmpty()){
                    System.out.println("Invalid datasources or no datasources in request event");
                    return null;
                }
                /// Real Call
                // generate CSV
                CSVBuilder csv = new CSVBuilder("temporal", "test.csv", data, ml.getRequestEvent().model.attributes, ml.getRequestEvent().model.period);
                csv.writeFile();
                // Calculate result
                ArrayList<Document> result;
                ArrayList< ArrayList <Double>> resAlg = ml.getPrediction();

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
        AlgorithmModel am = new AlgorithmModel();
        am.request = ml.getRequestEvent();
        am.layers = ml.getNet().getLayers();
        am.modelID = ml.getModelID();
        am.weights = ml.getNet().getParameters();
        Gson parser = new Gson();
        Document doc = Document.parse(parser.toJson(am));
        db.setModel(collection, doc);
    }

    private static TrainedModel loadModel(PredictionRequestEvent requestEvent, databaseAccess db) {
        if(requestEvent.model.modelID == null) return null;
        Document doc = db.getModel(requestEvent.modelsource.collection, requestEvent.model.modelID);
        if(doc == null) return null;
        AlgorithmModel am = null;
        try{
            Gson g = new Gson();
            am = g.fromJson(doc.toJson(), AlgorithmModel.class);
        } catch(Exception e){
            return null;
        }
        PredictionRequestEvent re_original = am.request;
        AlgorithmConfiguration ac = generateConfiguration(requestEvent, "temporal", "test.csv",0);
        ac.setWeights(am.weights);
        ac.setLayers(am.layers);
        if(requestEvent.operation.equals("IMPROVE")){
            // Connect to database with event and get data
            ArrayList<ArrayList<Document>> data = connectDB(requestEvent);
            if(data == null || data.isEmpty()){
                System.out.println("Invalid datasources or no datasources in request event");
                return null;
            }
            // generate CSV
            CSVBuilder csv = new CSVBuilder("temporal", "test.csv", data, requestEvent.model.attributes, requestEvent.model.period);
            csv.writeFile();
        } else {
            // Connect to database with event and get data
            ArrayList<ArrayList<Document>> data = connectDB(re_original);
            if(data == null || data.isEmpty()){
                System.out.println("Invalid datasources or no datasources in request event");
                return null;
            }
            // generate CSV
            CSVBuilder csv = new CSVBuilder("temporal", "test.csv", data, re_original.model.attributes, re_original.model.period);
            csv.writeFile();
        }

        TrainedModel model = new TrainedModel(ac);
        model.setModelID(am.modelID);
        model.setRequestEvent(re_original);
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

        if(requestEvent.model.type == null){
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

        return algorithm.getModel();
        //return algorithm.getPrediction();
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
        if(output != null){
            databaseAccess db = getDB(output.source);
            if(db == null){
                System.err.println("Incorrent Datasource");
            } else {
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
        for(ArrayList<Double> ai: res){
            max_date = max_date.plusSeconds(time);
            int index = 0;
            Document doc = new Document();
            doc.put("TimeInstant",DateTimeFormatter.ISO_INSTANT.format(max_date));
            for(Double i: ai){
                doc.put(titles.get(index), i);
                index++;
            }
            result.add(doc);
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
