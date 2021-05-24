package es.uma.main;
import com.mongodb.*;
import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoDatabase;
import org.bson.Document;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Set;

import static com.mongodb.client.model.Filters.*;

public class databaseAccess {
    private MongoDatabase mongoDB;

    public databaseAccess(String host,  Integer port, String db_name, String user, String passwd){
        String encoded_pwd = "";
        try {
            encoded_pwd = URLEncoder.encode(passwd, "UTF-8");
        } catch (UnsupportedEncodingException ex) {
            System.err.println("Cannot connect to DB:" + ex);
        }

        // Mongodb connection string.
        String client_url = "mongodb://" + user + ":" + encoded_pwd + "@" + host + ":" + port + "/" + db_name;

        connect(client_url, db_name);
    }

    public databaseAccess(String host,  Integer port, String db_name){

        // Mongodb connection string.
        String client_url = "mongodb://" + host + ":" + port + "/" + db_name;
        connect(client_url, db_name);
    }

    public databaseAccess(String db_name){
        this("127.0.0.1", 27017, db_name);
    }

    private void connect(String client_uri, String db_name){
        MongoClientURI uri = new MongoClientURI(client_uri);

        // Connecting to the mongodb server using the given client uri.
        MongoClient mongo_client = new MongoClient(uri);

        // Fetching the database from the mongodb.
        setMongoDB(mongo_client.getDatabase(db_name));

    }

    public MongoDatabase getMongoDB(){
        return mongoDB;
    }

    public void setMongoDB(MongoDatabase mongoDB){
        this.mongoDB = mongoDB;
    }

    public ArrayList<Document> getData(String collection, String from, String to){
        FindIterable<Document> iterable = null;
        //if(from == null || to == null) {
            iterable = getMongoDB().getCollection(collection).find().sort(new BasicDBObject("TimeInstant", 1));
        /* Date are not correct format
            } else {
            iterable = getMongoDB().getCollection(collection)
                                    .find(and(gte("TimeInstant",from), lte("TimeInstant",to)))
                                    .sort(new BasicDBObject("TimeInstant", 1));
        }*/
        // Iterate the results and apply a block to each resulting document.
        // Iteramos los resultados y aplicacimos un bloque para cada documento.
        ArrayList<Document> ld = new ArrayList<>();
        iterable.forEach((Block<Document>) document -> ld.add(document));
        return  ld;
    }

    public void setData(String collection, ArrayList<Document> data){
        try {
            // /**/ System.out.println("test: building collection");
            getMongoDB().createCollection(collection);
            // /**/ System.out.println("test: building collection Ok");
        }catch (MongoCommandException e){
            // /**/ System.out.println("test building collection: already done");
        }
        // /**/ System.out.println("test: writing data");
        getMongoDB().getCollection(collection).insertMany(data);
        // /**/ System.out.println("test: writing data done");
    }

    public Document getModel(String collection, String modelID){
        BasicDBObject whereQuery = new BasicDBObject();
        whereQuery.put("modelID", modelID);
        FindIterable<Document> iterable = null;
        iterable = getMongoDB().getCollection(collection).find(whereQuery);
        if(iterable.first() != null){
            return iterable.first();
        }
        return null;
    }

    public void setModel(String collection, Document model){
        // /**/ System.out.println("test Creando BD: Hay colección? " + collection);
        try {
            getMongoDB().createCollection(collection);
           // /**/ System.out.println("test No");
        }catch (MongoCommandException e){
            // /**/ System.out.println("test Sí");
        }

        BasicDBObject whereQuery = new BasicDBObject();
        whereQuery.put("modelID", model.get("modelID"));
        FindIterable<Document> iterable = null;
        iterable = getMongoDB().getCollection(collection).find(whereQuery);
        // /**/ System.out.println("test: Existe el modelo?");
        if(iterable.first() == null){
            // /**/ System.out.println("test: No, se crea");
            getMongoDB().getCollection(collection).insertOne(model);
        } else {
            // /**/ System.out.println("test: Si, se actualiza");
            getMongoDB().getCollection(collection).replaceOne(eq("modelID", model.get("modelID")), model);
        }
    }
}
