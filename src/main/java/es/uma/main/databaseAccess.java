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
        ArrayList<Document> ld = new ArrayList<>();
        iterable.forEach((Block<Document>) document -> ld.add(document));
        return  ld;
    }

    public void setData(String collection, ArrayList<Document> data){
        try {
            getMongoDB().createCollection(collection);
        }catch (MongoCommandException e){
        }
        getMongoDB().getCollection(collection).insertMany(data);
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
        try {
            getMongoDB().createCollection(collection);
        }catch (MongoCommandException e){
        }

        BasicDBObject whereQuery = new BasicDBObject();
        whereQuery.put("modelID", model.get("modelID"));
        FindIterable<Document> iterable = null;
        iterable = getMongoDB().getCollection(collection).find(whereQuery);
        if(iterable.first() == null){
            getMongoDB().getCollection(collection).insertOne(model);
        } else {
            getMongoDB().getCollection(collection).replaceOne(eq("modelID", model.get("modelID")), model);
        }
    }
}
