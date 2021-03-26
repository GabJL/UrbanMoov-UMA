package es.uma.auxiliar;

import org.bson.Document;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.time.temporal.TemporalAccessor;
import java.util.*;

public class CSVBuilder {
    private ArrayList<String> titles;
    private LinkedHashMap<Instant, ArrayList<Double>> data;
    private static final Integer TIME = 120;
    private String path;
    private String filename;

    // Data should be correct
    public CSVBuilder(String path, String filename, ArrayList<ArrayList<Document>> data, String [] attr, String per ){
        setData(new LinkedHashMap<>());
        setTitles(new ArrayList<>());
        getTitles().add("TimeInstant");
        this.path = path;
        this.filename = filename;

        int skip = 0;
        if(per == null) per = "15M";
        if(per.equals("30M")) skip = 1;
        else if(per.equals("60M")) skip = 3;
        else if(per.equals("1D")) skip = 95; // 24*4 - 1;

        Instant date = null;
        Integer counter_doc = 0;
        for(ArrayList<Document> ad: data) {
            Integer counter_row = 0;
            Integer index  = 0;
            String name = "Device" + counter_doc;
            for (Document d : ad) {
                String s = d.get("TimeInstant", String.class);
                TemporalAccessor ta = DateTimeFormatter.ISO_INSTANT.parse(s);
                date = Instant.from(ta);
                Integer counter_attr = 0;
                for (String a : attr) {
                    Double val = null;
                    try {
                        val = d.getDouble(a);
                    } catch (ClassCastException e) {
                        // System.out.println("Parameter is not exist or it is not a number");
                        continue;
                    }
                    if (counter_row == 0) {
                        name = name + "-" + a;
                        getTitles().add(name);
                    }
                    if (counter_row % (skip + 1) == 0) {
                        if (counter_doc == 0 && counter_attr == 0) {
                            ArrayList<Double> ai = new ArrayList<>();
                            getData().put(date, ai);
                        }
                        ArrayList<Double> ai = getData().get((getData().keySet().toArray())[index]);
                        ai.add(val);
                    }
                }
                counter_row++;
                if (counter_row % (skip + 1) == 0) index++;
            }
            counter_doc++;
        }
    }

    // Currently this complex is not needed, data are preprocessed (ETL)
    /* public CSVBuilder(String path, String filename, ArrayList<ArrayList<Document>> data ){
        setData(new HashMap<>());
        setTitles(new ArrayList<>());
        getTitles().add("TimeInstant");
        this.path = path;
        this.filename = filename;

        Instant date = null;
        Integer counter = 0;
        for(ArrayList<Document> ad: data){
            getTitles().add("Device"+counter);
            for(Document d: ad){
                String s = d.get("TimeInstant", String.class);
                TemporalAccessor ta = DateTimeFormatter.ISO_INSTANT.parse(s);
                date = Instant.from(ta);
                Integer i = null;
                try {
                    i = d.getInteger("per");
                } catch(ClassCastException e){
                    // System.out.println("Parameter is not integer");
                    continue;
                }
                if(counter == 0){
                    ArrayList<Integer> ai = new ArrayList<>();
                    ai.add(i);
                    getData().put(date,ai);
                } else {
                    for (Map.Entry<Instant, ArrayList<Integer>> entry : getData().entrySet()) {
                        if(entry.getValue().size() == counter) {
                            Instant later = entry.getKey().plusSeconds(TIME);
                            Instant previous = entry.getKey().plusSeconds(-TIME);
                            if(previous.compareTo(date) <= 0 && later.compareTo(date) >= 0){
                                ArrayList<Integer> ai = entry.getValue();
                                ai.add(i);
                                //getData().put(entry.getKey(), ai);
                            }
                        }
                    }

                }
            }
            counter++;;
            for (Map.Entry<Instant, ArrayList<Integer>> entry : getData().entrySet()) {
                if(entry.getValue().size() < counter) {
                    ArrayList<Integer> ai = entry.getValue();
                    Integer i = ai.get(ai.size()-1);
                    ai.add(i);
                    //getData().put(entry.getKey(), ai);
                }
            }
        }
    }*/

    public void writeFile(){
        File dir = new File(path);
        dir.mkdir();
        try {
            /// NEW: Date is not included
            PrintWriter file = new PrintWriter(path+"/"+filename);
            for(int i = 1; i < titles.size()-1; i++)
                file.print(titles.get(i)+" , ");
            file.println(titles.get(titles.size()-1));
            TreeSet<Instant> s = new TreeSet<>(getData().keySet());
            for (Instant ins : s) {
                //file.print(DateTimeFormatter.ISO_INSTANT.format(ins) + " , ");
                for (int i = 0; i < getData().get(ins).size()-1; i++){
                    file.print(getData().get(ins).get(i));
                    file.print(" , ");
                }
                file.println(getData().get(ins).get(getData().get(ins).size()-1));
            }
            file.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }

    public void setTitles(ArrayList<String> titles){
        this.titles = titles;
    }

    public ArrayList<String> getTitles(){
        return titles;
    }

    public void setData(LinkedHashMap<Instant, ArrayList<Double>> data) {
        this.data = data;
    }

    public LinkedHashMap<Instant, ArrayList<Double>> getData() {
        return data;
    }
}
