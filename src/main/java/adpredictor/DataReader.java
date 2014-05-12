package adpredictor;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

import com.google.gson.Gson;

public class DataReader implements Iterator <Datapoint>{
    
    private final BufferedReader reader;
    private String currentLine;
    
    private Gson gson;
    
    public DataReader(String filename) {
        try {
            this.reader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        
        this.gson = new Gson();
    }

    @Override
    public boolean hasNext() {
        try {
            this.currentLine = this.reader.readLine();
            if (this.currentLine != null) {
                return true;
            } else {
                this.reader.close();
                return false;
            }
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    @Override
    public Datapoint next() {
        return this.gson.fromJson(this.currentLine, Datapoint.class);
    }

    @Override
    public void remove() {
        // No-op.
    }


}