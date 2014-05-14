package adpredictor;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.mahout.math.Vector;

import junit.framework.TestCase;

public class TestAccuracy extends TestCase {



    public void testAccuracy() {
        DataReader reader = new DataReader("newsgroups.json");

        AdPredictor model = new AdPredictor(130107, 0.2, 0.05, 0.05);
        model.lambda(0.0000001);

        double accuratelyClassified = 0.0;
        int allClassified = 0;

        while (reader.hasNext()) {
            allClassified++;
            Datapoint datapoint = reader.next();
            Vector x = datapoint.toVector();
            double predictedClass = (model.predict(x) < 0.5) ? -1 : 1;
            if (predictedClass == datapoint.classification) {
                accuratelyClassified += 1.0;
            }
            model.train(datapoint.classification, x);
        }
        System.out.println("Accuracy: " + accuratelyClassified / allClassified);
        System.out.println(model.getWeightMeans().getNumNonZeroElements());
        // System.out.println(model.getWeightVariances());
        
        // Let's serialize and deserialize
        ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        DataOutputStream outputStream = new DataOutputStream(byteStream);
        try {
            model.write(outputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        AdPredictor deserializedModel = new AdPredictor();
        DataInputStream input = new DataInputStream(new ByteArrayInputStream(byteStream.toByteArray()));
        try {
            deserializedModel.readFields(input);
            System.out.println("deserialized");
        } catch (IOException e) {
            e.printStackTrace();
        }

        
        assertTrue(model.equals(deserializedModel));
        System.out.println("Finished");
         
    }

}
