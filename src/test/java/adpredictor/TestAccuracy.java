package adpredictor;

import org.apache.mahout.math.Vector;

import junit.framework.TestCase;

public class TestAccuracy extends TestCase {



    public void testAccuracy() {
        DataReader reader = new DataReader("newsgroups.json");

        Adpredictor model = new Adpredictor(130107, 1.0, 1.0, 0.01);

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
    }

}
