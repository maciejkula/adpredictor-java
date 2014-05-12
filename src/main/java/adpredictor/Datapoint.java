package adpredictor;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public class Datapoint {
    
    public int classification;
    private int cardinality;
    private Integer[] featureIndices;
    
    public Vector toVector() {
        Vector vector = new RandomAccessSparseVector(this.cardinality);
        
        for (int index : this.featureIndices) {
            vector.setQuick(index, 1.0);
        }
        
        return vector;
    }

}
