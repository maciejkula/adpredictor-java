package adpredictor;

import java.util.Random;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.jet.random.Normal;

public class Adpredictor {
    
    private double beta;
    private double priorVariance;
    private double epsilon;
    
    private final int cardinality;
    private Normal normal;
    private Vector mean;
    private Vector variance;
    
    public Adpredictor(int cardinality, double priorVariance, double beta, double epsilon) {
        
        this.beta = beta;
        this.priorVariance = priorVariance;
        this.epsilon = epsilon;
        
        this.cardinality = cardinality;
        this.normal = new Normal(0.0, 1.0, new Random());
        
        this.mean = new DenseVector(this.cardinality);
        this.variance = new DenseVector(this.cardinality);
        this.variance.assign(priorVariance);
    }
    
    public void train(double y, Vector x) {
        
       double totalDeviation = this.totalDeviation(x);
       double predictionLocation = this.predictionLocation(y, x, totalDeviation);
       double vFunctionValue = this.vFunction(predictionLocation);
       double wFunctionValue = this.wFunction(predictionLocation, vFunctionValue);
        
        for (Element elem : x.nonZeroes()) {
            double variance = this.variance.get(elem.index());
            double meanUpdate =  y * elem.get() * (variance / totalDeviation) * vFunctionValue;
            double varianceUpdate = variance * (1 - elem.get() * (variance / totalDeviation) * wFunctionValue );
            this.mean.incrementQuick(elem.index(), meanUpdate);
            this.variance.setQuick(elem.index(), varianceUpdate);
        }
        
        // this.correctDynamics();
        
        //System.out.println(y);
        //System.out.println(this.mean);
    }
    
    public double predict(Vector x) {
        return this.normal.cdf(this.predictionLocation(1.0, x, this.totalDeviation(x)));
    }
    
    private void correctDynamics() {
        for (Element variance : this.variance.nonZeroes()) {
            variance.set((this.priorVariance * variance.get()) / ((1 - this.epsilon) * this.priorVariance + epsilon * variance.get()));
        }
        for (Element mean : this.mean.nonZeroes()) {
            double variance = this.variance.get(mean.index());
            mean.set(variance * ((1 - this.epsilon) * mean.get() / variance));
        }
    }
    
    private double predictionLocation(double y, Vector x, double totalDeviation) {
        return y * x.dot(this.mean) / totalDeviation;
    }
    
    private double vFunction(double t) {
        return this.normal.pdf(t) / this.normal.cdf(t);
    }
    
    private double wFunction(double t, double vFunctionAtT) {
        return vFunctionAtT * (vFunctionAtT + t);
    }
    
    private double totalDeviation(Vector vector) {
        return Math.sqrt(this.beta + this.variance.dot(vector));
    }
 


}
