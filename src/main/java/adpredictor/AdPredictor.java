package adpredictor;

import java.util.Random;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.jet.random.Normal;

public class AdPredictor {

    private double beta;
    private double priorVariance;
    private double epsilon;
    private double lambda = 0;
    private int regularizationStep = 100;

    private int regularizationCounter = 0;

    private final int cardinality;
    private Normal normal;
    private Vector mean;
    private Vector variance;

    public AdPredictor(int cardinality, double priorVariance, double beta, double epsilon) {

        this.beta = beta;
        this.priorVariance = priorVariance;
        this.epsilon = epsilon;

        this.cardinality = cardinality;
        this.normal = new Normal(0.0, 1.0, new Random());

        this.mean = new RandomAccessSparseVector(this.cardinality);
        this.variance = new DenseVector(this.cardinality);
        this.variance.assign(priorVariance);
    }

    public void lambda(double lambda) {
        this.lambda = lambda;
    }

    public void regularizationStep(int value) {
        this.regularizationStep = value;
    }

    public void train(double y, Vector x) {

        double totalDeviation = this.totalDeviation(x);
        double predictionLocation = this.predictionLocation(y, x, totalDeviation);
        double vFunctionValue = this.vFunction(predictionLocation);
        double wFunctionValue = this.wFunction(predictionLocation, vFunctionValue);

        for (Element elem : x.nonZeroes()) {
            double variance = this.variance.getQuick(elem.index());
            double meanUpdate =  y * elem.get() * (variance / totalDeviation) * vFunctionValue;
            double varianceUpdate = variance * (1 - elem.get() * (variance / totalDeviation) * wFunctionValue );
            this.mean.incrementQuick(elem.index(), meanUpdate);
            this.variance.setQuick(elem.index(), varianceUpdate);
        }

        this.regularizationCounter++;

        if (this.regularizationCounter >= this.regularizationStep) {
            this.regularize();
            this.regularizationCounter = 0;
        }
    }

    public double predict(Vector x) {
        return this.normal.cdf(this.predictionLocation(1.0, x, this.totalDeviation(x)));
    }
    
    public Vector getWeightMeans() {
        return this.mean;
    }
    
    public Vector getWeightVariances() {
        return this.variance;
    }

    public void regularize() {
        double eps = this.epsilon;
        for (Element mean : this.mean.nonZeroes()) {
            
            // Pull back towards the prior
            Element variance = this.variance.getElement(mean.index());
            double var = variance.get();
            mean.set(var * ((1 - eps) * mean.get() / var));
            variance.set((this.priorVariance * var) / ((1 - eps) * this.priorVariance + eps * var));
            
            // Prune
            if (this.klDivergence(mean.get(), var) * this.cardinality < this.lambda) {
                mean.set(0.0);
                variance.set(this.priorVariance);
            }
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

    private double klDivergence(double weightMean, double weightVariance) {
        double q = 0.5;
        double p = this.normal.cdf(weightMean / (Math.pow(this.beta, 2) + (this.cardinality - 1)  * this.priorVariance + weightVariance));
        return p * Math.log(p / q) + (1.0 - p) * Math.log((1.0 - p) / (1.0 - q));
    }

}
