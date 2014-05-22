package adpredictor;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.jet.random.Normal;

public class AdPredictor implements Writable {

    private double beta;
    private double priorVariance;
    private double epsilon;
    private double lambda = 0;
    private int regularizationStep = 100;

    private int regularizationCounter = 0;

    private  int cardinality;
    private Normal normal = new Normal(0.0, 1.0, new Random());
    private Vector mean;
    private Vector variance;

    public AdPredictor(int cardinality, double priorVariance, double beta, double epsilon) {

        this.beta = beta;
        this.priorVariance = priorVariance;
        this.epsilon = epsilon;

        this.cardinality = cardinality;

        this.mean = new RandomAccessSparseVector(this.cardinality);
        this.variance = new DenseVector(this.cardinality);
        this.variance.assign(priorVariance);
    }

    public AdPredictor() {
        // For deserialization
    }

    public AdPredictor lambda(double lambda) {
        this.lambda = lambda;
        return this;
    }

    public AdPredictor regularizationStep(int value) {
        this.regularizationStep = value;
        return this;
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

    public double classifyScalar(Vector x) {
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
        List<Integer> indicesToRemove = new ArrayList<Integer>();

        for (Element mean : this.mean.nonZeroes()) {

            // Pull back towards the prior
            Element variance = this.variance.getElement(mean.index());
            double var = variance.get();
            mean.set(var * ((1 - eps) * mean.get() / var));
            variance.set((this.priorVariance * var) / ((1 - eps) * this.priorVariance + eps * var));

            // Identify weights to prune
            if (this.klDivergence(mean.get(), var) * this.cardinality < this.lambda) {
                indicesToRemove.add(mean.index());
            }
        }

        // Prune
        for (Integer idx : indicesToRemove) {
            this.mean.setQuick(idx, 0.0);
            this.variance.setQuick(idx, this.priorVariance);
        }
    }
    
    public double[] drawWeights () {
        double[] weights = new double[this.cardinality];
        
        for (int i=0; i < this.cardinality; i++) {
            weights[i] = (new Normal(this.mean.getQuick(i), this.variance.getQuick(i), new Random())).nextDouble();
        }
        
        return weights;
        
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
        double v = this.beta;
        for (Element elem : vector.nonZeroes()) {
            v += Math.abs(elem.get()) * this.variance.getQuick(elem.index());
        }
        return Math.sqrt(v);
    }

    private double klDivergence(double weightMean, double weightVariance) {
        double q = 0.5;
        double p = this.normal.cdf(weightMean / (Math.pow(this.beta, 2) + (this.cardinality - 1)  * this.priorVariance + weightVariance));
        return p * Math.log(p / q) + (1.0 - p) * Math.log((1.0 - p) / (1.0 - q));
    }

    @Override
    public void readFields(DataInput input) throws IOException {

        this.priorVariance = input.readDouble();
        this.beta = input.readDouble();
        this.epsilon = input.readDouble();
        this.lambda = input.readDouble();

        this.regularizationCounter = input.readInt();
        this.regularizationStep = input.readInt();
        this.cardinality = input.readInt();

        this.mean = VectorWritable.readVector(input);
        this.variance = VectorWritable.readVector(input); 
    }

    @Override
    public void write(DataOutput output) throws IOException {
        for (double v : Arrays.asList(this.priorVariance, this.beta, this.epsilon, this.lambda)) {
            output.writeDouble(v);
        }
        for (int v : Arrays.asList(this.regularizationCounter, this.regularizationStep, this.cardinality)) {
            output.writeInt(v);
        }
        for (Vector v : Arrays.asList(this.mean, this.variance)) {
            VectorWritable.writeVector(output, v);
        }

    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        AdPredictor other = (AdPredictor) obj;
        if (Double.doubleToLongBits(beta) != Double.doubleToLongBits(other.beta)) {
            return false;
        }
        if (cardinality != other.cardinality) {
            return false;
        }
        if (Double.doubleToLongBits(epsilon) != Double.doubleToLongBits(other.epsilon)) {
            return false;
        }
        if (Double.doubleToLongBits(lambda) != Double.doubleToLongBits(other.lambda)) {
            return false;
        }
        if (mean == null) {
            if (other.mean != null) {
                return false;
            }
        } else if (mean.minus(other.mean).zSum() != 0.0) {
            return false;
        }
        if (Double.doubleToLongBits(priorVariance) != Double.doubleToLongBits(other.priorVariance)) {
            return false;
        }
        if (regularizationCounter != other.regularizationCounter) {
            return false;
        }
        if (regularizationStep != other.regularizationStep) {
            return false;
        }
        if (variance == null) {
            if (other.variance != null) {
                return false;
            }
        } else if (variance.minus(other.variance).zSum() != 0.0) {
            return false;
        }
        return true;
    }

}
