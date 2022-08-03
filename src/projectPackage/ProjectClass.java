package projectPackage;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math.*;
import org.encog.Encog;
import org.encog.mathutil.randomize.RangeRandomizer;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.benchmark.RandomTrainingFactory;
import org.encog.util.simple.EncogUtility;



public class ProjectClass {
    public static final int INPUT_COUNT = 2;
    public static final int HIDDEN_COUNT = 10;
    public static final int OUTPUT_COUNT = 1;

    public static File outputFile = new File("output.txt");
    public static BasicNetwork generateNetwork()
    {
        final BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(INPUT_COUNT));
        network.addLayer(new BasicLayer(HIDDEN_COUNT));
        network.addLayer(new BasicLayer(OUTPUT_COUNT));
        network.getStructure().finalizeStructure();
        network.reset();
        return network;
    }

    public static MLDataSet generateSet(int seed) // We use this method to generate at random 1000 samples of the function on the square [-pi,pi] x [-pi,pi] (Point 1. and 2.)
    {
        final MLDataSet dataSet = RandomTrainingFactory.generate(seed,1000,
                INPUT_COUNT, OUTPUT_COUNT, -Math.PI, Math.PI);
        for(MLDataPair inputPair : dataSet) {
            inputPair.setIdealArray(new double[] { Math.sin(inputPair.getInputArray()[0])*Math.cos(inputPair.getInputArray()[1])});
        }
        return dataSet;

    }
    public static MLDataSet generateTestSamples() { //This method generates test samples of the function (point 4.)

        final MLDataSet testSample = new BasicMLDataSet();

        double space = 2 * Math.PI / 99; //Here we compute space between points. 2 * Math.PI because the range is [-pi, pi] and divided by 99, because between 100 points there are 99 connections.
        for(double i = 0; i<100; i++) {
            for(double j = 0; j<100; j++) {
                double x = -Math.PI + (i * space);
                double y = -Math.PI + (j * space);
                double[] point = {x, y};
                //System.out.print(x+" "+y+" "+Math.sin(point[0])*Math.cos(point[1]));
                MLData input = new BasicMLData(point);
                MLData output = new BasicMLData(new double[] {Math.sin(point[0])*Math.cos(point[1])});
                MLDataPair dataPair = new BasicMLDataPair(input, output);
                testSample.add(dataPair);
            }
        }
        return testSample;
    }
    //Method below trains a multilayer perceptron with the 1000 training samples of the function(3.) and uses class BufferedWritter and FileWriter to save output to text file
    public static void trainPerceptron(BasicNetwork network,MLDataSet trainingData, MLDataSet validationData ) throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        System.out.println("Training is starting:");
        writer.append("Training is starting:\n");
        ResilientPropagation train = new ResilientPropagation(network,trainingData);
        train.setThreadCount(1);
        int epoch =1 ;
        do{
            train.iteration();
            String text = "Epoch #" + epoch + " Training error: "+ train.getError()+" Validation error: "+network.calculateError(validationData);
            System.out.println(text);
            writer.append(text+'\n');
            epoch++;
        } while (train.getError()> 0.01) ;
        writer.close();
        train.finishTraining();

    }
    //Method below simulates the multilayer perceptron on the test samples, computes the mean squared error for these samples, prints it out on the console and uses class BufferedWritter and FileWriter to save output to text file
    public static void simulatePerceptron(BasicNetwork network,MLDataSet data) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile, true));
        double sum = 0;
        System .out .println("Neural Network Results: ") ;
        for (MLDataPair pair : data ) {
            MLData output = network.compute(pair.getInput()) ;
            sum = sum + Math.pow(pair.getIdeal().getData(0) - output.getData(0), 2);
            //System.out.println("result: "+output.toString()+" ideal:"+pair.getIdeal());
        }
        double meanError = sum/data.size();
        String text = "Mean quadratic error of test samples: " + meanError;
        System.out.println(text);
        writer.append(text);
        writer.close();
    }
    public static void main(String args[]) throws IOException
    {
        BasicNetwork network = generateNetwork();
        MLDataSet training = generateSet(0);
        MLDataSet validation = generateSet(1);
        MLDataSet testSample = generateTestSamples();
        trainPerceptron(network, training, validation);

        simulatePerceptron(network,testSample);
    }
}