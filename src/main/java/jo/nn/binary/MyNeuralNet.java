package jo.nn.binary;

import jo.nn.math.np;

public class MyNeuralNet {

	private double[][] predict(double[][] inputs, double[][] weights) {
		double[][] neuralOutputs = feedForward(inputs, weights);
		return neuralOutputs;
	}

	private double[][] train(double[][] inputs, double[][] outputs, double[][] weights) {
		
		for (int i = 0; i < 10000; i++) {
			double[][] neuralOutputs = feedForward(inputs, weights);
			
			
			weights = backPropagation(inputs, neuralOutputs, outputs, weights);
			weights = np.T(weights);
		}
		return weights;
	}

	private double[][] feedForward(double[][] inputs, double[][] weights) {
		double[][] multiplied = np.dot(inputs, np.T( weights )); //dot is a scalar multiplification different than multiply
		double[][] neuralOutputs = np.sigmoid(multiplied);
		return neuralOutputs;
	}
	
	private double[][] backPropagation(double[][] inputs, double[][] neuralOutputs, double[][] outputs, double[][] weights) {
		double[][] errors = np.subtract(outputs, neuralOutputs);
		double[][] derivatives = SigmoidDerivative.derivative(np.T( neuralOutputs ));
		double[][] deltas = np.multiply(errors, np.T(derivatives));
		double[][] dots = np.dot( np.T( inputs), deltas);
		double[][] updatedWeights = np.add( np.T( weights ), dots);
		return updatedWeights;
	}

	public static void main(String[] args) {

		double[][] inputs = {{0,0,1}, {0,1,0}, {1,0,1}, {1,1,0}};
		double[][] outputs = {{0,0,1,1}};
		double[][] weights = {{0.5, 0.5, 0.5}};
		

		double[][] updatedWeights = new MyNeuralNet().train(inputs, np.T( outputs ), weights);
	
		//MatrixPrinter.printMatrix(updatedWeights);
		
		double[][] inp = {{1,0,0}};
		double[][] predicted = new MyNeuralNet().predict(inp, updatedWeights);
		
		System.out.println("predicted (should be close to 1):");
		MatrixPrinter.printMatrix(predicted);

	}
}

