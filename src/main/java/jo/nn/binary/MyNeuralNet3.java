package jo.nn.binary;

import java.util.ArrayList;

import jo.nn.math.np;

public class MyNeuralNet3 {

	private ArrayList<double[][]> predict(double[][] inputs, double[][] weightsLayer1, double[][] weightsLayer2) {
		return feedForward(inputs, weightsLayer1, weightsLayer2);
		//return new ArrayList<double[][]>(){{add(neuralOutputs1);add(neuralOutputs2);}};;
	}

	private ArrayList<double[][]> train(double[][] inputs, double[][] outputs, double[][] weightsLayer1, double[][] weightsLayer2, int n) {
		ArrayList<double[][]> updatedWeights = null;
		for (int i = 0; i < n; i++) {
			ArrayList<double[][]> neuralOutputs = feedForward(inputs, weightsLayer1, weightsLayer2);
			
			//System.out.println("neuralOutputs1:");
			//MatrixPrinter.printMatrix(neuralOutputs.get(0));
			//System.out.println("neuralOutputs2:");
			//MatrixPrinter.printMatrix(neuralOutputs.get(1));
			
			updatedWeights = backPropagation(inputs, neuralOutputs.get(0), neuralOutputs.get(1), outputs, weightsLayer1, weightsLayer2);
			weightsLayer1 = updatedWeights.get(0);
			weightsLayer2 = updatedWeights.get(1);
			//updatedWeights = np.T(updatedWeights);
		}
		return updatedWeights;
	}

	private ArrayList<double[][]> feedForward(double[][] inputs, double[][] weightsLayer1, double[][] weightsLayer2) {
		
		//layer1Outputs = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
	    //layer2Outputs = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
		        
		double[][] multiplied1 = np.dot(inputs, weightsLayer1 ); //dot is a scalar multiplification different than multiply
		double[][] neuralOutputs1 = np.sigmoid(multiplied1);
		
		double[][] multiplied2 = np.dot(neuralOutputs1, weightsLayer2 ); //dot is a scalar multiplification different than multiply
		double[][] neuralOutputs2 = np.sigmoid(multiplied2);
		
		
		return new ArrayList<double[][]>(){{add(neuralOutputs1);add(neuralOutputs2);}};
	}
	
	private ArrayList<double[][]> backPropagation(double[][] inputs, double[][] neuralOutputs1, double[][] neuralOutputs2, double[][] outputs, double[][] weightsLayer1, double[][] weightsLayer2) {
/*		double[][] errors = np.subtract(outputs, neuralOutputs);
		double[][] derivatives = SigmoidDerivative.derivative(np.T( neuralOutputs ));
		double[][] deltas = np.multiply(errors, np.T(derivatives));
		double[][] dots = np.dot( np.T( inputs), deltas);
		double[][] updatedWeights = np.add( np.T( weights ), dots);*/
		
		 			//Calculate the error for layer 2 (The difference between the desired output
		            //and the predicted output).
		   			//layer2_error = training_set_outputs - output_from_layer_2
		            //layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
		            double[][] errorsLayer2 = np.subtract(outputs, neuralOutputs2);
		    		double[][] derivativesLayer2 = SigmoidDerivative.derivative(np.T( neuralOutputs2 ));
		    		double[][] deltasLayer2 = np.multiply(errorsLayer2, np.T(derivativesLayer2));
		            // Calculate the error for layer 1 (By looking at the weights in layer 1,
		            // we can determine by how much layer 1 contributed to the error in layer 2).
		            //layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)		            		
		            //layer1_delta = layer1_error * self.__sigmoid_derivative(neuralOutputs1)
		            double[][] errorsLayer1 = np.dot(deltasLayer2, np.T( weightsLayer2));
		            double[][] derivativesLayer1 = SigmoidDerivative.derivative(np.T( neuralOutputs1 ));
		            double[][] deltasLayer1 = np.multiply(errorsLayer1, np.T(derivativesLayer1));
		            
		            //System.out.println("deltasLayer1::");
		           // MatrixPrinter.printMatrix(deltasLayer1);
		           // System.out.println("-----------");
		            
		            // Calculate how much to adjust the weights by
		            //layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
		            //layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
		            double[][] weightAdjustmentLayer1 = np.dot( np.T( inputs), deltasLayer1);
		            double[][] weightAdjustmentLayer2 = np.dot( np.T( neuralOutputs1), deltasLayer2);
			            
		            //MatrixPrinter.printMatrix(weightAdjustmentLayer2); //OK!!!
		            // Adjust the weights.
		            //self.layer1.synaptic_weights += layer1_adjustment
		            //self.layer2.synaptic_weights += layer2_adjustment
		            double[][] updatedWeightsLayer1 = np.add( weightsLayer1 , weightAdjustmentLayer1);
		            double[][] updatedWeightsLayer2 = np.add( weightsLayer2 , weightAdjustmentLayer2);
		            
		            //System.out.println("---");
		            //MatrixPrinter.printMatrix(updatedWeightsLayer2);
		            //System.out.println("---");

		            return new ArrayList<double[][]>(){{add(updatedWeightsLayer1);add(updatedWeightsLayer2);}};
	}

	public static void main(String[] args) {

		//double[][] inputs = {{0,0,1}, {0,1,0}, {1,0,1}, {1,1,0}};
		//double[][] outputs = {{0,0,1,1}};
		
		double[][] inputs = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}};
	    double[][] outputs = {{0, 1, 0, 1, 0, 1, 0}};
	    	    
		double[][] weightsLayer1 = {
						 {-0.1,  0.4, -0.9, -0.3},
				 		 {-0.7, -0.8, -0.6, -0.5},
						 {-0.2,  0.1, -0.1,  0.3  }};
		
		double[][] weightsLayer2 = {{-0.5}, {0.7}, {-0.9}, {0.3}};
		

		ArrayList<double[][]> updatedWeights = new MyNeuralNet3().train(inputs, np.T( outputs ), weightsLayer1, weightsLayer2, 60000);
		weightsLayer1 = updatedWeights.get(0);
		weightsLayer2 = updatedWeights.get(1);
		//MatrixPrinter.printMatrix(updatedWeights);
		
		double[][] inp = {{1,1,0}};
		ArrayList<double[][]> predicted = new MyNeuralNet3().predict(inp, weightsLayer1, weightsLayer2);
		
		//System.out.println("predicted (should be close to 1):");
		MatrixPrinter.printMatrix(predicted.get(1));

	}
}

