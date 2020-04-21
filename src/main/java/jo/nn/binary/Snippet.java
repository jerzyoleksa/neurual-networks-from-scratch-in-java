package jo.nn.binary;

import jo.nn.math.np;

public class Snippet {
	public static void main(String[] args) {
			
			double[][] input = {{-0.1127}};
		
			double[][] sigmoid = np.sigmoid(input);
			
			MatrixPrinter.printMatrix(sigmoid);
			
			double out2 = 0.450166*(-0.5)+ 0.52497919*(0.7)+0.47502081*(-0.9)+0.57444252*(0.3);
			System.out.println(out2);
	}
}

