package jo.nn.binary;

public class SigmoidDerivative {
	
    public static double sigmoid(double x) {
        return (1.0 / (1 + Math.exp(-x)));
    }
	
	private static double derivativeFromInputs(double x) {
		return sigmoid(x)*(1 - sigmoid(x));
	}
	
	private static double derivativeFromOuputs(double y) {
		return y*(1 - y);
	}
	
	public static double[][] derivative(double[][] a) {	
		int m1 = a.length;
        int n1 = a[0].length;
        
        double[][] c = new double[m1][n1];
        
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n1; j++) {
               
                //c[i][j] = derivative(a[i][j]);
            	c[i][j] = derivativeFromOuputs(a[i][j]);
                
            }
        }
        return c;
	}
	
	public static void main(String[] args) {
		double[][] inputs = {{0.5, 0.0001, 50}};
		double[][] derivative = derivative(inputs);
		MatrixPrinter.printMatrix(derivative);
		
	}

}

