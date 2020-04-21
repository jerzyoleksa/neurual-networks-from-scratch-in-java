package jo.nn.binary;
import java.util.Arrays;

public class MatrixPrinter {

    public static void main(String[] args) {
        final double[][] matrix = new double[4][4];
        printMatrix(matrix);
    }

    public static void printMatrix(double[][] matrix) {
        Arrays.stream(matrix)
        .forEach(
            (row) -> {
                System.out.print("[");
                Arrays.stream(row)
                .forEach((el) -> System.out.print(" " + el + " "));
                System.out.println("]");
            }
        );
    }
    
    public static void printMatrix(int[][] matrix) {
        Arrays.stream(matrix)
        .forEach(
            (row) -> {
                System.out.print("[");
                Arrays.stream(row)
                .forEach((el) -> System.out.print(" " + el + " "));
                System.out.println("]");
            }
        );
    }

}