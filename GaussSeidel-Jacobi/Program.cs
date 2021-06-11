using System;
using System.Collections.Generic;
using System.Linq;

namespace GaussSeidel_Jacobi
{
    class Program
    {
        private static int roundingDecimals = 4;
        static void Main(string[] args)
        {
            //string values = "5.0000,-1.0000,3.00000,2.00000,-8.00000,1.00000,-2.000000,0.000000,4.000000";
            string values = "7.0232,1.4113,-1.5633,-1.1351,8.9842,3.0873,2.6762,3.5653,-6.8717";

            Random r = new Random();
            List<double> matrixValues = values.Split(',').Select(x=>Convert.ToDouble(x)).ToList();

            //Generate Random matrix in n.dddd form which isn't diagonally dominant
            double[,] matrix = new double[3, 3];
            matrix = CreateMatrix(matrixValues, 3, 3, 4);

            //Check if the Matrix is Diagonally dominant
            bool isMatrixDiagonallyDominant = CheckMatrixDiagonallyDominant(matrix, 4);

            //if(isMatrixDiagonallyDominant)
            //Solve the convergence equation Gauss-Seidel
            // -(I + L)^-1 .U
            var conv_Matrix_GaussSeidel = getConvergenceMatrix(matrix, 0, 4);

            //Solve the convergence equation Gauss-Jacobi
            var conv_Matrix_GaussJacobi = getConvergenceMatrix(matrix, 1, 4);

            //Check if any of the norms are lesser than 1 Gauss-Seidel
            var isNormTestPassed_GS = TestNorms(conv_Matrix_GaussSeidel, 4);
            if (isNormTestPassed_GS)
            {
                Console.WriteLine("Norm test has been passed for Gauss Seidel method for the Convergence Matrix");
            }
            else
                Console.WriteLine("Norm test isn't passed for Gauss Seidel method for the Convergence Matrix");

            //Check if any of the norms are lesser than 1 Gauss-Jacobi
            var isNormTestPassed_GJ = TestNorms(conv_Matrix_GaussJacobi, 4);
            if (isNormTestPassed_GJ) {
                Console.WriteLine("Norm test has been passed for Gauss Jacobi method for the Convergence Matrix");
            }
            else
                Console.WriteLine("Norm test isn't passed for Gauss Jacobi method for the Convergence Matrix");

            List<double> knownCoeffs = new List<double>();

            knownCoeffs.Add(3.3953);
            knownCoeffs.Add(1.6259);
            knownCoeffs.Add(4.7649);
            List<double[]> finalGSList = new List<double[]>();
            List<double[]> finalGJList = new List<double[]>();


            //Gauss Seidel Method
            if (isMatrixDiagonallyDominant || isNormTestPassed_GS)
                finalGSList = ComputeGaussSeidelMethod(matrix,knownCoeffs);
            if (finalGSList.Any())
            {
                Console.WriteLine("\n\n\n\n");
                Console.WriteLine("Gauss Seidel Observations :");
                Console.WriteLine("---x1----        ----x2----           ----x3----");
                foreach(var record in finalGSList)
                {
                    Console.WriteLine(record[0] + "              " + record[1] + "              " + record[2]);
                }
                Console.WriteLine("\n\n\n\n");
            }
            //Gauss Jacobi Method
            if (isMatrixDiagonallyDominant || isNormTestPassed_GS)
                finalGJList = ComputeGaussJacobiMethod(matrix, knownCoeffs);
            if (finalGJList.Any())
            {
                Console.WriteLine("\n\n\n\n");
                Console.WriteLine("Gauss Jacobi Observations :");
                Console.WriteLine("---x1----        ----x2----           ----x3----");
                foreach (var record in finalGJList)
                {
                    Console.WriteLine(record[0] + "              " + record[1] + "              " + record[2]);
                }
                Console.WriteLine("\n\n\n\n");
             }
            //Check for relative error

        }

        private static List<double[]> ComputeGaussJacobiMethod(double[,] matrix, List<double> knownValues)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            int unknowns = rows;
            List<double[]> finalConvergenceArrayList = new List<double[]>();
            double x1 = 0, x2 = 0, x3 = 0;
            double x1_prev = 0, x2_prev = 0, x3_prev = 0;
            double relative_error = 0;
            int iterationCount = 1;
            try
            {
                Console.WriteLine("Number of Unknowns :" + unknowns);
                Console.WriteLine("Starting values of x1 = 2, x2 = 3. x3 = 4");

                //First Iteration to get first Relative Error 
                x1 = Math.Round((knownValues[0] - (matrix[0, 1] * x2_prev) - (matrix[0, 2] * x3_prev)) / matrix[0, 0], roundingDecimals);
                x2 = Math.Round((knownValues[1] - (matrix[1, 0] * x1_prev) - (matrix[1, 2] * x3_prev)) / matrix[1, 1], roundingDecimals);
                x3 = Math.Round((knownValues[2] - (matrix[2, 0] * x1_prev) - (matrix[2, 1] * x2_prev)) / matrix[2, 2], roundingDecimals);
                finalConvergenceArrayList.Add(new double[] { x1, x2, x3 });
                relative_error = (x1 - x1_prev) / x1;

                // x1 = (a - n2x2 - n3x3)/n1
                while (Math.Abs(relative_error) > 0.01)
                {
                    x1_prev = x1; x2_prev = x2; x3_prev = x3;
                    x1 = Math.Round((knownValues[0] - (matrix[0, 1] * x2_prev) - (matrix[0, 2] * x3_prev)) / matrix[0, 0], roundingDecimals);
                    x2 = Math.Round((knownValues[1] - (matrix[1, 0] * x1_prev) - (matrix[1, 2] * x3_prev)) / matrix[1, 1], roundingDecimals);
                    x3 = Math.Round((knownValues[2] - (matrix[2, 0] * x1_prev) - (matrix[2, 1] * x2_prev)) / matrix[2, 2], roundingDecimals);
                    finalConvergenceArrayList.Add(new double[] { x1, x2, x3 });
                    relative_error = (x3 - x3_prev) / x3;
                    iterationCount++;
                    if (iterationCount > 50)
                        break;
                }
            }
            catch (Exception)
            {

                throw;
            }
            return finalConvergenceArrayList;
        }

        private static List<double[]> ComputeGaussSeidelMethod(double[,] matrix,List<double> knownValues)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            int unknowns = rows;
            List<double[]> finalConvergenceArrayList = new List<double[]>();
            double x1=0, x2 = 0, x3 = 0;
            double x1_prev = 0, x2_prev = 0, x3_prev = 0;
            double relative_error = 0;
            int iterationCount = 1;
            try
            {
                Console.WriteLine("Number of Unknowns :" + unknowns);
                Console.WriteLine("Starting values of x1 = 2, x2 = 3. x3 = 4");

                //First Iteration to get first Relative Error 
                x1 = Math.Round((knownValues[0] - (matrix[0, 1] * x2) - (matrix[0, 2] * x3)) / matrix[0, 0], roundingDecimals);
                x2 = Math.Round((knownValues[1] - (matrix[1, 0] * x1) - (matrix[1, 2] * x3)) / matrix[1, 1], roundingDecimals);
                x3 = Math.Round((knownValues[2] - (matrix[2, 0] * x1) - (matrix[2, 1] * x2)) / matrix[2, 2], roundingDecimals);
                finalConvergenceArrayList.Add(new double[] { x1, x2, x3 });
                relative_error = (x1 - x1_prev) / x1;

                // x1 = (a - n2x2 - n3x3)/n1
                while (Math.Abs(relative_error) > 0.01)
                {
                    x1_prev = x1; x2_prev = x2; x3_prev = x3;
                    x1 = Math.Round((knownValues[0] - (matrix[0, 1] * x2) - (matrix[0, 2] * x3)) / matrix[0, 0], roundingDecimals);
                    x2 = Math.Round((knownValues[1] - (matrix[1, 0] * x1) - (matrix[1, 2] * x3)) / matrix[1, 1], roundingDecimals);
                    x3 = Math.Round((knownValues[2] - (matrix[2, 0] * x1) - (matrix[2, 1] * x2)) / matrix[2, 2], roundingDecimals);
                    finalConvergenceArrayList.Add(new double[] { x1, x2, x3 });
                    relative_error = (x3 - x3_prev) / x3;
                    iterationCount++;
                    if (iterationCount > 50)
                        break;
                }
            }
            catch (Exception)
            {

                throw;
            }
            return finalConvergenceArrayList;
        }

        private static bool TestNorms(double[,] matrix, int roundingDecimals)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            try
            {
                double norm1, norm2, normInfinity = 0;
                norm1 = CalculateNorm1(matrix);
                Console.WriteLine("Norm1 of the Convergence Matrix C is: " + norm1);
                if (norm1 <= 1)
                    return true;
                else
                    norm2 = CalculateNorm2(matrix);
                Console.WriteLine("Norm2 of the Convergence Matrix C is: " + norm2);
                if (norm2 <= 1)
                    return true;
                else
                    normInfinity = CalculateNormInfinity(matrix, roundingDecimals);
                Console.WriteLine("Norm Infinity of the Convergence Matrix C is: " + normInfinity);
                if (normInfinity <= 1)
                    return true;
                else
                    return false;
            }
            catch (Exception ex)
            {

                return false;
            }
            return false;
        }

        private static double CalculateNormInfinity(double[,] matrix,int roundingDecimals)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double infinity_norm_sum = 0;
            try
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        infinity_norm_sum = infinity_norm_sum + Math.Pow(matrix[i,j],2);
                    }
                }
            }
            catch (Exception ex)
            {

            }
            return Math.Round(Math.Sqrt(infinity_norm_sum),roundingDecimals);
        }

        private static double CalculateNorm2(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double max_row_sum = 0;
            double row_sum = 0;
            try
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        row_sum = row_sum + Math.Abs(matrix[i, j]);
                    }
                    if (Math.Abs(row_sum) > Math.Abs(max_row_sum))
                        max_row_sum = Math.Abs(row_sum);
                    row_sum = 0;
                }
            }
            catch (Exception ex)
            {

            }
            return max_row_sum;
        }

        private static double CalculateNorm1(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double max_column_sum = 0;
            double column_sum = 0;
            try
            {
                for (int j = 0; j < columns; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        column_sum = column_sum + Math.Abs(matrix[i, j]);
                    }
                    if (Math.Abs(column_sum) > Math.Abs(max_column_sum))
                        max_column_sum = Math.Abs(column_sum);
                    column_sum = 0;
                }
            }
            catch (Exception ex)
            {

            }
            return max_column_sum;
        }

        private static double[,] getConvergenceMatrix(double[,] matrix, int isGaussSeidel, int roundingDecimals)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double[,] convMatrix = new double[rows, columns];
            try
            {
                switch (isGaussSeidel)
                {
                    case 0:
                        convMatrix = GetGaussSeidelConvMatrix(matrix ,roundingDecimals);
                        break;
                    case 1:
                        convMatrix = GetGaussJacobiConvMatrix(matrix, roundingDecimals);
                        break;

                    default:
                        break;
                }

            }
            catch (Exception ex)
            {

                throw;
            }
            return convMatrix;
            
        }

        private static double[,] NormalizeMatrix(double[,] matrix,int roundingDecimals)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double[,] normalized_matrix = new double[rows, columns];
            try
            {
                double multiplier = 0;
                Console.WriteLine("--------Addition of Matrices (I + L) -----------");
                for (int i = 0; i < rows; i++)
                {
                    multiplier = matrix[i, i];
                    for (int j = 0; j < columns; j++)
                    {
                        normalized_matrix[i, j] = Math.Round(matrix[i, j] / multiplier, roundingDecimals);
                        Console.Write(normalized_matrix[i, j] + " ");
                    }
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return normalized_matrix;
        }
        private static double[,] GetGaussSeidelConvMatrix(double[,] matrix,int roundingDecimals)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double[,] gaussSeidelConvMatrix = new double[rows, columns];
            try
            {
                var normalized_matrix = NormalizeMatrix(matrix, roundingDecimals);
                //-(I + L)^-1 . U
                var identity_Matrix = GetIdentityMatrix(matrix.GetLength(0), matrix.GetLength(1));
                var lower_Triangular_Matrix = GetLowerTriangularMatrix(normalized_matrix);
                var upper_Triangular_Matrix = GetUpperTriangulaMatrix(normalized_matrix);

                List<double[,]> matricesToBeAdded = new List<double[,]>();
                matricesToBeAdded.Add(identity_Matrix);
                matricesToBeAdded.Add(lower_Triangular_Matrix);
                
                // (I + L)
                var identity_lt_Triangular = AddMatrices(identity_Matrix,lower_Triangular_Matrix,rows, columns);

                // (I + L)^ -1
                var transposed_identity_lt_triangular = CreateTransposeMatrix(identity_lt_Triangular, rows, columns);

                // -1(I + L)^ -1
                var scalar_Multiplied_Matrix = ScalarMultiply(identity_lt_Triangular, -1, rows, columns);
                //-(I + L)^-1 . U

                gaussSeidelConvMatrix = MultiplyMatrix(scalar_Multiplied_Matrix,upper_Triangular_Matrix,rows,columns,roundingDecimals);

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return gaussSeidelConvMatrix;
        }

        private static double[,] GetGaussJacobiConvMatrix(double[,] matrix, int roundingDecimals)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double[,] gaussJacobiConvMatrix = new double[rows, columns];
            try
            {
                var normalized_matrix = NormalizeMatrix(matrix, roundingDecimals);
                //-(L + U)
                var lower_Triangular_Matrix = GetLowerTriangularMatrix(normalized_matrix);
                var upper_Triangular_Matrix = GetUpperTriangulaMatrix(normalized_matrix);

                // (L + U)
                var identity_lt_Triangular = AddMatrices(lower_Triangular_Matrix, upper_Triangular_Matrix, rows, columns);

                // -1(L + U)
                var scalar_Multiplied_Matrix = ScalarMultiply(identity_lt_Triangular, -1, rows, columns);

               gaussJacobiConvMatrix = scalar_Multiplied_Matrix;

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return gaussJacobiConvMatrix;
        }

        

        private static double[,] MultiplyMatrix(double[,] matrix1, double[,] matrix2, int rows, int columns,int roundingDecimals)
        {
            double[,] multiplied_matrix = new double[rows, columns];
            try
            {
                Console.WriteLine("Multiplying Matrices ");
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        multiplied_matrix[i, j] = 0;
                        for (int k = 0; k < columns; k++)
                            multiplied_matrix[i, j] = Math.Round(multiplied_matrix[i, j] + matrix1[i, k] * matrix2[k, j],roundingDecimals);
                        Console.Write(multiplied_matrix[i, j] + " ");
                    }
                    Console.WriteLine();
                }
            }
            catch (Exception)
            {

                throw;
            }
            Console.WriteLine("Multiplying Matrices Completed");
            return multiplied_matrix;
        }

        private static double[,] CreateTransposeMatrix(double[,] matrix, int rows, int columns)
        {
            double[,] transposed_matrix = new double[rows, columns];
            try
            {
                Console.WriteLine("--------Transpose of (I + L) Matrix -----------");
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        transposed_matrix[j, i] = matrix[i,j];
                    }
                }
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        Console.Write(transposed_matrix[i, j] + " ");
                    }
                    Console.WriteLine();
                }
            }
            catch (Exception)
            {

                throw;
            }
            Console.WriteLine("--------Transpose of (I + L) Matrix Created -----------");
            return transposed_matrix;
        }

        private static double[,] AddMatrices(double[,] matrix1, double[,] matrix2, int rows, int columns)
        {
            double[,] summed_matrix = new double[rows, columns];
            try
            {
                Console.WriteLine("--------Addition of Matrices (I + L) -----------");
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        summed_matrix[i, j] = matrix1[i, j] + matrix2[i, j];
                        Console.Write(summed_matrix[i, j] + " ");
                    }
                    Console.WriteLine();
                }
            }
            catch (Exception)
            {

                throw;
            }
            Console.WriteLine("--------Addition of Matrices (I + L) Completed -----------");
            return summed_matrix;
        }
        private static double[,] ScalarMultiply(double[,] matrix, int scalar, int rows, int columns, bool isForAllRows = true, int rowNumber = 0 )
        {
            try
            {

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        matrix[i, j] = matrix[i, j] * scalar;
                    }
                }
            }
            catch (Exception)
            {

                throw;
            }
            return matrix;
        }

        private static double[,] GetLowerTriangularMatrix(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double[,] lt_matrix = new double[rows, columns];
            
            try
            {
                Console.WriteLine("--------Lower Traingular L Matrix -----------");
                for (int i = 0; i < rows; i++)
                {
                    int cumulative_diagonal = i + i;
                    for (int j = 0; j < columns; j++)
                    {
                        if ((i == j) || ((i + j) > cumulative_diagonal))
                            lt_matrix[i, j] = 0;
                        else
                            lt_matrix[i, j] = matrix[i,j];
                        Console.Write(lt_matrix[i, j] + " ");
                    }
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {

                throw;
            }
            Console.WriteLine("--------Lower Traingular L Matrix Created-----------");
            return lt_matrix;
        }

        private static double[,] GetUpperTriangulaMatrix(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            double[,] up_matrix = new double[matrix.GetLength(0), matrix.GetLength(1)];
            try
            {
                Console.WriteLine("--------Upper Traingular L Matrix -----------");
                for (int i = 0; i < rows; i++)
                {
                    int cumulative_diagonal = i + i;
                    for (int j = 0; j < columns; j++)
                    {
                        if ((i == j) || ((i + j) < cumulative_diagonal))
                            up_matrix[i, j] = 0;
                        else
                            up_matrix[i, j] = matrix[i, j];
                        Console.Write(up_matrix[i, j] + " ");
                    }
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {

                throw;
            }
            Console.WriteLine("--------Lower Traingular L Matrix Created-----------");
            return up_matrix;
        }

        private static double[,] GetIdentityMatrix(int rows, int columns)
        {
            double[,] matrix = new double[rows, columns];
            try
            {
                Console.WriteLine("--------Identity I Matrix -----------");
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        if (i == j)
                            matrix[i, j] = 1;
                        else
                            matrix[i, j] = 0;
                        Console.Write(matrix[i, j] +" ");
                    }
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {

                throw;
            }
            Console.WriteLine("--------Identity I Matrix Created-----------");
            return matrix;
        }

        private static bool CheckMatrixDiagonallyDominant(double[,] matrix, int roundingDecimals)
        {
            //double[,] matrix = new double[3, 3];
            int rows = matrix.GetLength(0);
            double nonDiagonalSum = 0;
            int columns = matrix.GetLength(1);
            bool isDiagonallyDominant = false;
            try
            {
                for (int i = 0; i < columns; i++)
                {
                    for (int j = 0; j < rows; j++)
                    {
                        if (i == j)
                            continue;
                        else
                            nonDiagonalSum = Math.Round(nonDiagonalSum, roundingDecimals) + matrix[i, j];
                    }
                    if (Math.Abs(matrix[i, i]) < Math.Abs(nonDiagonalSum))
                        return false;
                    else
                        isDiagonallyDominant = true;
                    nonDiagonalSum = 0;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                
            }
            return isDiagonallyDominant;
        }

        static double[,] CreateMatrix(List<double> valuesToBeUpdated, int columns, int rows, int roundingDecimals, bool isMatrixToBePivoted = false, double[,] matrix = null)
        {
            double[,] final_matrix = new double[rows, columns];
            try
            {

                if (matrix == null)
                {
                    double[,] new_matrix = new double[rows, columns];
                    int count = 0;
                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < columns; j++)
                        {
                            new_matrix[i, j] = Math.Round(valuesToBeUpdated[count], roundingDecimals);
                            count++;
                        }
                    }
                    final_matrix = new_matrix;
                }
                else
                {
                    double[,] new_matrix = new double[rows, columns];
                    int count = 0;
                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < columns; j++)
                        {
                            if (j > matrix.GetLength(0) - 1)
                                new_matrix[i, j] = Math.Round(valuesToBeUpdated[i], roundingDecimals);
                            else
                                new_matrix[i, j] = Math.Round(matrix[i, j], roundingDecimals);
                            count++;
                        }
                    }
                    final_matrix = new_matrix;
                }
                //if (isMatrixToBePivoted)
                //{
                //    final_matrix = ReArrangeOriginalMatrix(final_matrix);
                //}
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return final_matrix;
        }
    }
}
