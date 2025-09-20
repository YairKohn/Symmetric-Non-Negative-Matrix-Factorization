#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "symnmf.h"

/* Constants */
#define BETA 0.5
#define MAX_ITER 300
#define EPS 0.0001
#define DELIMITER ','
#define SYM "sym"
#define DDG "ddg"
#define NORM "norm"

const char *ERR_MSG = "An Error Has Occurred\n";

/*
 * Dynamically allocating memory for a new 2D array (matrix)
 * at the address of 'arr'. 
 * Returns 0 on success.
*/
int allocate_2D_array(double ***arr, int rows, int cols)
{
    int i;
    (*arr) = (double **)malloc(rows * sizeof(double *));
    if (*arr == NULL) return 1;

    for (i = 0; i < rows; i++)
    {
        (*arr)[i] = (double *)calloc(cols, sizeof(double));

        if ((*arr)[i] == NULL)
        {
            /* If any row allocation fails: free allocated rows */
            free_2D_array(arr, i);
            return 1;
        }
    }
    return 0;
}

/*
 * Free 2D array 'arr' dynamic memory that contains 'rows' rows.
 * Returns 0 on success.
 *
 * 'arr' - Address of 2D array with 'rows' rows.
 * 'rows' - number of rows to free.
 */
int free_2D_array(double ***arr, int rows)
{
    int i;
    for (i = 0; i < rows; i++)
        free((*arr)[i]);
    free((*arr));
    return 0;
}

/*
 * Prints in stdout 2D array (matrix) 'M'
 * Output example:
 * 0.0011, 0.0012
 * 0.0021, 0.0022
 * 0.0031, 0.0032
 *
 * Returns 0 on success.
 *
 * 'M' - Address of 2D array matrix with dimension 'rows' x 'cols'.
 */
int print_matrix(double ***M, const int rows, const int cols)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%.4f", (*M)[i][j]);

            if (j != cols - 1)
                printf("%c", DELIMITER);
        }
        printf("\n");
    }
    return 0;
}

/*
 * Returns the norm2 squared of 'vector'.
 * norm2_squared(v) = v[0]^2 + v[1]^2 + ... + v[len-1]^2
 *
 * 'vector' - pointer to 1D array with length 'len'.
 */
double norm2_squared(const double *vector, const int len)
{
    int i;
    double norm_squared = 0;
    for (i = 0; i < len; i++)
        norm_squared += vector[i] * vector[i];

    return norm_squared;
}

/*
 * Calculate the element-wise vector subtraction ('V1' - 'V2') and store the result vector in 'result'.
 * precondition: '*result' is dynamically allocated array with the length 'len'.
 * Returns 0 on success.
 *
 * 'result' - Address of 1D array with length 'len'.
 * 'V1', 'V2' - pointer to 1D array with length 'len'.
 */
int vector_sub(double **result, const double *V1, const double *V2, const int len)
{
    int i;
    for (i = 0; i < len; i++)
        (*result)[i] = V1[i] - V2[i];
    return 0;
}

/*
 * Calculate the element-wise matrix subtraction ('M1' - 'M2') and store the result matrix in '*result'.
 * precondition: '*result' is dynamically allocated matrix with dimensions 'rows' x 'cols'.
 * Returns 0 on success.
 *
 * 'result' - Address of 2D array (matrix) with dimensions 'rows' x 'cols'.
 * 'M1', 'M2' - pointer to 2D array (matrix) with dimensions 'rows' x 'cols'.
 */
int matrix_sub(double ***result, const double * const *M1, const double * const *M2, const int rows, const int cols)
{
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            (*result)[i][j] = M1[i][j] - M2[i][j];
    
    return 0;
}

/*
 * Copy the 2D array (matrix) 'copyFrom' to the 2D array (matrix) 'copyTo'.
 * precondition: 'copyTo' is dynamically allocated with the dimensions 'rows' x 'cols'.
 * Returns 0 on success.
 *
 * 'copyTo' - Address of 2D array with dimensions 'rows' x 'cols'.
 * 'copyFrom' - pointer to 2D array with dimensions 'rows' x 'cols'.
 */
int copy_matrix(double ***copyTo, const double * const *copyFrom, const int rows, const int cols)
{
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            (*copyTo)[i][j] = copyFrom[i][j];
    
    return 0;
}

/*
 * Multiply by a diagonal matrix on the left or right and write to result.
 * precondition: 'result' is NOT dynamically allocated.
 * Caller must free(*result) after use.
 * Returns 0 on success.
 *
 * 'M' - pointer to 2D array (matrix) with dimensions 'N' x 'N'.
 * 'D' - pointer to length-N vector that represents the diagonal of a diagonal matrix with dimensions 'N' x 'N'.
 * diag_on_right_flag - 1 for M*diag(D), 0 for diag(D)*M.
 */
int mul_diag_matrix(double ***result, const double * const *M, const double *D, const int N, const int diag_on_right_flag)
{
    int i, j;

    if (!result || !M || !D) return 1;
    if (diag_on_right_flag != 0 && diag_on_right_flag != 1) return 1;
    if (allocate_2D_array(result, N, N) != 0) return 1;

    if (diag_on_right_flag == 1)
    {
        /* M × D: (MD)_ij = m_ij * d_j */
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                (*result)[i][j] = M[i][j] * D[j];
    } else {
        /* D × M: (DM)_ij = d_i * m_ij */
        for (i = 0; i < N; i++)
        {
            double di = D[i];  
            for (j = 0; j < N; j++)
                (*result)[i][j] = M[i][j] * di;
        }
    }
    return 0;
}

/*
 * Calculate the squared matrix 'D' raised by power 'power' and place it in 'result'.
 * precondition: 'result' is NOT dynamically allocated.
 * Returns 0 on success.
 *
 * 'D' - pointer to 1D array that represents the diagonal of a diagonal matrix with dimensions 'N' x 'N'.
 */
int pow_diag_matrix(double **result, const double *D, const int N, const double power)
{
    int i;

    if (!result || !D || N <= 0) return 1;
    double *out = malloc((size_t)N * sizeof(double));
    if (!out) return 1;

    for (i = 0; i < N; ++i) 
    {
        /* Protection against non-positive elements */
        if (D[i] <= 0.0) 
        {  
            free(out);
            return 1;
        }
        /* Equivalent to pow(D[i], -0.5) */
        out[i] = 1.0 / sqrt(D[i]);  
    }

    *result = out;
    return 0;
}

/*
 * Compute matrix multiplication of A * B and place it in 'result'.
 * precondition: '*result' is NOT dynamically allocated.
 * Caller must free(*result) after use.
 * Returns 0 on success.
 *
 * C == A * B iff for every i,j in [0,N): c_ij == Σ(k=0 to N-1) a_ik*b_kj
 *
 * 'A' - pointer to 'rows_A' x 'cols_A' matrix.
 * 'B' - pointer to 'rows_B' x 'cols_B' matrix.
 */
int mul_matrix(double ***result, const double * const *A, const int rows_A, const int cols_A,
               const double * const *B, const int rows_B, const int cols_B)
{
    int i, j, k;
    double sum;

    if (!result || !A || !B) return 1;
    if (rows_A <= 0 || cols_A <= 0 || rows_B <= 0 || cols_B <= 0) return 1;
    if (cols_A != rows_B) return 1;

    if (allocate_2D_array(result, rows_A, cols_B) != 0) return 1;

    for (i = 0; i < rows_A; i++)
        for (j = 0; j < cols_B; j++)
        {
            sum = 0.0;
            for (k = 0; k < cols_A; k++)
                sum += A[i][k] * B[k][j];
            (*result)[i][j] = sum;
        }

    return 0;
}

/*
 * Calculate the transposed matrix of 'M' and place it in 'result'.
 * precondition: '*result' is NOT dynamically allocated.
 * Caller must free(*result) after use.
 * Returns 0 on success.
 *
 * T == M_T iff for every i,j in [0,N): t_ij == m_ji
 *
 * 'M' - pointer to 2D array (matrix) with dimensions 'rows' x 'cols'.
 */
int transpose(double ***result, const double * const *M, const int rows, const int cols)
{
    int i, j;

    if (!result || !M || rows <= 0 || cols <= 0) return 1;
    if (allocate_2D_array(result, cols, rows) != 0)
        return 1;

    for (i = 0; i < cols; i++)
        for (j = 0; j < rows; j++)
            (*result)[i][j] = M[j][i];
        
    return 0;
}

/*
 * Returns the squared frobenius norm of the matrix 'M'.
 *
 * F_norm_squared(A) == ΣΣ |a_ij|^2
 *
 * 'M' - pointer to of 2D array with dimensions 'rows' x 'cols'.
 */
double F_norm_squared(const double * const *M, const int rows, const int cols)
{
    int i, j;
    double val, norm_squared = 0.0;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
        {
            val = M[i][j];
            norm_squared += val * val;
        }
    
    return norm_squared;
}

/*
 * Convert a diagonal matrix 'D' from 1D array representation
 * to 2D array representation and place it in '*M'.
 * precondition: '*M' is NOT dynamically allocated.
 * Caller must free(*M) after use.
 * Returns 0 on success.
 *
 * 'D' - pointer to 1D array that represents the diagonal of a diagonal matrix with dimensions 'N' x 'N'.
 */
int parse_diag_to_matrix_form(double ***M, const double *D, const int N)
{
    int i;
    
    if (allocate_2D_array(M, N, N) != 0) return 1;

    for (i = 0; i < N; i++)
        (*M)[i][i] = D[i];
    
    return 0;
}

/*
 * Calculate the similarity matrix 'A' based on the instructions.
 * precondition: '*A' is NOT dynamically allocated.
 * Caller must free(*A) after use.
 * Returns 0 on success.
 *
 * 'X' - Address of 2D array (matrix) that contains 'N' (rows_X) vectors, each having 'd' dimensions (cols_X).
 */
int C_sym(double ***A, double ***X, const int rows_X, const int cols_X)
{
    double a_ij, *temp_sub;
    int i, j;

    if (allocate_2D_array(A, N, N) != 0) return 1;

    temp_sub = (double *)malloc(d * sizeof(double));
    if (temp_sub == NULL) return 1;

    for (i = 0; i < N; i++)
        for (j = i + 1; j < N; j++)
        {
            vector_sub(&temp_sub, (*X)[i], (*X)[j], d);
            a_ij = exp(-0.5 * norm2_squared(temp_sub, d));

            (*A)[i][j] = a_ij;
            (*A)[j][i] = a_ij;
        }
    
    free(temp_sub);

    return 0;
}

/*
 * Calculate the diagonal degree matrix 'D' based on the instructions.
 * precondition: '*D' is NOT dynamically allocated.
 * Caller must free(*D) after use.
 * Returns 0 on success.
 *
 * 'D' - Address of 1D array that represents a diagonal matrix.
 * 'A' - Address of a similarity matrix with dimensions 'N' x 'N'.
 */
int C_ddg(double **D, double ***A, const int N)
{
    int i, j;
    
    (*D) = (double *)calloc(N, sizeof(double));
    if (*D == NULL) return 1;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            (*D)[i] += (*A)[i][j];
    
    return 0;
}

/*
 * Calculate the normalized similarity matrix 'W' based on the instructions.
  * precondition: '*W' is NOT dynamically allocated.
 * Caller must free(*W) after use.
 * Returns 0 on success.
 *
 * W = D^-0.5 * A * D^-0.5
 * 'D' - pointer to a diagonal degree matrix with dimensions 'N' x 'N'.
 * 'A' - pointer to a similarity matrix with dimensions 'N' x 'N'.
 */

int C_norm(double ***W, const double *D, const double * const *A, const int N)
{
    double *P = NULL, **temp_calc = NULL;

    if (!W || !D || !A || N <= 0) return 1;

    /* Let P = D^(-1/2) */
    if (pow_diag_matrix(&P, D, N) != 0) return 1;
    
    /* temp_calc = A * diag(P) */
    if (mul_diag_matrix(&temp_calc, A, P, N, 1) != 0) {
        free(P);
        return 1;
    }
    
    /* W = diag(P) * temp_calc */
    if (mul_diag_matrix(W, &temp_calc, P, N, 0) != 0) {
        free(P);
        free_2D_array(&temp_calc, N);
        return 1;
    }

    free(P);
    free_2D_array(&temp_calc, N);
    return 0;
}

/*
 * Calculate the updated 'H_out' matrix from 'H_in' based on the instructions.
 * precondition: '*H_out' is dynamically allocated.
 * Returns 0 on success.
 *
 * 'H_out' - Address of a 2D array (matrix) with dimensions 'rows_H' x 'cols_H'.
 * 'H_in' - pointer to a 2D array with dimensions 'rows_H' x 'cols_H'.
 * 'W' - pointer to a 2D array with dimensions 'N_W' x 'N_W'.
 */
int update_H(double ***H_out, const double * const *H_in, const int rows_H, const int cols_H,
             const double * const *W, const int N_W)
{
    double **H_T = NULL, **NUM = NULL, **DEN_tmp = NULL, **DEN = NULL;
    int i, j, result = 0;
    
    if (!H_out || !H_in || !W || rows_H <= 0 || cols_H <= 0 || N_W <= 0 || rows_H != N_W) return 1;

    if (transpose(&H_T, H_in, rows_H, cols_H) != 0) return 1;
    if (mul_matrix(&NUM, W, N_W, N_W, H_in, rows_H, cols_H) != 0) goto FAIL;
    if (mul_matrix(&DEN_tmp, H_in, rows_H, cols_H, H_T, cols_H, rows_H) != 0) goto FAIL;
    if (mul_matrix(&DEN, DEN_tmp, rows_H, rows_H, H_in, rows_H, cols_H) != 0) goto FAIL;

    for (i = 0; i < rows_H; i++)
        for (j = 0; j < cols_H; j++) {
            /* Avoid division by zero */
            double denom = fabs(DEN[i][j]) < 1e-12 ? copysign(1e-12, DEN[i][j]) : DEN[i][j];
            (*H_out)[i][j] = H_in[i][j] * (1.0 - BETA + BETA * (NUM[i][j] / denom));
        }
    goto CLEANUP;

FAIL:
    result = 1;

CLEANUP:
    if (H_T) free_2D_array(&H_T, cols_H);
    if (NUM) free_2D_array(&NUM, rows_H);
    if (DEN_tmp) free_2D_array(&DEN_tmp, rows_H);
    if (DEN) free_2D_array(&DEN, rows_H);
    return result;
}

/*
 * Calculate the optimal 'H_out' matrix from the initial 'H_in' based on the instructions.
 * precondition: '*H_out' is NOT dynamically allocated.
 * Caller must free(*H_out) after use.
 * Returns 0 on success.
 *
 * 'H_out', 'H_in' - Address of a 2D array (matrix) with dimensions 'rows_H' x 'cols_H'.
 * 'W' - Address of a 2D array with dimensions 'N_W' x 'N_W'.
 */
int C_symnmf(double ***H_out, double ***H_in, const int rows_H, const int cols_H,
             double ***W, const int N_W)
{
    int i;
    double F_norm_s_val;
    double **SUB;

    if (allocate_2D_array(&SUB, rows_H, cols_H) != 0) return 1;
    if (allocate_2D_array(H_out, rows_H, cols_H) != 0)
    {
        free_2D_array(&SUB, rows_H);
        return 1;
    }

    F_norm_s_val = EPS + 1; /* Initial value */
    i = 0;
    while (i < MAX_ITER && F_norm_s_val >= EPS)
    {
        if (update_H(H_out, H_in, rows_H, cols_H, W, N_W) != 0)
        {
            free_2D_array(&SUB, rows_H);
            free_2D_array(H_out, rows_H)
            return 1;
        }
        matrix_sub(&SUB, H_out, *H_in, rows_H, cols_H);
        F_norm_s_val = F_norm_squared(SUB, rows_H, cols_H);
        copy_matrix(H_in, *H_out, rows_H, cols_H);
        i++;
    }
    free_2D_array(&SUB, rows_H);
    return 0;
}

/*
 * Finds the rows and columns of the data inside 'file_name' and place it on '*rows' and '*cols' respectively.
 * Returns 0 on success.
 *
 * 'file_name' - Address of 1D char array that represents the name of a file.
 * 'rows', 'cols' - Address for an integer variable.
 */
int get_file_rows_cols(char **file_name, int *rows, int *cols)
{
    FILE *file;
    int c;
    double fake_num;
    file = fopen(*file_name, "r");
    if (file == NULL)
        return 1;
    
    /* Initialize Counters */
    (*rows) = 0;
    (*cols) = 0;
    while (fscanf(file, "%lf", &fake_num) != EOF)
    {
        c = fgetc(file);
        /* If it's the first row, increment the column counter */
        if ((*rows) == 0)
            (*cols)++;
        /* If it's a new line or EOF, increment the row counter */
        if (c == '\n' || c == EOF)
            (*rows)++;
    }
    fclose(file);
    return 0;
}

/*
 * Read the data points from 'file_name' and place them into '*X'.
 * precondition: '*X' is NOT dynamically allocated.
 * Caller must free(*X) after use.
 * returns 0 on success.
 *
 * 'file_name' - Address of 1D char array that represents the name of a file.
 * 'rows', 'cols' - Address for an integer variable.
 */
int read_file(double ***X, char **file_name, int *rows, int *cols)
{
    FILE *file;
    int c, row, col;
    if (get_file_rows_cols(file_name, rows, cols) != 0) return 1;
    if (allocate_2D_array(X, *rows, *cols) != 0) return 1;

    file = fopen(*file_name, "r");
    if (file == NULL) return 1;

    row = 0;
    col = 0;
    /* read float number and then read ',' or '\n'*/
    while (fscanf(file, "%lf", &((*X)[row][col++])) != EOF)
    {
        c = fgetc(file);
        if (c == '\n')
        {
            row++;
            col = 0;
        }
        else if (c != DELIMITER && c != EOF)
        {
            free_2D_array(X, *rows);
            return 1;
        }
    }
    fclose(file);
    
    return 0;
}

/* Main program
 * Print the requested matrix by the 'goal'.
 * Expected argv: [{Program Name}, {'goal'}, {'file_name'}]
 * Returns 0 on success.
 *
 * 'goal' - "string" that equals to one of the following: ["sym", "ddg", "norm"]
 * 'file_name' - "string" of an existing file in the project folder that contains data point by the format:
 * 1.111111,2.2222222
 * 3.333333333,4.4444444
 * 5.0,6.6666
 */
int main(int argc, char *argv[])
{
    char *goal, *file_name;
    double **X, **A, *D, **D_out, **W;
    int N, d;

    if (argc != 3)
    {
        printf("%s", ERR_MSG);
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];

    if (read_file(&X, &file_name, &N, &d) != 0) 
    {
        printf("%s", ERR_MSG);
        return 1;
    }
    /* if goal in {'sym', 'ddg', 'norm'}, the method gets the suitable matrix and prints it. */
    if (strcmp(goal, SYM) == 0)
    {
        C_sym(&A, &X, N, d);

        print_matrix(&A, N, N);

        free_2D_array(&A, N);
    }
    else if (strcmp(goal, DDG) == 0)
    {
        C_sym(&A, &X, N, d);
        C_ddg(&D, &A, N);
         if (parse_diag_to_matrix_form(&D_out, D, N) != 0)
        {
            printf("%s", ERR_MSG);
            free_2D_array(&A, N);
            free(D);
            free_2D_array(&X, N);
            return 1;
        }
        
        print_matrix(&D_out, N, N);

        free_2D_array(&A, N);
        free(D);
        free_2D_array(&D_out, N);
    }
    else if (strcmp(goal, NORM) == 0)
    {
        C_sym(&A, &X, N, d);
        C_ddg(&D, &A, N);
        C_norm(&W, D, A, N);

        print_matrix(&W, N, N);

        free_2D_array(&A, N);
        free(D);
        free_2D_array(&W, N);
    }
    else
    {
        printf("%s", ERR_MSG);
        free_2D_array(&X, N);
        return 1;
    }

    free_2D_array(&X, N);

    return 0;
}
