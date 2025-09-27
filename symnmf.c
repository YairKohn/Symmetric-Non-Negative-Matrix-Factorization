#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

/**
 * @brief Implementation of Symmetric Non-Negative Matrix Factorization (SymNMF)
 * 
 * Command Line Interface:
 *   ./symnmf <goal> <input_file>
 * 
 * Goals:
 *   - sym  : compute similarity matrix A from data points
 *   - ddg  : compute diagonal degree matrix D of A
 *   - norm : compute normalized similarity matrix W : D^{-1/2} A D^{-1/2}
 * 
 * Input file: N lines, each line is a data point of dimension d, comma-separated
 * Output: requested matrix, comma-separated values, 4 decimal places, each row on its own line
*/

#define BETA 0.5
#define EPSILON_DIV 1e-10

/**
 * @brief Prints error message and exits the program
 * 
 * Utility function for consistent error handling throughout the program.
 * Prints "An Error Has Occurred" to stdout and exits with code 1.
 */
static void print_error_and_exit(void) {
    const char *msg;
    msg = "An Error Has Occurred";
    fprintf(stdout, "%s\n", msg);
    exit(1);
}

/**
 * @brief Allocates a 2D matrix with error checking
 * 
 * Dynamically allocates a num_rows x num_cols matrix and initializes all
 * elements to zero. Exits program on memory allocation failure.
 * 
 * @param num_rows Number of rows in the matrix
 * @param num_cols Number of columns in the matrix
 * @return Pointer to allocated matrix, exits on error
 */
static double **allocate_matrix(int num_rows, int num_cols) {
    double **matrix;
    int i;

    matrix = (double **)malloc(num_rows * sizeof(double *));
    if (matrix == NULL) print_error_and_exit();

    for (i = 0; i < num_rows; i++) {
        matrix[i] = (double *)calloc((size_t)num_cols, sizeof(double));
        if (matrix[i] == NULL) print_error_and_exit();
    }
    return matrix;
}

/**
 * @brief Frees memory allocated for a 2D matrix
 * 
 * Safely deallocates memory for a matrix allocated by allocate_matrix().
 * Handles NULL matrix pointer.
 * 
 * @param matrix Pointer to matrix to free
 * @param num_rows Number of rows
 */
static void free_matrix(double **matrix, int num_rows) {
    int i;
    
    if (matrix == NULL) return;
    for (i = 0; i < num_rows; i++)
        free(matrix[i]);
    free(matrix);
}

/**
 * @brief Rewinds the file pointer to the beginning of the file (opened with fopen)
 * 
 * @param fp File pointer to rewind
 */
static void rewind_file(FILE *fp) {
    if (fseek(fp, 0L, SEEK_SET) != 0) {
        print_error_and_exit();
    }
}

/**
 * @brief Load points from a file into a matrix
 * 
 * @param fp File pointer to load points from
 * @param points Matrix to store the points
 * @param n Number of points
 * @param d Length of the points
 */
static void load_points(FILE *fp, double **points, int n, int d) {
    int row, col, ch;
    for (row = 0; row < n; ++row) {
        for (col = 0; col < d; ++col) {
            /* read number; " %lf" skips spaces/previous lines */
            if (fscanf(fp, " %lf", &points[row][col]) != 1) print_error_and_exit();
            
            ch = fgetc(fp);

            if (col < d - 1) {
                /* expected comma between values */
                if (ch != ',') print_error_and_exit();
            } else {
                /* end of line: expected \n or EOF */
                if (ch == '\n') {
                    /* valid, move to next line */
                } else if (ch == EOF) {
                    /* valid only if this is the last row */
                    if (row != n - 1) print_error_and_exit();
                } else {
                    /* unexpected character after last value in row */
                    print_error_and_exit();
                }
            }
        }
    }
}

/**
 * @brief Count the number of rows and columns in a file
 * 
 * @param fp File pointer to count rows and columns from
 * @param out_rows Number of rows
 * @param out_cols Number of columns
 */
static void count_rows_cols(FILE *fp, int *out_rows, int *out_cols) {
    int c;
    double dummy_num;
    
    /* Initialize Counters */
    *out_rows = 0;
    *out_cols = 0;
    
    while (fscanf(fp, "%lf", &dummy_num) != EOF) {
        c = fgetc(fp);
        /* If it's the first row, increment the column counter */
        if (*out_rows == 0) {
            (*out_cols)++;
        }
        /* If it's a new line or EOF, increment the row counter */
        if (c == '\n' || c == EOF) {
            (*out_rows)++;
        }
    }
}

/**
 * @brief Read data points from file and allocate matrix
 * 
 * Reads data points from a CSV file and dynamically allocates a matrix
 * to store them. The matrix dimensions are determined by counting rows
 * and columns in the file.
 * 
 * @param file_path Path to the CSV file containing data points
 * @param out_n Pointer to store number of rows (data points)
 * @param out_d Pointer to store number of columns (dimensions)
 * @return Pointer to allocated matrix (n x d), NULL on error
 */
static double **read_points_from_file(const char *file_path, int *out_n, int *out_d) {
    FILE *fp;
    int n, d;
    double **points;

    fp = fopen(file_path, "r");
    if (fp == NULL) print_error_and_exit();

    count_rows_cols(fp, &n, &d);
    rewind_file(fp);

    points = allocate_matrix(n, d);
    load_points(fp, points, n, d);

    fclose(fp);

    *out_n = n;
    *out_d = d;
    return points;
}

/**
 * @brief Print the matrix to the standard output
 * 
 * @param matrix The matrix to print
 * @param n The number of rows in the matrix
 * @param m The number of columns in the matrix
 */
static void print_matrix(double **matrix, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < m - 1) 
                printf(",");
        }
        printf("\n");
    }
}

/**
 * @brief Compute the squared Euclidean distance between two vectors
 * 
 * @param a The first vector
 * @param b The second vector
 * @param d The dimension of the vectors
 * @return The squared Euclidean distance between the two vectors
 */
static double squared_euclidean(const double *a, const double *b, int d) {
    int t;
    double sum, diff;
    sum = 0.0;
    for (t = 0; t < d; t++) {
        diff = a[t] - b[t];
        sum += diff * diff;
    }
    return sum;
}

/**
 * @brief Multiply A(n x m) by B(m x p) -> returns new matrix C(n x p)
 * 
 * @param A The first matrix
 * @param n The number of rows in the first matrix
 * @param m The number of columns in the first matrix
 * @param B The second matrix
 * @param p The number of columns in the second matrix
 * @return The new matrix C(n x p)
 */
static double **matrix_multiply(double **A, int n, int m, double **B, int p) {
    int i, j, t;
    double sum;
    double **C;
    C = allocate_matrix(n, p);
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            sum = 0.0;
            for (t = 0; t < m; t++)
                sum += A[i][t] * B[t][j];
            C[i][j] = sum;
        }
    }
    return C;
}

/**
 * @brief Compute S = A * A^T for A(n x k) → returns n x n
 * 
 * @param A The matrix
 * @param n The number of rows in the matrix
 * @param k The number of columns in the matrix
 * @return The new matrix S(n x n)
 */
static double **matrix_transpose_multiply(double **A, int n, int k) {
    int i, j, t;
    double sum;
    double **S;
    S = allocate_matrix(n, n);  /* n x n */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (t = 0; t < k; t++) {
                sum += A[i][t] * A[j][t];
            }
            S[i][j] = sum;
        }
    }
    return S;
}


/**
 * @brief Compute Frobenius norm squared of A-B, both n x k
 * 
 * @param A The first matrix
 * @param B The second matrix
 * @param n The number of rows in the matrices
 * @param k The number of columns in the matrices
 * @return The Frobenius norm squared of A-B
 */
static double frobenius_norm_sq_diff(double **A, double **B, int n, int k) {
    int i, j;
    double sum, d;
    sum = 0.0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            d = A[i][j] - B[i][j];
            sum += d * d;
        }
    }
    return sum;
}

/**
 * @brief Helper to compute degree vector: degrees[i] = sum_j A[i][j]
 * 
 * @param A The matrix
 * @param n The number of rows in the matrix
 * @param degrees The degree vector
 */
static void compute_degrees(double **A, int n, double *degrees) {
    int i, j;
    double sum;
    for (i = 0; i < n; i++) {
        sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }
        degrees[i] = sum;
    }
}

/* ================= Core required functions ================= */

/**
 * @brief Computes similarity matrix A from data points using Gaussian kernel
 * 
 * Creates a symmetric similarity matrix where a_ij = exp(-||x_i - x_j||²/2)
 * and diagonal elements are zero. The Gaussian kernel captures local similarity
 * between data points based on Euclidean distance.
 * 
 * @param points Array of data points (n x d matrix)
 * @param n Number of data points
 * @param d Dimension of each data point
 * @return Newly allocated similarity matrix A (n x n), NULL on error
 */
double **sym(double **points, int n, int d) {
    double **A;
    double dist2;
    int i, j;

    A = allocate_matrix(n, n);

    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            dist2 = squared_euclidean(points[i], points[j], d);
            A[i][j] = exp(-dist2 / 2.0);
            A[j][i] = A[i][j];
        }
        A[i][i] = 0.0;
    }

    return A;
}

/**
 * @brief Computes diagonal degree matrix D from similarity matrix A
 * 
 * Creates a diagonal matrix where D_ii = Σ_j A_ij (sum of row i).
 * The degree matrix captures the total similarity/connectivity of each node.
 * 
 * @param A Similarity matrix (n x n)
 * @param n Size of the matrix
 * @return Newly allocated diagonal degree matrix D (n x n), NULL on error
 */
double **ddg(double **A, int n) {
    double **D;
    double *degrees;
    int i;

    D = allocate_matrix(n, n);
    degrees = (double *)malloc((size_t)n * sizeof(double));
    if (degrees == NULL) {
        print_error_and_exit();
    }

    compute_degrees(A, n, degrees);

    for (i = 0; i < n; i++) {
        D[i][i] = degrees[i];
    }

    free(degrees);
    return D;
}

/**
 * @brief Computes normalized similarity matrix W : D^(-1/2) A D^(-1/2)
 * 
 * @param A Similarity matrix (n x n)
 * @param n Size of the matrix
 * @return Newly allocated normalized matrix (n x n), NULL on error
 */
double **norm(double **A, int n) {
    double **W;
    double *degrees, *inv_sqrt_d;
    int i, j;

    degrees = (double *)malloc((size_t)n * sizeof(double));
    if (degrees == NULL) print_error_and_exit();

    compute_degrees(A, n, degrees);

    inv_sqrt_d = (double *)malloc((size_t)n * sizeof(double));
    if (inv_sqrt_d == NULL) {
        free(degrees);
        print_error_and_exit();
    }

    for (i = 0; i < n; i++) {
        if (degrees[i] <= 0.0) /* avoid division by zero */
            degrees[i] = EPSILON_DIV;
        
        inv_sqrt_d[i] = 1.0 / sqrt(degrees[i]);
    }

    W = allocate_matrix(n, n);
    for (i = 0; i < n; i++) 
        for (j = 0; j < n; j++) 
            W[i][j] = inv_sqrt_d[i] * A[i][j] * inv_sqrt_d[j];

    free(inv_sqrt_d);
    free(degrees);
    return W;
}

/* ================= SymNMF iterative optimization ================= */

/**
 * @brief Copy a matrix from src to dst
 * 
 * @param src The source matrix
 * @param dst The destination matrix
 * @param rows The number of rows in the matrices
 * @param cols The number of columns in the matrices
 */
static void copy_matrix_rect(double **src, double **dst, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

/**
 * @brief Perform a multiplicative update step for SymNMF
 * 
 * @param W The similarity matrix
 * @param H_curr The current factor matrix
 * @param H_next The next factor matrix
 * @param n The number of data points
 * @param k The number of clusters
 */
static void multiplicative_update_step(double **W, double **H_curr, double **H_next, int n, int k) {
    int i, j;
    double **num, **HHt, **den;
    double ratio, val, denom;

    num = matrix_multiply(W, n, n, H_curr, k);      
    HHt = matrix_transpose_multiply(H_curr, n, k);  
    den = matrix_multiply(HHt, n, n, H_curr, k);     

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            denom = den[i][j];
            if (denom == 0) /* avoid division by zero */
                denom = EPSILON_DIV;
            ratio =  (num[i][j] / denom) ;
            val = H_curr[i][j] * (1.0 - BETA + BETA * ratio);
            H_next[i][j] = (val > 0.0) ? val : 0.0;
        }
    }
    free_matrix(num, n);
    free_matrix(HHt, n);
    free_matrix(den, n);
}

/**
 * @brief Performs SymNMF factorization by minimizing ||W - HH^T||_F² subject to H ≥ 0
 * 
 * @param W Similarity matrix (n x n)
 * @param H_init Initial factor matrix (n x k)
 * @param n Number of data points
 * @param k Number of clusters
 * @param max_iter Maximum number of iterations
 * @param epsilon Convergence threshold
 * @return Optimized factor matrix H (n x k), NULL on error
 */
double **symnmf(double **W, double **H_init, int n, int k, int max_iter, double epsilon) {
    int iter;
    double **H_curr, **H_next;

    if ( 1>= k || k >= n) print_error_and_exit();

    H_curr = allocate_matrix(n, k);
    H_next = allocate_matrix(n, k);
    copy_matrix_rect(H_init, H_curr, n, k);

    for (iter = 0; iter < max_iter; iter++) {
        multiplicative_update_step(W, H_curr, H_next, n, k);
        if (frobenius_norm_sq_diff(H_next, H_curr, n, k) < epsilon) {
            free_matrix(H_curr, n); /* free original H_curr, copy H_next into H_curr */
            H_curr = H_next;
            H_next = NULL;  /* mark that it's now H_curr */
            break;
        }
        {
            double **tmp = H_curr;
            H_curr = H_next;
            H_next = tmp;
        }
    }
    if (H_next != H_curr) {
        free_matrix(H_next, n);
    }
    return H_curr;
}

/* ========================= main ========================= */

/**
 * @brief Run the sym goal
 * 
 * @param points The points
 * @param n The number of data points
 * @param d The dimension of the points
 */
static void run_goal_sym(double **points, int n, int d) {
    double **M;
    M = sym(points, n, d);
    print_matrix(M, n, n);
    free_matrix(M, n);
}

/**
 * @brief Run the ddg goal
 * 
 * @param points The points
 * @param n The number of data points
 * @param d The dimension of the points
 */
static void run_goal_ddg(double **points, int n, int d) {
    double **W, **M;
    W = sym(points, n, d);
    M = ddg(W, n);
    print_matrix(M, n, n);
    free_matrix(M, n);
    free_matrix(W, n);
}

/**
 * @brief Run the norm goal
 * 
 * @param points The points
 * @param n The number of data points
 * @param d The dimension of the points
 */
static void run_goal_norm(double **points, int n, int d) {
    double **W, **M;

    W = sym(points, n, d);
    M = norm(W, n);
    print_matrix(M, n, n);
    free_matrix(M, n);
    free_matrix(W, n);
}

/**
 * @brief Run the main function
 * 
 * @param argc The number of arguments
 * @param argv The arguments
 * @return The exit code
 */
int main(int argc, char **argv) {
    const char *goal, *file_path;
    int n, d;
    double **points;

    if (argc != 3) {
        print_error_and_exit();
    }
    goal = argv[1];
    file_path = argv[2];

    points = read_points_from_file(file_path, &n, &d);
    if (strcmp(goal, "sym") == 0) {
        run_goal_sym(points, n, d);
    } else if (strcmp(goal, "ddg") == 0) {
        run_goal_ddg(points, n, d);
    } else if (strcmp(goal, "norm") == 0) {
        run_goal_norm(points, n, d);
    } else {
        free_matrix(points, n);
        print_error_and_exit();
    }
    free_matrix(points, n);
    return 0;
}
