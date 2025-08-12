#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
  CLI: ./symnmf <goal> <input_file>
  goals:
    - sym  : compute similarity matrix W
    - ddg  : compute diagonal degree matrix D of W
    - norm : compute normalized similarity matrix D^{-1/2} A D^{-1/2}

  Input file: N lines, each line is a data point of dimension d, comma-separated
  Output: requested matrix, comma-separated values, 4 decimal places, each row on its own line
*/

#define LINE_BUFFER_SIZE 16384

static void print_error_and_exit(void) {
    const char *msg = "An Error Has Occurred";
    fprintf(stdout, "%s\n", msg);
    exit(1);
}

static void print_invalid_input_and_exit(void) {
    const char *msg = "Invalid Input!";
    fprintf(stdout, "%s\n", msg);
    exit(1);
}

static double **allocate_matrix(int num_rows, int num_cols) {
    double **matrix;
    int i;

    matrix = (double **)malloc(num_rows * sizeof(double *));
    if (matrix == NULL) {
        print_error_and_exit();
    }
    for (i = 0; i < num_rows; i++) {
        matrix[i] = (double *)calloc((size_t)num_cols, sizeof(double));
        if (matrix[i] == NULL) {
            print_error_and_exit();
        }
    }
    return matrix;
}

static void free_matrix(double **matrix, int num_rows) {
    int i;
    if (matrix == NULL) return;
    for (i = 0; i < num_rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

static void rewind_file(FILE *fp) {
    if (fseek(fp, 0L, SEEK_SET) != 0) {
        print_error_and_exit();
    }
}

static void count_rows_cols(FILE *fp, int *out_rows, int *out_cols) {
    char line[LINE_BUFFER_SIZE];
    int rows;
    int cols;
    int first_line_cols;
    char *ptr;

    rows = 0;
    first_line_cols = -1;

    while (fgets(line, LINE_BUFFER_SIZE, fp) != NULL) {
        /* count commas in this line */
        cols = 0;
        ptr = line;
        while (*ptr != '\0' && *ptr != '\n') {
            if (*ptr == ',') {
                cols++;
            }
            ptr++;
        }
        cols++; /* number of values = commas + 1 */

        if (first_line_cols == -1) {
            first_line_cols = cols;
        } else if (cols != first_line_cols) {
            print_invalid_input_and_exit();
        }
        rows++;
    }

    if (rows <= 0 || first_line_cols <= 0) {
        print_invalid_input_and_exit();
    }

    *out_rows = rows;
    *out_cols = first_line_cols;
}

static double **read_points_from_file(const char *file_path, int *out_n, int *out_d) {
    FILE *fp;
    int n;
    int d;
    double **points;
    char line[LINE_BUFFER_SIZE];
    int i, j;
    char *token;

    fp = fopen(file_path, "r");
    if (fp == NULL) {
        print_invalid_input_and_exit();
    }

    count_rows_cols(fp, &n, &d);
    rewind_file(fp);

    points = allocate_matrix(n, d);

    i = 0;
    while (i < n && fgets(line, LINE_BUFFER_SIZE, fp) != NULL) {
        token = strtok(line, ",\n");
        j = 0;
        while (token != NULL) {
            if (j >= d) {
                print_invalid_input_and_exit();
            }
            points[i][j] = strtod(token, NULL);
            token = strtok(NULL, ",\n");
            j++;
        }
        if (j != d) {
            print_invalid_input_and_exit();
        }
        i++;
    }

    if (i != n) {
        print_invalid_input_and_exit();
    }

    fclose(fp);

    *out_n = n;
    *out_d = d;
    return points;
}

static void print_matrix(double **matrix, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < m - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

static double squared_euclidean(const double *a, const double *b, int d) {
    int t;
    double sum;
    double diff;
    sum = 0.0;
    for (t = 0; t < d; t++) {
        diff = a[t] - b[t];
        sum += diff * diff;
    }
    return sum;
}

/* Helper to compute degree vector: degrees[i] = sum_j W[i][j] */
static void compute_degrees(double **W, int n, double *degrees) {
    int i, j;
    for (i = 0; i < n; i++) {
        double sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += W[i][j];
        }
        degrees[i] = sum;
    }
}

/* ================= Core required functions ================= */

/* Compute similarity matrix W: w_ij = exp(-||xi-xj||^2 / 2), w_ii = 0 */
double **sym(double **points, int n, int d) {
    double **W;
    int i, j;
    double dist2;

    W = allocate_matrix(n, n);

    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            dist2 = squared_euclidean(points[i], points[j], d);
            W[i][j] = exp(-dist2 / 2.0);
            W[j][i] = W[i][j];
        }
        W[i][i] = 0.0;
    }

    return W;
}

/* Compute diagonal degree matrix D from W: D_ii = sum_j W_ij */
double **ddg(double **W, int n) {
    double **D;
    double *degrees;
    int i;

    D = allocate_matrix(n, n);
    degrees = (double *)malloc((size_t)n * sizeof(double));
    if (degrees == NULL) {
        print_error_and_exit();
    }

    compute_degrees(W, n, degrees);

    for (i = 0; i < n; i++) {
        D[i][i] = degrees[i];
    }

    free(degrees);
    return D;
}

/* Compute normalized similarity matrix: D^{-1/2} A D^{-1/2} */
double **norm(double **W, int n) {
    double **N;
    double *degrees;
    double *inv_sqrt_d;
    int i, j;

    degrees = (double *)malloc((size_t)n * sizeof(double));
    if (degrees == NULL) {
        print_error_and_exit();
    }
    compute_degrees(W, n, degrees);

    inv_sqrt_d = (double *)malloc((size_t)n * sizeof(double));
    if (inv_sqrt_d == NULL) {
        free(degrees);
        print_error_and_exit();
    }

    for (i = 0; i < n; i++) {
        if (degrees[i] <= 0.0) {
            inv_sqrt_d[i] = 0.0; /* isolated node */
        } else {
            inv_sqrt_d[i] = 1.0 / sqrt(degrees[i]);
        }
    }

    N = allocate_matrix(n, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            N[i][j] = inv_sqrt_d[i] * W[i][j] * inv_sqrt_d[j];
        }
    }

    free(inv_sqrt_d);
    free(degrees);
    return N;
}

/* ========================= main ========================= */

int main(int argc, char **argv) {
    const char *goal;
    const char *file_path;
    double **points;
    double **W;
    double **M;
    int n, d;

    if (argc != 3) {
        print_invalid_input_and_exit();
    }

    goal = argv[1];
    file_path = argv[2];

    points = read_points_from_file(file_path, &n, &d);

    if (strcmp(goal, "sym") == 0) {
        M = sym(points, n, d);
        print_matrix(M, n, n);
        free_matrix(M, n);
    } else if (strcmp(goal, "ddg") == 0) {
        W = sym(points, n, d);
        M = ddg(W, n);
        print_matrix(M, n, n);
        free_matrix(M, n);
        free_matrix(W, n);
    } else if (strcmp(goal, "norm") == 0) {
        W = sym(points, n, d);
        M = norm(W, n);
        print_matrix(M, n, n);
        free_matrix(M, n);
        free_matrix(W, n);
    } else {
        free_matrix(points, n);
        print_invalid_input_and_exit();
    }

    free_matrix(points, n);
    return 0;
} 