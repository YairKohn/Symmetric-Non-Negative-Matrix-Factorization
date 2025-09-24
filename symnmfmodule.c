#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "symnmf.h"

/**
 * @brief Helpers for converting Python list-of-lists to C double **
 * 
 * Checks that `list2d` is a list of lists of equal length.
 * 
 * @param list2d The Python list-of-lists
 * @param out_rows Pointer to store the number of rows in the matrix
 * @param out_cols Pointer to store the number of columns in the matrix
 * @return 1 if the input is valid and dimensions were retrieved, 0 otherwise
 */
static int get_pylist_shape(PyObject *list2d, int *out_rows, int *out_cols) {
    if (!PyList_Check(list2d)) { PyErr_SetString(PyExc_TypeError, "Invalid Input!"); return 0; }
    Py_ssize_t n = PyList_Size(list2d);
    if (n <= 0) { PyErr_SetString(PyExc_ValueError, "Invalid Input!"); return 0; }
    PyObject *row0 = PyList_GetItem(list2d, 0);
    if (!PyList_Check(row0)) { PyErr_SetString(PyExc_TypeError, "Invalid Input!"); return 0; }
    Py_ssize_t d = PyList_Size(row0);
    if (d <= 0) { PyErr_SetString(PyExc_ValueError, "Invalid Input!"); return 0; }
    /* verify rectangular */
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *row = PyList_GetItem(list2d, i);
        if (!PyList_Check(row) || PyList_Size(row) != d) {
            PyErr_SetString(PyExc_TypeError, "Invalid Input!");
            return 0;
        }
    }
    *out_rows = (int)n;
    *out_cols = (int)d;
    return 1;
}

/**
 * @brief Allocate a C matrix with of doubles with the given dimensions.
 * 
 * @param rows The number of rows to allocate in the matrix
 * @param cols The number of columns to allocate in the matrix
 * @return A pointer to the allocated matrix, or NULL on error
 */
static double **allocate_cmatrix_rows(int rows, int cols) {
    double **mat = (double **)malloc((size_t)rows * sizeof(double *));
    if (!mat) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < rows; i++) {
        mat[i] = (double *)malloc((size_t)cols * sizeof(double));
        if (!mat[i]) {
            for (int t = 0; t < i; t++) free(mat[t]);
            free(mat);
            PyErr_NoMemory();
            return NULL;
        }
    }
    return mat;
}

/**
 * @brief Fill a pre-allocated C matrix with values from a Python list of lists
 * 
 * @param list2d The Python list-of-lists
 * @param mat Pre-allocated C matrix of shape (rows x cols) to fill.
 * @param rows The number of rows in the matrix
 * @param cols The number of columns in the matrix
 * @return 1 if successful, 0 if error
 */
static int fill_cmatrix_from_pylist(PyObject *list2d, double **mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        PyObject *row = PyList_GetItem(list2d, (Py_ssize_t)i);
        for (int j = 0; j < cols; j++) {
            PyObject *val = PyList_GetItem(row, (Py_ssize_t)j);
            double v = PyFloat_AsDouble(val);
            if (PyErr_Occurred()) {
                return 0;
            }
            mat[i][j] = v;
        }
    }
    return 1;
}

/**
 * @brief Convert a rectangular Python list of lists into a newly allocated C matrix (double **).
 * 
 * @param list2d The Python list-of-lists
 * @param out_rows Pointer to store the number of rows in the matrix
 * @param out_cols Pointer to store the number of columns in the matrix
 * @return A pointer to the allocated matrix, or NULL on error
 */
static double **pylist_to_cmatrix(PyObject *list2d, int *out_rows, int *out_cols) {
    int rows, cols;
    if (!get_pylist_shape(list2d, &rows, &cols)) return NULL;
    double **mat = allocate_cmatrix_rows(rows, cols);
    if (!mat) return NULL;
    if (!fill_cmatrix_from_pylist(list2d, mat, rows, cols)) {
        for (int t = 0; t < rows; t++) free(mat[t]);
        free(mat);
        return NULL;
    }
    *out_rows = rows;
    *out_cols = cols;
    return mat;
}

/**
 * @brief Free a C matrix
 * 
 * @param m The matrix to free
 * @param rows The number of rows in the matrix
 */
static void free_cmatrix(double **m, int rows) {
    if (!m) return;
    for (int i = 0; i < rows; i++) free(m[i]);
    free(m);
}

/**
 * @brief Convert C matrix to Python list of lists
 * 
 * @param m The C matrix
 * @param rows The number of rows in the matrix
 * @param cols The number of columns in the matrix
 * @return A pointer to the Python list of lists, or NULL on error
 */
static PyObject *cmatrix_to_pylist(double **m, int rows, int cols) {
    PyObject *outer = PyList_New(rows);
    if (!outer) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < rows; i++) {
        PyObject *row = PyList_New(cols);
        if (!row) { PyErr_NoMemory(); Py_DECREF(outer); return NULL; }
        for (int j = 0; j < cols; j++) {
            PyObject *val = PyFloat_FromDouble(m[i][j]);
            if (!val) { PyErr_NoMemory(); Py_DECREF(row); Py_DECREF(outer); return NULL; }
            PyList_SetItem(row, j, val); /* steals ref */
        }
        PyList_SetItem(outer, i, row); /* steals ref */
    }
    return outer;
}

/**
 * @brief Python wrapper for sym() function
 * 
 * Computes similarity matrix A from data points using Gaussian kernel.
 * 
 * Args:
 *   points: List of lists representing data points (n x d matrix)
 * 
 * Returns:
 *   Python list of lists representing similarity matrix A (n x n)
 * 
 * Raises:
 *   TypeError, ValueError, MemoryError
 */
static PyObject *py_sym(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    int n, d;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return NULL;
    double **points = pylist_to_cmatrix(points_obj, &n, &d);
    if (!points) return NULL;
    double **A = sym(points, n, d);
    if (!A) { free_cmatrix(points, n); PyErr_NoMemory(); return NULL; }
    PyObject *res = cmatrix_to_pylist(A, n, n);
    free_cmatrix(points, n);
    free_cmatrix(A, n);
    return res;
}

/**
 * @brief Python wrapper for ddg() function
 * 
 * Computes diagonal degree matrix D from data points by first computing
 * the similarity matrix A, then extracting diagonal degrees.
 * 
 * Args:
 *   points: List of lists representing data points (n x d matrix)
 * 
 * Returns:
 *   Python list of lists representing diagonal degree matrix D (n x n)
 * 
 * Raises:
 *   TypeError, ValueError, MemoryError
 */
static PyObject *py_ddg(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    int n, d;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return NULL;
    double **points = pylist_to_cmatrix(points_obj, &n, &d);
    if (!points) return NULL;
    double **A = sym(points, n, d);
    if (!A) { free_cmatrix(points, n); PyErr_NoMemory(); return NULL; }
    double **D = ddg(A, n);
    if (!D) { free_cmatrix(points, n); free_cmatrix(A, n); PyErr_NoMemory(); return NULL; }
    PyObject *res = cmatrix_to_pylist(D, n, n);
    free_cmatrix(points, n);
    free_cmatrix(A, n);
    free_cmatrix(D, n);
    return res;
}

/**
 * @brief Python wrapper for norm() function
 * 
 * Computes normalized similarity matrix D^(-1/2) A D^(-1/2) from data points
 * by first computing the similarity matrix A, then applying symmetric normalization.
 * 
 * Args:
 *   points: List of lists representing data points (n x d matrix)
 * 
 * Returns:
 *   List of lists representing normalized similarity matrix (n x n)
 * 
 * Raises:
 *   TypeError, ValueError, MemoryError
 */
static PyObject *py_norm(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return NULL;
    int n, d;
    double **points = pylist_to_cmatrix(points_obj, &n, &d);
    if (!points) return NULL;
    double **A = sym(points, n, d);
    if (!A) { free_cmatrix(points, n); PyErr_NoMemory(); return NULL; }
    double **W = norm(A, n);
    if (!W) { free_cmatrix(points, n); free_cmatrix(A, n); PyErr_NoMemory(); return NULL; }
    PyObject *res = cmatrix_to_pylist(W, n, n);
    free_cmatrix(points, n);
    free_cmatrix(A, n);
    free_cmatrix(W, n);
    return res;
}

/**
 * @brief Python wrapper for symnmf(W, H_init, max_iter, epsilon) function
 * 
 * Performs SymNMF factorization W ≈ HH^T using multiplicative updates.
 * 
 * Args:
 *   W: List of lists representing similarity matrix (n x n)
 *   H_init: List of lists representing initial factor matrix (n x k)
 *   max_iter: Maximum number of iterations (int)
 *   epsilon: Convergence threshold (float)
 * 
 * Returns:
 *   List of lists representing optimized factor matrix H (n x k)
 * 
 * Raises:
 *   TypeError, ValueError, MemoryError
 */
static PyObject *py_symnmf(PyObject *self, PyObject *args) {
    PyObject *W_obj, *H_obj;
    int n1, n2, k;
    int max_iter;
    double epsilon;
    if (!PyArg_ParseTuple(args, "OOid", &W_obj, &H_obj, &max_iter, &epsilon)) return NULL;
    double **W = pylist_to_cmatrix(W_obj, &n1, &n2);
    double **H = pylist_to_cmatrix(H_obj, &n2, &k);
    if (!W || !H || n1 != n2) {
        if (W) free_cmatrix(W, n1);
        if (H) free_cmatrix(H, n2);
        PyErr_SetString(PyExc_ValueError, "Invalid shapes for W or H");
        return NULL;
    }
    double **H_res = symnmf(W, H, n1, k, max_iter, epsilon);
    if (!H_res) { free_cmatrix(W, n1); free_cmatrix(H, n2); PyErr_NoMemory(); return NULL; }
    PyObject *res = cmatrix_to_pylist(H_res, n1, k);

    free_cmatrix(W, n1);
    free_cmatrix(H, n2);
    free_cmatrix(H_res, n1);
    return res;
}

/**
 * Method table for the SymNMF Python module
*/
static PyMethodDef SymnmfMethods[] = {
    {"sym", (PyCFunction)py_sym, METH_VARARGS, "Compute similarity matrix A from points"},
    {"ddg", (PyCFunction)py_ddg, METH_VARARGS, "Compute diagonal degree matrix from points"},
    {"norm", (PyCFunction)py_norm, METH_VARARGS, "Compute normalized similarity matrix from points"},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS, "Run SymNMF given A and initial H"},
    {NULL, NULL, 0, NULL}
};

/**
 * Python module definition
*/
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    SymnmfMethods
};

/**
 * Python module initialization
*/
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
