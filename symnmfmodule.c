#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "symnmf.h"

/* Utility: convert Python list of lists to C matrix (double **). Expects rectangular matrix. */
static double **pylist_to_cmatrix(PyObject *list2d, int *out_rows, int *out_cols) {
    if (!PyList_Check(list2d)) {
        PyErr_SetString(PyExc_TypeError, "Invalid Input!");
        return NULL;
    }
    Py_ssize_t n = PyList_Size(list2d);
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid Input!");
        return NULL;
    }
    PyObject *row0 = PyList_GetItem(list2d, 0);
    if (!PyList_Check(row0)) {
        PyErr_SetString(PyExc_TypeError, "Invalid Input!");
        return NULL;
    }
    Py_ssize_t d = PyList_Size(row0);
    if (d <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid Input!");
        return NULL;
    }

    double **mat = (double **)malloc((size_t)n * sizeof(double *));
    if (!mat) { PyErr_NoMemory(); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *row = PyList_GetItem(list2d, i);
        if (!PyList_Check(row) || PyList_Size(row) != d) {
            PyErr_SetString(PyExc_TypeError, "Invalid Input!");
            for (Py_ssize_t t = 0; t < i; t++) free(mat[t]);
            free(mat);
            return NULL;
        }
        mat[i] = (double *)malloc((size_t)d * sizeof(double));
        if (!mat[i]) {
            for (Py_ssize_t t = 0; t < i; t++) free(mat[t]);
            free(mat);
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t j = 0; j < d; j++) {
            PyObject *val = PyList_GetItem(row, j);
            double v = PyFloat_AsDouble(val);
            if (PyErr_Occurred()) {
                for (Py_ssize_t t = 0; t <= i; t++) free(mat[t]);
                free(mat);
                return NULL;
            }
            mat[i][j] = v;
        }
    }
    *out_rows = (int)n;
    *out_cols = (int)d;
    return mat;
}

static void free_cmatrix(double **m, int rows) {
    if (!m) return;
    for (int i = 0; i < rows; i++) free(m[i]);
    free(m);
}

/* Utility: convert C matrix to Python list of lists */
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

/* Wrapper: sym(points) -> A */
static PyObject *py_sym(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return NULL;
    int n, d;
    double **points = pylist_to_cmatrix(points_obj, &n, &d);
    if (!points) return NULL;
    double **W = sym(points, n, d);
    if (!W) { free_cmatrix(points, n); PyErr_NoMemory(); return NULL; }
    PyObject *res = cmatrix_to_pylist(W, n, n);
    free_cmatrix(points, n);
    free_cmatrix(W, n);
    return res;
}

/* Wrapper: ddg(points) -> D; build A inside */
static PyObject *py_ddg(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return NULL;
    int n, d;
    double **points = pylist_to_cmatrix(points_obj, &n, &d);
    if (!points) return NULL;
    double **W = sym(points, n, d);
    if (!W) { free_cmatrix(points, n); PyErr_NoMemory(); return NULL; }
    double **D = ddg(W, n);
    if (!D) { free_cmatrix(points, n); free_cmatrix(W, n); PyErr_NoMemory(); return NULL; }
    PyObject *res = cmatrix_to_pylist(D, n, n);
    free_cmatrix(points, n);
    free_cmatrix(W, n);
    free_cmatrix(D, n);
    return res;
}

/* Wrapper: norm(points) -> N; build A inside */
static PyObject *py_norm(PyObject *self, PyObject *args) {
    PyObject *points_obj;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return NULL;
    int n, d;
    double **points = pylist_to_cmatrix(points_obj, &n, &d);
    if (!points) return NULL;
    double **W = sym(points, n, d);
    if (!W) { free_cmatrix(points, n); PyErr_NoMemory(); return NULL; }
    double **N = norm(W, n);
    if (!N) { free_cmatrix(points, n); free_cmatrix(W, n); PyErr_NoMemory(); return NULL; }
    PyObject *res = cmatrix_to_pylist(N, n, n);
    free_cmatrix(points, n);
    free_cmatrix(W, n);
    free_cmatrix(N, n);
    return res;
}

/* Wrapper: symnmf(W, H_init, max_iter, epsilon) -> H */
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

static PyMethodDef SymnmfMethods[] = {
    {"sym",  (PyCFunction)py_sym, METH_VARARGS, "Compute similarity matrix A from points"},
    {"ddg",  (PyCFunction)py_ddg, METH_VARARGS, "Compute diagonal degree matrix from points"},
    {"norm", (PyCFunction)py_norm, METH_VARARGS, "Compute normalized similarity matrix from points"},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS, "Run SymNMF given A and initial H"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",   /* name of module */
    NULL,        /* module documentation */
    -1,          /* size of per-interpreter state of the module */
    SymnmfMethods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}

