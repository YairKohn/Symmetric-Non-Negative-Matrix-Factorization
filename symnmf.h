#ifndef SYM_NMF_H
#define SYM_NMF_H

/* Public API implemented in symnmf.c */

double **sym(double **points, int n, int d);
double **ddg(double **W, int n);
double **norm(double **W, int n);

/* Full SymNMF factorization: returns a newly allocated H (n x k) */
double **symnmf(double **W, double **H_init, int n, int k, int max_iter, double epsilon);

#endif /* SYM_NMF_H */
