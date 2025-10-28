SymNMF Project
Description

This project implements the Symmetric Non-negative Matrix Factorization (SymNMF) algorithm in both C and Python, using the C implementation for performance and a Python wrapper for usability.

SymNMF is used for clustering and dimensionality reduction of symmetric, non-negative matrices — commonly similarity matrices derived from datasets.

Project Structure
.
├── symnmf.py            # Python wrapper for the SymNMF algorithm
├── symnmfmodule.c       # C extension module (Python–C interface)
├── symnmf.c             # Core C implementation of SymNMF
├── symnmf.h             # Header file for C implementation
├── setup.py             # Build script for the Python extension module
├── tester.py            # Test script (optional)
└── README.md

Compilation and Installation
Build the extension

Run the following command from the project root:

python3 setup.py build_ext --inplace


This will compile the C code and create a shared object file (e.g., symnmf.cpython-311-x86_64-linux-gnu.so) that can be imported directly in Python.

Usage
Command line

Run the program using:

python3 symnmf.py <goal> <input_file>


Where:

<goal> is one of the following:

sym — compute and print the similarity matrix.

ddg — compute and print the diagonal degree matrix.

norm — compute and print the normalized Laplacian matrix.

symnmf — perform the full SymNMF algorithm.

<input_file> — path to a CSV file containing the data points (each line is a point in ℝᵈ).

Example:

python3 symnmf.py symnmf input.txt

Algorithm Overview

Given a symmetric, non-negative matrix A ∈ ℝⁿˣⁿ, SymNMF aims to find a non-negative matrix H ∈ ℝⁿˣᵏ that minimizes:

∥
𝐴
−
𝐻
𝐻
𝑇
∥
𝐹
2
∥A−HH
T
∥
F
2
	​


This is done iteratively using the following update rule:

𝐻
𝑖
𝑗
←
𝐻
𝑖
𝑗
×
(
𝐴
𝐻
)
𝑖
𝑗
(
𝐻
𝐻
𝑇
𝐻
)
𝑖
𝑗
H
ij
	​

←H
ij
	​

×
(HH
T
H)
ij
	​

(AH)
ij
	​

	​


The process repeats until convergence or until the maximum number of iterations is reached.

Implementation Details
In C (symnmf.c)

Handles:

Reading matrices from input

Matrix multiplication, normalization, and update rules

Memory management

Uses only standard libraries: stdio.h, stdlib.h, math.h

In Python (symnmf.py)

Handles:

Parsing arguments and input

Data preprocessing

Calling C functions through the compiled extension module

Printing formatted results

Output

The program prints the resulting matrix to stdout, formatted as:

x11,x12,...,x1k
x21,x22,...,x2k
...
xn1,xn2,...,xnk


Each value is rounded to four decimal digits.

Example
Input (input.txt)
1.0,1.0
1.0,0.0
0.0,1.0
0.0,0.0

Run
python3 symnmf.py symnmf input.txt

Output (example)
0.9124,0.2341
0.8773,0.2010
0.1102,0.9345
0.1120,0.8873
