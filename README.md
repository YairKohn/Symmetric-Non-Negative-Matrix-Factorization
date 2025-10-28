# SymNMF Project

## 📘 Description
This project implements the **Symmetric Non-negative Matrix Factorization (SymNMF)** algorithm in both **C** and **Python**, using the C implementation for performance and a Python wrapper for usability.  

SymNMF is used for **clustering** and **dimensionality reduction** of symmetric, non-negative matrices (for example, similarity matrices).

---

## 📂 Project Structure
```
symnmf.py          # Python wrapper for the SymNMF algorithm  
symnmfmodule.c     # C extension module (Python–C interface)  
symnmf.c           # Core C implementation of SymNMF  
symnmf.h           # Header file for C implementation  
setup.py           # Build script for the Python extension module  
tester.py          # Optional test script  
README.md
```

---

## ⚙️ Build and Installation

### Build the extension
Run:
```bash
python3 setup.py build_ext --inplace
```

This compiles the C code and generates a `.so` file that can be imported in Python.

---

## ▶️ Usage
Run:
```bash
python3 symnmf.py <goal> <input_file>
```

**Arguments:**
- `<goal>` — one of:
  - `sym` : compute similarity matrix  
  - `ddg` : compute diagonal degree matrix  
  - `norm` : compute normalized Laplacian  
  - `symnmf` : run the full SymNMF algorithm  
- `<input_file>` — path to CSV file containing the data points  

**Example:**
```bash
python3 symnmf.py symnmf input.txt
```

---

## 🧮 Algorithm Overview
Given a symmetric, non-negative matrix **A ∈ ℝⁿˣⁿ**, SymNMF finds a non-negative matrix **H ∈ ℝⁿˣᵏ** minimizing:

```
||A - H·Hᵀ||²_F
```

Update rule:

```
H ← H * ((A·H) / (H·Hᵀ·H))
```

The process repeats until convergence or until the maximum number of iterations is reached.

---

## 🧾 Output
The program prints the resulting matrix to **stdout**, formatted as:
```
x11,x12,...,x1k
x21,x22,...,x2k
...
xn1,xn2,...,xnk
```
Values are rounded to **four decimal digits**.

---

## 💡 Example

**Input (`input.txt`):**
```
1.0,1.0
1.0,0.0
0.0,1.0
0.0,0.0
```

**Run:**
```bash
python3 symnmf.py symnmf input.txt
```

**Output (example):**
```
0.9124,0.2341
0.8773,0.2010
0.1102,0.9345
0.1120,0.8873
```

