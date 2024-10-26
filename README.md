Neural Networks and Learning Machines
===============

Rusty solutions to selected problems in [1], e.g. half moon dataset:

* prog1_6: Pereptron 
* prog2_8: Least Squares
* prog5_11: Radial Basis Function network; kmeans + LMS/RHS training of weights

References:
-----------
[1] Neural Networks and Learning Machines, Third Edition, Simon Haykin

Run:
----

See src/bin - e.g. problem 5.11, train a Radial Basis Function network on the halfmoon classification task: 

```
% cargo run --bin prog5_11 --release -- -h

Usage: prog5_11 [OPTIONS]

Options:
  -l, --lms           Weight training: Least Mean Squares (default: Recursive Least Squares)
  -k, --kmeans        Kernel training: k-means (default: Expectation Maximisation)
  -d, --dist <D>      distance between halfmoons (e.g. -5.0 to 5.0) [default: -5]
  -n, --nkernels <N>  number of RBF kernels [default: 20]
  -h, --help          Print help
  -V, --version       Print version

```
```
% cargo run --bin prog5_11 --release
```

![PNG](https://raw.githubusercontent.com/jesper-olsen/nnlm/refs/heads/master/Assets/prog5_11_kernels.png)

```
[snip]
RBF
Weights:    12.32   -12.24    23.82   -25.24    18.99   -16.80    25.61   -28.16     4.89   -16.85     0.75    -7.51    25.58   -14.91     3.64    -4.54    17.58   -16.18    15.79    -0.52
0: GKernel
mean:    -9.48     1.31
var:      2.79     0.46
1: GKernel
mean:    10.10    -6.69
var:      4.89     0.31
2: GKernel
mean:    -0.24    11.01
var:      4.77     1.07
3: GKernel
mean:     3.92    -2.31
var:      4.55     1.27
4: GKernel
mean:    -5.75     7.70
var:      1.84     1.94
5: GKernel
mean:    19.41     3.51
var:      3.07     0.84
6: GKernel
mean:     9.68     2.35
var:      2.56     2.17
7: GKernel
mean:     0.66     1.98
var:      2.71     2.28
8: GKernel
mean:    -9.43     7.52
var:      0.56     0.54
9: GKernel
mean:    17.61    -1.81
var:      2.07     1.49
10: GKernel
mean:    -5.32    10.99
var:      0.28     0.04
11: GKernel
mean:    18.24     0.93
var:      3.51     0.18
12: GKernel
mean:     3.34     8.35
var:      4.09     1.53
13: GKernel
mean:    14.29    -3.93
var:      1.70     1.48
14: GKernel
mean:    -2.17     7.62
var:      0.43     0.34
15: GKernel
mean:     4.67    -5.52
var:      0.56     0.33
16: GKernel
mean:     7.50     6.22
var:      2.37     1.65
17: GKernel
mean:     9.64    -3.67
var:      1.91     1.24
18: GKernel
mean:    -8.71     4.01
var:      3.18     0.78
19: GKernel
mean:    12.75    -7.13
var:      0.12     0.08

Errors - Training data:: 0/1000 =   0.00%
Errors - Test data:: 0/2000 =   0.00%
```

