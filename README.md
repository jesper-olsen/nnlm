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
% cargo run --bin prog5_11
```

![PNG](https://raw.githubusercontent.com/jesper-olsen/nnlm/refs/heads/master/Assets/prog5_11_kernels.png)

```
[snip]
RBF
Weights:    14.50   -12.79    10.24   -24.57    16.41   -20.04    36.68   -28.23     9.11   -23.49     2.58    -0.03    14.66   -14.66     4.10    -5.72    24.64   -15.07    11.54    -0.38
0: GKernal
mean:    -9.46     1.53
var:      2.77     0.66
1: GKernal
mean:    10.12    -6.67
var:      4.98     0.32
2: GKernal
mean:    -1.69    10.69
var:      0.67     1.49
3: GKernal
mean:     4.00    -2.20
var:      5.03     1.07
4: GKernal
mean:    -5.48     7.54
var:      1.32     1.87
5: GKernal
mean:    19.56     3.24
var:      3.05     1.30
6: GKernal
mean:     9.35     3.29
var:      2.66     4.37
7: GKernal
mean:     0.66     1.98
var:      2.71     2.28
8: GKernal
mean:    -9.15     7.33
var:      0.81     1.19
9: GKernal
mean:    17.70    -0.83
var:      2.10     2.61
10: GKernal
mean:    -4.97    10.97
var:      0.55     0.18
11: GKernal
mean:    16.60     1.23
var:      0.00     0.03
12: GKernal
mean:     1.36     9.48
var:      0.72     3.01
13: GKernal
mean:    14.40    -3.87
var:      1.59     1.48
14: GKernal
mean:    -1.88     7.56
var:      0.80     0.22
15: GKernal
mean:     4.68    -5.32
var:      0.71     0.45
16: GKernal
mean:     5.34     7.92
var:      1.96     2.87
17: GKernal
mean:     9.91    -3.72
var:      1.78     1.13
18: GKernal
mean:    -8.58     4.11
var:      3.36     0.39
19: GKernal
mean:    12.78    -7.26
var:      0.14     0.04

Errors - Training data:: 0/1000 =   0.00%
Errors - Test data:: 0/2000 =   0.00%
```

