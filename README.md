Neural Networks and Learning Machines
===============

Rusty solutions to selected problems in [1], e.g. half moon dataset:

* prog1_6: Pereptron 
* prog2_8: Least Squares
* prog5_11: Radial Basis Function network; kmeans/EM kernel training + LMS/RHS weight training 

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
  -s, --seed <S>      seed rng [default: 12]
  -n, --nkernels <N>  number of RBF kernels [default: 20]
  -h, --help          Print help
  -V, --version       Print version

```
```
% cargo run --bin prog5_11 --release
```

![PNG](https://raw.githubusercontent.com/jesper-olsen/nnlm/refs/heads/master/Assets/prog5_11_kernels.png)

```
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 0; cdist: 291.65
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 1; cdist:  8.24
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 2; cdist:  2.54
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 3; cdist:  2.99
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 4; cdist:  2.69
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 5; cdist:  2.33
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 6; cdist:  2.38
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 7; cdist:  2.62
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 8; cdist:  2.51
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 9; cdist:  2.04
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 10; cdist:  1.50
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 11; cdist:  1.08
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 12; cdist:  0.78
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 13; cdist:  0.58
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 14; cdist:  0.45
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 15; cdist:  0.37
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 16; cdist:  0.31
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 17; cdist:  0.28
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 18; cdist:  0.26
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 19; cdist:  0.26
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 20; cdist:  0.27
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 21; cdist:  0.29
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 22; cdist:  0.32
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 23; cdist:  0.34
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 24; cdist:  0.33
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 25; cdist:  0.31
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 26; cdist:  0.28
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 27; cdist:  0.27
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 28; cdist:  0.25
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 29; cdist:  0.25
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 30; cdist:  0.24
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 31; cdist:  0.24
[2024-10-31T12:51:02Z WARN  nnlm::gmm] defunkt kernel 0 - removing
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 32; cdist:  0.18
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 33; cdist:  0.16
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 34; cdist:  0.15
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 35; cdist:  0.13
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 36; cdist:  0.12
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 37; cdist:  0.12
[2024-10-31T12:51:02Z WARN  nnlm::gmm] defunkt kernel 2 - removing
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 38; cdist:  0.12
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 39; cdist:  0.11
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 40; cdist:  0.13
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 41; cdist:  0.14
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 42; cdist:  0.15
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 43; cdist:  0.17
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 44; cdist:  0.18
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 45; cdist:  0.19
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 46; cdist:  0.18
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 47; cdist:  0.17
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 48; cdist:  0.15
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 49; cdist:  0.13
[2024-10-31T12:51:02Z WARN  nnlm::gmm] defunkt kernel 10 - removing
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 50; cdist:  0.10
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 51; cdist:  0.10
[2024-10-31T12:51:02Z WARN  nnlm::gmm] defunkt kernel 14 - removing
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 52; cdist:  0.06
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 53; cdist:  0.07
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 54; cdist:  0.07
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 55; cdist:  0.08
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 56; cdist:  0.09
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 57; cdist:  0.10
[2024-10-31T12:51:02Z WARN  nnlm::gmm] defunkt kernel 14 - removing
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 58; cdist:  0.09
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 59; cdist:  0.23
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 60; cdist:  0.05
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 61; cdist:  0.04
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 62; cdist:  0.04
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 63; cdist:  0.04
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 64; cdist:  0.04
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 65; cdist:  0.05
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 66; cdist:  0.05
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 67; cdist:  0.05
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 68; cdist:  0.05
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 69; cdist:  0.05
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 70; cdist:  0.05
[2024-10-31T12:51:02Z WARN  nnlm::gmm] defunkt kernel 14 - removing
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 71; cdist:  0.03
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 72; cdist:  0.07
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 73; cdist:  0.03
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 74; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 75; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 76; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 77; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 78; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 79; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 80; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 81; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 82; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 83; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 84; cdist:  0.01
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 85; cdist:  0.01
[2024-10-31T12:51:02Z WARN  nnlm::gmm] defunkt kernel 10 - removing
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 86; cdist:  0.01
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 87; cdist:  0.06
[2024-10-31T12:51:02Z WARN  nnlm::gmm] defunkt kernel 4 - removing
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 88; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 89; cdist:  0.17
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 90; cdist:  0.03
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 91; cdist:  0.02
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 92; cdist:  0.01
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 93; cdist:  0.01
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 94; cdist:  0.01
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 95; cdist:  0.01
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 96; cdist:  0.01
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 97; cdist:  0.01
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 98; cdist:  0.01
[2024-10-31T12:51:02Z INFO  nnlm::gmm] EM training - ep: 99; cdist:  0.01

RBF
Weights:    52.40    35.89    30.86   -33.19   -20.76    -8.27   -22.89   -30.50    28.56   -41.58    26.70   -22.84
0: GKernel
mean:     0.92     9.50
var:     10.54     3.00
1: GKernel
mean:    -9.24     3.32
var:      3.15     4.06
2: GKernel
mean:    -6.01     8.11
var:      4.32     3.12
3: GKernel
mean:     0.61     2.04
var:      3.19     3.25
4: GKernel
mean:    10.21    -6.59
var:      7.89     0.59
5: GKernel
mean:    20.06     4.22
var:      3.40     0.30
6: GKernel
mean:    19.22     1.71
var:      3.27     2.05
7: GKernel
mean:    13.16    -3.67
var:      8.34     1.57
8: GKernel
mean:     6.96     6.40
var:      4.72     2.16
9: GKernel
mean:     4.66    -2.77
var:      5.72     3.18
10: GKernel
mean:     9.60     2.45
var:      3.03     2.33
11: GKernel
mean:    17.37    -1.73
var:      3.58     2.83

[2024-10-31T12:43:24Z INFO  nnlm::rbf] Errors - Training data:: 1/3000 =   0.03%
[2024-10-31T12:43:24Z INFO  nnlm::rbf] Errors - Test data:: 1/2000 =   0.05%

```

