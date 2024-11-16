Neural Networks and Learning Machines
===============

Rusty solutions to selected problems in [1], e.g. half moons dataset:

* prog1_6: Pereptron 
* prog2_8: Least Squares
* prog4_16: Multi Layer Perceptron
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
  -k, --kmeans        Kernel training: k-means (default: EM)
  -b, --hierarchical  Kernel initialisation: binary splitting from global mean (default: kmeans++)
  -d, --dist <D>      distance between halfmoons (e.g. -5.0 to 5.0 [default: -5]
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
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 0; #kernels: 20; cdist: 291.65
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 1; #kernels: 20; cdist:  7.08
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 2; #kernels: 20; cdist:  2.83
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 3; #kernels: 20; cdist:  3.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 4; #kernels: 20; cdist:  2.65
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 5; #kernels: 20; cdist:  2.30
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 6; #kernels: 20; cdist:  2.42
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 7; #kernels: 20; cdist:  2.79
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 8; #kernels: 20; cdist:  2.79
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 9; #kernels: 20; cdist:  2.37
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 10; #kernels: 20; cdist:  1.84
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 11; #kernels: 20; cdist:  1.37
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 12; #kernels: 20; cdist:  1.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 13; #kernels: 20; cdist:  0.74
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 14; #kernels: 20; cdist:  0.55
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 15; #kernels: 20; cdist:  0.42
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 16; #kernels: 20; cdist:  0.32
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 17; #kernels: 20; cdist:  0.26
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 18; #kernels: 20; cdist:  0.23
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 19; #kernels: 20; cdist:  0.23
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 20; #kernels: 20; cdist:  0.24
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 21; #kernels: 20; cdist:  0.27
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 22; #kernels: 20; cdist:  0.29
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 23; #kernels: 20; cdist:  0.29
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 24; #kernels: 20; cdist:  0.29
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 25; #kernels: 20; cdist:  0.29
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 26; #kernels: 20; cdist:  0.30
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 27; #kernels: 20; cdist:  0.33
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 28; #kernels: 20; cdist:  0.39
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 29; #kernels: 20; cdist:  0.46
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 30; #kernels: 20; cdist:  0.55
[2024-11-06T09:40:03Z WARN  nnlm::gmm] defunkt kernel 0 - removing
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 31; #kernels: 19; cdist:  0.56
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 32; #kernels: 19; cdist:  0.34
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 33; #kernels: 19; cdist:  0.14
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 34; #kernels: 19; cdist:  0.11
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 35; #kernels: 19; cdist:  0.10
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 36; #kernels: 19; cdist:  0.10
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 37; #kernels: 19; cdist:  0.10
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 38; #kernels: 19; cdist:  0.10
[2024-11-06T09:40:03Z WARN  nnlm::gmm] defunkt kernel 18 - removing
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 39; #kernels: 18; cdist:  0.09
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 40; #kernels: 18; cdist:  0.08
[2024-11-06T09:40:03Z WARN  nnlm::gmm] defunkt kernel 12 - removing
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 41; #kernels: 17; cdist:  0.08
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 42; #kernels: 17; cdist:  0.07
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 43; #kernels: 17; cdist:  0.06
[2024-11-06T09:40:03Z WARN  nnlm::gmm] defunkt kernel 11 - removing
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 44; #kernels: 16; cdist:  0.06
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 45; #kernels: 16; cdist:  0.05
[2024-11-06T09:40:03Z WARN  nnlm::gmm] defunkt kernel 14 - removing
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 46; #kernels: 15; cdist:  0.03
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 47; #kernels: 15; cdist:  0.03
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 48; #kernels: 15; cdist:  0.02
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 49; #kernels: 15; cdist:  0.02
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 50; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 51; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 52; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 53; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 54; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 55; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 56; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 57; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 58; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 59; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 60; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 61; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 62; #kernels: 15; cdist:  0.01
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 63; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 64; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 65; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 66; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 67; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 68; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 69; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 70; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 71; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 72; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 73; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 74; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 75; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 76; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 77; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 78; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 79; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 80; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 81; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 82; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 83; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 84; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 85; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 86; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 87; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 88; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 89; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 90; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 91; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 92; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 93; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 94; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 95; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 96; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 97; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 98; #kernels: 15; cdist:  0.00
[2024-11-06T09:40:03Z INFO  nnlm::gmm] EM training - ep: 99; #kernels: 15; cdist:  0.00

RBF
Weights:    29.34    26.61    12.20    27.14   -24.78   -11.76   -25.75   -23.74   -29.34   -23.28    22.36   -25.07    24.24   -14.06    32.16
0: GKernel
mean:     2.60     8.33
var:     10.77     1.14
1: GKernel
mean:    -8.71     4.49
var:      3.07     2.92
2: GKernel
mean:    -9.86     1.30
var:      3.07     0.64
3: GKernel
mean:    -5.60     8.22
var:      4.91     2.05
4: GKernel
mean:     0.33     2.53
var:      2.96     2.29
5: GKernel
mean:    10.77    -6.87
var:      4.60     0.33
6: GKernel
mean:     5.70    -3.74
var:      4.20     3.24
7: GKernel
mean:    19.53     2.92
var:      3.19     1.91
8: GKernel
mean:    17.64    -0.97
var:      3.93     2.79
9: GKernel
mean:    11.30    -3.53
var:      9.16     1.73
10: GKernel
mean:     7.42     5.77
var:      4.56     1.37
11: GKernel
mean:     2.61    -1.40
var:      3.45     3.32
12: GKernel
mean:     9.53     2.23
var:      3.03     1.91
13: GKernel
mean:    14.52    -4.09
var:      5.78     1.46
14: GKernel
mean:    -0.03    11.08
var:     10.94     1.02

[2024-11-06T09:40:44Z INFO  nnlm::rbf] Errors - Training data:: 0/3000 =   0.00%
[2024-11-06T09:40:44Z INFO  nnlm::rbf] Errors - Test data:: 0/2000 =   0.00%
```

