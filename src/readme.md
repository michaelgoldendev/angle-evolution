
How to compile and run the C++ codes from console
=================================================

## First versions

### loglik

 - Compilation: `g++ loglik.cpp -o loglik -larmadillo -O2`
 - Codification arguments: `t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc pairs`
 - Running example: `./loglik 1 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1 0 1 -1 -1`
 - Output: `-19.7968`

### sampstat

 - Compilation: `g++ sampstat.cpp -o sampstat -larmadillo -O2`
 - Codification arguments: `N mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2`
 - Running example: `./sampstat 5 0 -0.57 1 2 0 1 1.0`
 - Output: `1.1325  -0.5104  -0.1832  -0.4755   0.1236  -0.3409  -1.0599  -1.7549  -0.2136  -0.3725`

### samptrans

 - Compilation: `g++ samptrans.cpp -o samptrans -larmadillo -O2`
 - Codification arguments: `N x01 x02 t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc`
 - Running example: `./samptrans 5 0 0 1 0 -0.57 1 2 0 1 1.0 2 50
 - Output: `0.7660   0.1412   0.7678  -0.3187   0.4122   0.5109  -0.1105  -0.8379   0.0494  -0.5586`

### densstat

 - Compilation: `g++ densstat.cpp -o densstat -larmadillo -O2`
 - Codification arguments: `mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc`
 - Running example: `./densstat 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1`
 - Output: `0.0012   0.0033`

### denstrans

 - Compilation: `g++ denstrans.cpp -o denstrans -larmadillo -O2`
 - Codification arguments: `x01 x02 t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x`
 - Running example: `./denstrans 0 0 1 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1`
 - Output: `0.0016   0.0052`

## Newer versions with vectorized times - C++ 11 dependent!

They are called with the same arguments but allowing `t` and `x0` to be a vector and a matrix, respectively. Note the change in order of these parameters.

### logliktime

 - Compilation: `g++ -std=c++11 logliktime.cpp -o logliktime -larmadillo -O2`
 - Codification arguments: `mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc pairs t`
 - Running example 1: `./logliktime 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1 0 1 -1 -1 1 1` (same output as before)
 - Output 1: `-19.7968`
 - Running example 2: `./logliktime 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1 0 1 -1 -1 1 2` (two `t`'s)
 - Output 2: `-19.3116`
 - Running example 3: `./logliktime 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1 1; ./logliktime 0 -0.57 1 2 0 1 1.0 2 50 0 1 -1 -1 2` (the sum of the single `t`'s)
 - Output 3: `-11.3552 -7.9564`

### samptranstime

 - Compilation: `g++ -std=c++11 samptranstime.cpp -o samptranstime -larmadillo -O2`
 - Codification arguments: `N mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x0 t`
 - Running example 1: `./samptranstime 5 0 -0.57 1 2 0 1 1.0 2 50 0 0 1` (single `x0` and `t`)
 - Output 1:
    ```
    [cube slice 0]
    0.7660   0.1412
    0.7678  -0.3187
    0.4122   0.5109
   -0.1105  -0.8379
    0.0494  -0.5586
    ```
 - Running example 2: `./samptranstime 5 0 -0.57 1 2 0 1 1.0 2 50 0 0 0 0 1 2` (same `x0` with different `t`'s)
 - Output 2:
    ```
    [cube slice 0]
      -1.5582  -0.8081
       0.2597  -0.8718
      -0.4369   0.3648
       0.0291  -0.4209
      -0.1126  -0.1445

    [cube slice 1]
       0.8436   0.2810
       0.0656  -0.4225
      -0.4848  -0.4720
       0.1829   0.0335
      -0.4918  -0.9910
    ```
### denstranstime

 - Compilation: `g++ -std=c++11 denstranstime.cpp -o denstranstime -larmadillo -O2`
 - Codification arguments: `mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x0 x t`
 - Running example 1: `./denstranstime 0 -0.57 1 2 0 1 1.0 2 50 0 0 0 0 -1 1 0 1 1 1` (same `t`)
 - Output 1: `0.0016   0.0052`
 - Running example 2: `./denstranstime 0 -0.57 1 2 0 1 1.0 2 50 0 0 0 0 -1 1 0 1 1 2` (different `t`)
 - Output 2: `0.0016   0.0035`
