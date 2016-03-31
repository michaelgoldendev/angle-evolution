
How to compile and run the C++ codes from console
=================================================

### loglik

 - Compilation: g++ loglik.cpp -o loglik -larmadillo -O2
 - Codification arguments: t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc pairs
 - Running example: ./loglik 1 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1 0 1 -1 -1
 - Output: -19.7968

 ### logliktime

  - Compilation: g++ logliktime.cpp -o logliktime -larmadillo -O2
  - Codification arguments: t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc pairs
  - Running example: ./loglik 1 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1 0 1 -1 -1
  - Output: -19.7968

### sampstat

 - Compilation: g++ sampstat.cpp -o sampstat -larmadillo -O2
 - Codification arguments: N mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2
 - Running example: ./sampstat 5 0 -0.57 1 2 0 1 1.0
 - Output: 1.1325  -0.5104  -0.1832  -0.4755   0.1236  -0.3409  -1.0599  -1.7549  -0.2136  -0.3725

### samptrans

 - Compilation: g++ samptrans.cpp -o samptrans -larmadillo -O2
 - Codification arguments: N x01 x02 t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc
 - Running example: ./samptrans 5 0 0 1 0 -0.57 1 2 0 1 1.0 2 50
 - Output: 0.7660   0.1412   0.7678  -0.3187   0.4122   0.5109  -0.1105  -0.8379   0.0494  -0.5586

 ### samptranstime

  - Compilation: g++ samptranstime.cpp -o samptranstime -larmadillo -O2
  - Codification arguments: N mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x0 t
  - Running example: ./samptranstime 5 1 0 -0.57 1 2 0 1 1.0 2 50 0 0 1
  - Output:

### densstat

 - Compilation: g++ densstat.cpp -o densstat -larmadillo -O2
 - Codification arguments: mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc
 - Running example: ./densstat 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1
 - Output: 0.0012   0.0033

### denstrans

 - Compilation: g++ denstrans.cpp -o denstrans -larmadillo -O2
 - Codification arguments: x01 x02 t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x
 - Running example: ./denstrans 0 0 1 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1
 - Output: 0.0016   0.0052

 ## denstranstime

  - Compilation: g++ denstranstime.cpp -o denstranstime -larmadillo -O2
  - Codification arguments: x01 x02 t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x
  - Running example: ./denstrans 0 0 1 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1
  - Output: 0.0016   0.0052
