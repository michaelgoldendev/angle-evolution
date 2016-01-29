
How to compile and run the C++ codes from console 
=================================================

### mainloglik

 - Compilation: g++ mainloglik.cpp -o maintpd -larmadillo -O2
 - Codification arguments: t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc pairs
 - Running example: ./mainloglik 1 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1 0 1 -1 -1
 - Output: -19.7968

### mainsampstat

 - Compilation: g++ mainsampstat.cpp -o mainsampstat -larmadillo -O2
 - Codification arguments: N mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2
 - Running example: ./mainsampstat 5 0 -0.57 1 2 0 1 1.0
 - Output: 1.1325  -0.5104  -0.1832  -0.4755   0.1236  -0.3409  -1.0599  -1.7549  -0.2136  -0.3725

### mainsamptrans

 - Compilation: g++ mainsamptrans.cpp -o mainsamptrans -larmadillo -O2
 - Codification arguments: N x01 x02 t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc
 - Running example: ./mainsamptrans 5 0 0 1 0 -0.57 1 2 0 1 1.0 2 50
 - Output: 0.7660   0.1412   0.7678  -0.3187   0.4122   0.5109  -0.1105  -0.8379   0.0494  -0.5586


### maindensstat

 - Compilation: g++ maindensstat.cpp -o maindensstat -larmadillo -O2
 - Codification arguments: mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc
 - Running example: ./maindensstat 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1
 - Output: 0.0012   0.0033

### maindenstrans

 - Compilation: g++ maindenstrans.cpp -o maindenstrans -larmadillo -O2
 - Codification arguments: x01 x02 t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x
 - Running example: ./maindenstrans 0 0 1 0 -0.57 1 2 0 1 1.0 2 50 -1 1 0 1
 - Output: 0.0016   0.0052

