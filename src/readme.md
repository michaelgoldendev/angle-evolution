
How to compile and run the C++ codes from console 
=================================================

### maintpd

 - Compilation: g++ maintpd.cpp -o maintpd -larmadillo
 - Codification arguments: t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc pairs
 - Running example: ./maintpd 1 0 -0.57 1 2 0 1 1.0 1 25 -1 1 0 1 0 1 -1 -1

### mainsamp

 - Compilation: g++ mainsamp.cpp -o mainsamp -larmadillo
 - Codification arguments: N mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2
 - Running example: ./mainsamp 100 0 -0.57 1 2 0 1 1.0

### maindens

 - Compilation: g++ maindens.cpp -o maindens -larmadillo
 - Codification arguments: mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x
 - Running example: ./maindens 0 -0.57 1 2 0 1 1.0 1 25 -1 1 0 1