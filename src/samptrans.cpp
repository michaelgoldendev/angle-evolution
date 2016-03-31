/* 
 * File:   maintpd.cpp
 * Author: Eduardo
 *
 * Created on November 20, 2015, 11:38 PM
 */

#include <stdio.h>
#include <cstdlib>
#include <armadillo>
#include <iostream>

using namespace std;

// Declaration
arma::mat SampTransMWNOU(int N, arma::vec x0, double t, arma::vec mu, arma::vec alpha, arma::vec sigma, int maxK, double etrunc);

// Main
int main(int argc, char** argv) {
  
  /*
   *  Argument codification in argv: 
   *  N x01 x02 t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc
   */
  
  // Set parameters
  double N = atoi(argv[1]);
  arma::vec x0(2); 
  x0(0) = atof(argv[2]); x0(1) = atof(argv[3]);
  double t = atof(argv[4]);
  arma::vec mu(2); 
  mu(0) = atof(argv[5]); mu(1) = atof(argv[6]);
  arma::vec alpha(3); 
  alpha(0) = atof(argv[7]); alpha(1) = atof(argv[8]); alpha(2) = atof(argv[9]);
  arma::vec sigma(2);
  sigma(0) = atof(argv[10]); sigma(1) = atof(argv[11]);
  int maxK = atoi(argv[12]);
  double etrunc = atof(argv[13]);
  
  // Call function
  arma::mat result = SampTransMWNOU(N, x0, t, mu, alpha, sigma, maxK, etrunc);

  // Print result concatenating rows as a vector 
  vectorise(result, 1).print();

  return 0;
    
}

// Subroutine
arma::mat SampTransMWNOU(int N, arma::vec x0, double t, arma::vec mu, arma::vec alpha, arma::vec sigma, int maxK = 2, double etrunc = 50) {
  
  /*
  * Description: Sampling from the transition distribution of a MWN-OU diffusion (with diagonal diffusion matrix)
  * 
  * Arguments:
  *
  * - N: sample size.
  * - x0: a vector of length 2 containing the initial angles (phi0, psi0). They must be in [-PI, PI) so that the truncated wrapping by 
  *      maxK windings is able to capture periodicity.
  * - t: time between x0 and the samples.
  * - mu: a vector of length 2 with the mean parameter of the MWN-OU process. The mean of the MWN stationary distribution.
  * - alpha: vector of length 3 containing the A matrix of the drift of the MWN-OU process in the following codification: 
  *          A = [alpha[0], alpha[2] * sqrt(sigma[0] / sigma[1]); alpha[2] * sqrt(sigma[1] / sigma[0]), alpha[1]]. 
  *          This enforces that A^(-1) * Sigma is symmetric. Positive definiteness is guaranteed if 
  *          alpha[0] * alpha[1] > alpha[2] * alpha[2]. The function checks for it and, if violated, returns the sample
  *          from a close A^(-1) * Sigma that is positive definite.
  * - sigma: vector of length 2 containing the diagonal of Sigma, the diffusion matrix. Note that these are the *squares*
  *          (i.e. variances) of the diffusion coefficients that multiply the Wiener process.
  * - maxK: maximum number of winding number considered in the sampling from the approximated transition probability density.
  * - etrunc: truncation for exponential. exp(x) with x <= -etrunc is set to zero.
  * 
  * Warning: 
  * 
  *  - A combination of small etrunc (< 30) and low maxK (<= 1) can lead to NaNs produced by 0 / 0 in the weight computation. 
  *    This is specially dangerous if sigma is large and there are values in x or x0 outside [-PI, PI).
  *    
  * Value: 
  * 
  * - x: matrix of size N x 2 containing the samples of the trasition distribution conditioned on that 
  *      the process was at x0 at t instants ago. The samples are in [-PI, PI) x [-PI, PI).
  * 
  * Author: Eduardo García-Portugués (egarcia@math.ku.dk) 
  * 
  */
  
  /*
  * Create basic objects
  */
  
  // Create and initialize A
  double quo = sqrt(sigma(0) / sigma(1));
  arma::mat A(2, 2); 
  A(0, 0) = alpha(0); 
  A(1, 1) = alpha(1); 
  A(0, 1) = alpha(2) * quo;
  A(1, 0) = alpha(2) / quo;
  
  // Create and initialize Sigma
  arma::mat Sigma = diagmat(sigma);
  
  // Sequence of winding numbers
  const int lk = 2 * maxK + 1;
  const int lk2 = lk * lk;
  arma::vec twokpi = arma::linspace<arma::vec>(-maxK * 2 * M_PI, maxK * 2 * M_PI, lk);
  
  // Bivariate vector (2 * K1 * PI, 2 * K2 * PI) for weighting
  arma::vec twokepivec(2);
  
  // Bivariate vector (2 * K1 * PI, 2 * K2 * PI) for wrapping
  arma::vec twokapivec(2);
  
  /*
  * Computation of A^(-1) * Sigma (we do not check positive definiteness)
  */
  
  // Inverse of 1/2 * A^(-1) * Sigma: 2 * Sigma^(-1) * A
  arma::mat invSigmaA = 2 * diagmat(1 / diagvec(Sigma)) * A;
  
  // Log-normalizing constant for the Gaussian with covariance SigmaA
  double lognormconstSigmaA = -log(2 * M_PI) + log(det(invSigmaA)) / 2;
  
  /*
  * Computation of Gammat and exp(-t * A) analytically
  */
  
  // A * Sigma
  arma::mat ASigma = A * Sigma;
  
  // A * Sigma * A^T
  arma::mat ASigmaA = ASigma * A.t();
  
  // Update with A * Sigma + Sigma * A^T
  ASigma += ASigma.t();
  
  // Quantities for computing exp(-t * A)
  double s = trace(A) / 2;
  double q = sqrt(fabs(det(A - s * arma::eye<arma::mat>(2, 2))));
  
  // Avoid indetermination in sinh(q * t) / q when q == 0
  if(q == 0){
    
    q = 1e-6;
    
  }
  
  // Repeated terms in the analytic integrals
  double q2 = q * q;
  double s2 = s * s;
  double est = exp(s * t);
  double e2st = est * est;
  double inve2st = 1 / e2st;
  double c2st = exp(2 * q * t);
  double s2st = (c2st - 1/c2st) / 2;
  c2st = (c2st + 1/c2st) / 2;
  
  // Integrals
  double cte = inve2st / (4 * q2 * s * (s2 - q2));
  double integral1 = cte * (- s2 * (3 * q2 + s2) * c2st - q * s * (q2 + 3 * s2) * s2st - q2 * (q2 - 5 * s2) * e2st + (q2 - s2) *  (q2 - s2));
  double integral2 = cte * s * ((q2 + s2) * c2st + 2 * q * s * s2st - 2 * q2 * e2st + q2 - s2);
  double integral3 = cte * (- s * (s * c2st + q * s2st) + (e2st - 1) * q2 + s2);
  
  // Gammat    
  arma::mat Gammat = integral1 * Sigma + integral2 * ASigma + integral3 * ASigmaA;
  
  // Matrix exp(-t*A)
  double eqt = exp(q * t);
  double cqt = (eqt + 1/eqt) / 2;
  double sqt = (eqt - 1/eqt) / 2;
  arma::mat ExptA = ((cqt + s * sqt / q) * arma::eye<arma::mat>(2, 2) - sqt / q * A) / est;
  
  // Inverse and log-normalizing constant for the Gammat
  arma::mat invGammat = inv_sympd(Gammat);
  
  /* 
  * Weights of the winding numbers for each data point
  */
  
  // We store the weights in a vector
  arma::vec weightswindsinitial(lk2);
  
  // Matrix of different mut's depending on the windings
  arma::mat mut = arma::zeros(lk2, 2);
  mut.each_row() = (mu + ExptA * (x0 - mu)).t();
  
  // Compute the factors in the exponent that do not depend on the windings
  arma::vec xmu = x0 - mu;
  arma::vec xmuinvSigmaA = invSigmaA * xmu;
  double xmuinvSigmaAxmudivtwo = dot(xmuinvSigmaA, xmu) / 2;
  
  // Loop in the winding weight K1
  for(int wek1 = 0; wek1 < lk; wek1++){
    
    // 2 * K1 * PI
    twokepivec(0) = twokpi(wek1); 
    
    // Compute once the index
    int wekl1 = wek1 * lk;
    
    // Loop in the winding weight K2  
    for(int wek2 = 0; wek2 < lk; wek2++){
      
      // 2 * K2 * PI
      twokepivec(1) = twokpi(wek2);
      
      // Decomposition of the exponent
      double exponent = xmuinvSigmaAxmudivtwo + dot(xmuinvSigmaA, twokepivec) + dot(invSigmaA * twokepivec, twokepivec) / 2 - lognormconstSigmaA;
      
      // Truncate the negative exponential and store mut
      if(exponent > etrunc){
        
        weightswindsinitial(wek1 * lk + wek2) = 0;
        mut.row(wekl1 + wek2).fill(0);
        
      }else{
        
        weightswindsinitial(wekl1 + wek2) = exp(-exponent);
        mut.row(wekl1 + wek2) += (ExptA * twokepivec).t();
        
      }
      
    }
    
  }
  
  // Standardize weights for the tpd
  weightswindsinitial /= sum(weightswindsinitial);
  
  /*
  * Sample the windings at x0 and select mut
  */
  
  // Probabilities of windings
  arma::vec probs = arma::cumsum(weightswindsinitial);
  
  // Sample uniforms in [0, 1]. There is no weigthed sampling in Armadillo!
  arma::vec runif = arma::randu(N);
  
  // Matrix of mut for each x
  arma::mat mutx(N, 2);
  
  // Assign muts
  for(int i = 0; i < N; i++){
    
    // Choose windings with probabilities weightswindsinitial
    for (int j = 0; j < lk2; j++) {
      
      // Weighted sampling
      if (runif(i) <= probs(j)) {
        
        // Set mut for x
        mutx.row(i) = mut.row(j);
        
        // Skip the j loop
        break;
        
      }
      
    }
    
  }
  
  /*
  * Sample correlated normal variables
  */
  
  // Sample of independent N(0, 1)
  arma::mat x = arma::randn(N, 2);
  
  // Cholesky decomposition for correlate random sample
  arma::mat ch = chol(Gammat);
  
  // Correlate random variables
  x = x * ch;
  
  // Recenter depending on mut
  x += mutx;
  
  // Wrap (convert to [-PI,PI) x [-PI,PI))
  x -= floor((x + M_PI) / (2 * M_PI)) * (2 * M_PI);
  
  return x;
  
}
