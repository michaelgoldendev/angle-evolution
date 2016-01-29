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
arma::mat SampStatMWNOU(int N, arma::vec mu, arma::vec alpha, arma::vec sigma);

// Main
int main(int argc, char** argv) {
  
  /*
   *  Argument codification in argv: 
   *  N mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2
   */
  
  // Set parameters
  double N = atoi(argv[1]);
  arma::vec mu(2); 
  mu(0) = atof(argv[2]); mu(1) = atof(argv[3]);
  arma::vec alpha(3); 
  alpha(0) = atof(argv[4]); alpha(1) = atof(argv[5]); alpha(2) = atof(argv[6]);
  arma::vec sigma(2); 
  sigma(0) = atof(argv[7]); sigma(1) = atof(argv[8]);
  
  // Call function
  arma::mat result = SampStatMWNOU(N, mu, alpha, sigma);

  // Print result concatenating rows as a vector 
  vectorise(result, 1).print();

  return 0;
    
}

// Subroutine
arma::mat SampStatMWNOU(int N, arma::vec mu, arma::vec alpha, arma::vec sigma) {
  
  /*
  * Description: Sampling from the stationary distribution of a MWN-OU diffusion (with diagonal diffusion matrix)
  * 
  * Arguments:
  *
  * - N: sample size.
  * - mu: a vector of length 2 with the mean parameter of the MWN-OU process. The mean of the MWN stationary distribution.
  *       It must be in [PI, PI) x [PI, PI).
  * - alpha: vector of length 3 containing the A matrix of the drift of the MWN-OU process in the following codification: 
  *          A = [alpha[0], alpha[2] * sqrt(sigma[0] / sigma[1]); alpha[2] * sqrt(sigma[1] / sigma[0]), alpha[1]]. 
  *          This enforces that A^(-1) * Sigma is symmetric. Positive definiteness is guaranteed if 
  *          alpha[0] * alpha[1] > alpha[2] * alpha[2]. The function checks for it and, if violated, returns the sample
  *          from a close A^(-1) * Sigma that is positive definite.
  * - sigma: vector of length 2 containing the diagonal of Sigma, the diffusion matrix. Note that these are the *squares*
  *          (i.e. variances) of the diffusion coefficients that multiply the Wiener process.
  * 
  * Value: 
  * 
  * - x: matrix of size N x 2 containing the samples of the stationary distribution in [-PI, PI) x [-PI, PI)
  * 
  * Author: Eduardo García-Portugués (egarcia@math.ku.dk) 
  * 
  */
  
  /*
  * Compute A^(-1) * Sigma and check for positive definiteness
  */
  
  // Create and initialize A
  double quo = sqrt(sigma(0) / sigma(1));
  arma::mat A(2, 2); 
  A(0, 0) = alpha(0); 
  A(1, 1) = alpha(1); 
  A(0, 1) = alpha(2) * quo;
  A(1, 0) = alpha(2) / quo;
  
  // Only positive definiteness can be violated with the parametrization of A
  double testalpha = alpha(0) * alpha(1) - alpha(2) * alpha(2);
  
  // Check positive definiteness 
  if(testalpha <= 0) {
    
    // Update alpha(2) such that testalpha > 0
    alpha(2) = std::signbit(alpha(2)) * sqrt(alpha(0) * alpha(1)) * 0.9999;
    
    // Reset A to a matrix with positive determinant and continue
    A(0, 1) = alpha(2) * quo;
    A(1, 0) = alpha(2) / quo;
    
  }
  
  // Create invASigma
  arma::mat invASigma = inv_sympd(A) * diagmat(sigma) / 2;
  
  /*
   * Sample correlated normal variables
   */
  
  // Sample of independent N(0, 1)
  arma::mat x = arma::randn(N, 2);
  
  // Cholesky decomposition for correlate random sample
  arma::mat ch = chol(invASigma);
  
  // Correlate random variables
  x = x * ch;
  
  // Recenter
  x.each_row() += mu.t();
  
  // Wrap (convert to [-PI, PI) x [-PI, PI))
  x -= floor((x + M_PI) / (2 * M_PI)) * (2 * M_PI);
  
  return x;
  
}








