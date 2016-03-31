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
arma::vec DensTransMWNOU(arma::mat x, arma::mat x0, double t, arma::vec mu, arma::vec alpha, arma::vec sigma, int maxK, double etrunc);

// Main
int main(int argc, char** argv) {
  
  /*
   *  Argument codification in argv: 
   *  x01 x02 t mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x
   *  (where x is the N x 2 matrix of evaluation points stored by rows as a vector)
   */
  
  // Set parameters
  double t = atof(argv[3]);
  arma::vec mu(2);
  mu(0) = atof(argv[4]); mu(1) = atof(argv[5]);
  arma::vec alpha(3); 
  alpha(0) = atof(argv[6]); alpha(1) = atof(argv[7]); alpha(2) = atof(argv[8]);
  arma::vec sigma(2);
  sigma(0) = atof(argv[9]); sigma(1) = atof(argv[10]);
  int maxK = atoi(argv[11]);
  double etrunc = atof(argv[12]);
  
  // Read matrix
  int count = 13;
  int n_rows = (argc - count)/2;
  arma::mat x(n_rows, 2);
  for (int i = 0; i < n_rows; i++){
    for (int j = 0; j < 2; j++){
      x(i, j) = atof(argv[count]);
      count++;
    }
  }

  // Create x0 as matrix (and not as a vector!)
  arma::mat x0(n_rows, 2);
  x0.col(0).fill(atof(argv[1])); x0.col(1).fill(atof(argv[2]));
  
  // Call function
  arma::vec result = DensTransMWNOU(x, x0, t, mu, alpha, sigma, maxK, etrunc);

  // Print result
  result.t().print();

  return 0;
    
}

// Subroutine
arma::vec DensTransMWNOU(arma::mat x, arma::mat x0, double t, arma::vec mu, arma::vec alpha, arma::vec sigma, int maxK = 2, double etrunc = 50) {
  
  /*
  * Description: Computation of the tpd for a MWN-OU diffusion (with diagonal diffusion matrix)
  * 
  * Arguments:
  *
  * - x: a n x 2 matrix of pairs of dihedrals. They must be in [-PI, PI) so that the truncated wrapping by 
  *      maxK windings is able to capture periodicity.
  * - x0: a n x 2 matrix of pairs of dihedrals, the starting values. They must be in [-PI, PI) so that the truncated wrapping by 
  *       maxK windings is able to capture periodicity.
  * - t: the common time between the observed pairs.
  * - mu: a vector of length 2 with the mean parameter of the MWN-OU process. The mean of the MWN stationary distribution.
  * - alpha: vector of length 3 containing the A matrix of the drift of the MWN-OU process in the following codification: 
  *          A = [alpha[0], alpha[2] * sqrt(sigma[0] / sigma[1]); alpha[2] * sqrt(sigma[1] / sigma[0]), alpha[1]]. 
  *          This enforces that A^(-1) * Sigma is symmetric. Positive definiteness is guaranteed if alpha[0] * alpha[1] > alpha[2] * alpha[2].
  *          The function checks for it and, if violated, returns samples from a close density with A^(-1) * Sigma positive definite.
  * - sigma: vector of length 2 containing the diagonal of Sigma, the diffusion matrix. Note that these are the *squares*
  *          (i.e. variances) of the diffusion coefficients that multiply the Wiener process.
  * - maxK: maximum number of winding number considered in the computation of the approximated transition probability density.
  * - etrunc: truncation for exponential. exp(x) with x <= -etrunc is set to zero. 
  * 
  * Warning: 
  * 
  *  - A combination of small etrunc (< 30) and low maxK (<= 1) can lead to NaNs produced by 0 / 0 in the weight computation. 
  *    This is specially dangerous if sigma is large and there are values in x or x0 outside [-PI, PI).
  * 
  * Value: 
  * 
  * - tpdfinal: tpd of the transitions from initial to final dihedrals.
  * 
  * Author: Eduardo García-Portugués (egarcia@math.ku.dk) 
  * 
  */
  
  /*
  * Create basic objects
  */
  
  // Number of pairs
  int N = x.n_rows;
  
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
  double lognormconstGammat = -log(2 * M_PI) + log(det(invGammat)) / 2;
  
  /* 
  * Weights of the winding numbers for each data point
  */
  
  // We store the weights in a matrix to skip the null later in the computation of the tpd
  arma::mat weightswindsinitial(N, lk * lk);
  
  // Loop in the data
  for(int i = 0; i < N; i++){
    
    // Compute the factors in the exponent that do not depend on the windings
    arma::vec xmu = x0.row(i).t() - mu;
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
        
        // Truncate the negative exponential
        if(exponent > etrunc){
          
          weightswindsinitial(i, wek1 * lk + wek2) = 0;
          
        }else{
          
          weightswindsinitial(i, wekl1 + wek2) = exp(-exponent);
          
        }
        
      }
      
    }
    
  }
  
  // Standardize weights for the tpd
  weightswindsinitial.each_col() /= sum(weightswindsinitial, 1);
  
  /*
  * Computation of the tpd: wrapping + weighting
  */
  
  // The evaluations of the tpd are stored in a vector, no need to keep track of wrappings
  arma::vec tpdfinal(N); tpdfinal.zeros();
  
  // Loop in the data
  for(int i = 0; i < N; i++){
    
    // Initial point x0 varying with i
    arma::vec x00 = x0.row(i).t();
    
    // Loop on the winding weight K1
    for(int wek1 = 0; wek1 < lk; wek1++){
      
      // 2 * K1 * PI
      twokepivec(0) = twokpi(wek1); 
      
      // Compute once the index
      int wekl1 = wek1 * lk;
      
      // Loop on the winding weight K2  
      for(int wek2 = 0; wek2 < lk; wek2++){
        
        // Skip zero weights
        if(weightswindsinitial(i, wekl1 + wek2) > 0){
          
          // 2 * K1 * PI
          twokepivec(1) = twokpi(wek2); 
          
          // mut
          arma::vec mut = mu + ExptA * (x00 + twokepivec - mu);
          
          // Compute the factors in the exponent that do not depend on the windings
          arma::vec xmut = x.row(i).t() - mut;
          arma::vec xmutinvGammat = invGammat * xmut;
          double xmutinvGammatxmutdiv2 = dot(xmutinvGammat, xmut) / 2;
          
          // Loop in the winding wrapping K1
          for(int wak1 = 0; wak1 < lk; wak1++){
            
            // 2 * K1 * PI
            twokapivec(0) = twokpi(wak1); 
            
            // Loop in the winding wrapping K2
            for(int wak2 = 0; wak2 < lk; wak2++){
              
              // 2 * K2 * PI
              twokapivec(1) = twokpi(wak2);
              
              // Decomposition of the exponent
              double exponent = xmutinvGammatxmutdiv2 + dot(xmutinvGammat, twokapivec) + dot(invGammat * twokapivec, twokapivec) / 2 - lognormconstGammat;
              
              // Truncate the negative exponential
              if(exponent < etrunc){
                
                tpdfinal(i) += exp(-exponent) * weightswindsinitial(i, wekl1 + wek2);
                
              }
              
            }
            
          }
          
        }
        
      }
      
    }
    
  }
  
  return tpdfinal;
  
}
