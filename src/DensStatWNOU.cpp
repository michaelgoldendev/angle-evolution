#include <armadillo>

using namespace arma;

vec DensStatWNOU(mat x, vec mu, vec alpha, vec sigma, int maxK = 1, double etrunc = 25) {
  
  /*
   * Description: Density of the stationary distribution of a MWN-OU diffusion (with diagonal diffusion matrix)
   * 
   * Arguments:
   *
   * - x: matrix of size N x 2 containing the evaluation points.
   * - mu: a vector of length 2 with the mean parameter of the MWN-OU process. The mean of the MWN stationary distribution.
   * - alpha: vector of length 3 containing the A matrix of the drift of the MWN-OU process in the following codification: 
   *        A = [alpha[0], alpha[2] * sqrt(sigma[0] / sigma[1]); alpha[2] * sqrt(sigma[1] / sigma[0]), alpha[1]]. 
   *        This enforces that A^(-1) * Sigma is symmetric. Positive definiteness is guaranteed if alpha[0] * alpha[1] > alpha[2] * alpha[2].
   *        The function checks for it and, if violated, returns the density from a close A^(-1) * Sigma that is positive definite.
   * - sigma: vector of length 2 containing the diagonal of Sigma, the diffusion matrix.
   * - maxK: maximum number of winding number considered in the computation of the approximated transition probability density.
   * - etrunc: truncation for exponential. exp(x) with x <= -etrunc is set to zero.
   * 
   * Value: 
   * 
   * - dens: vector of size N containing the density evaluated at x.
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
  mat A(2, 2); 
  A(0, 0) = alpha(0); 
  A(1, 1) = alpha(1); 
  A(0, 1) = alpha(2) * quo;
  A(1, 0) = alpha(2) / quo;
  
  // Create and initialize Sigma
  mat Sigma = diagmat(sigma);
  
  // Sequence of winding numbers
  const int lk = 2 * maxK + 1;
  vec twokpi = linspace<vec>(-maxK * 2 * M_PI, maxK * 2 * M_PI, lk);
  
  // Bivariate vector (2 * K1 * PI, 2 * K2 * PI) for weighting
  vec twokepivec(2);
  
  /*
  * Check for symmetry and positive definiteness of A^(-1) * Sigma
  */

  // Only positive definiteness can be violated with the parametrization of A
  double testalpha = alpha(0) * alpha(1) - alpha(2) * alpha(2);
  
  // Check positive definiteness 
  if(testalpha <= 0) {
    
    // Update alpha(2) such that testalpha > 0
    alpha(2) = signbit(alpha(2)) * sqrt(alpha(0) * alpha(1)) * 0.9999;
    
    // Reset A to a matrix with positive determinant
    A(0, 1) = alpha(2) * quo;
    A(1, 0) = alpha(2) / quo;
    
  }
  
  // Inverse of 1/2 * A^(-1) * Sigma: 2 * Sigma^(-1) * A
  mat invSigmaA = 2 * diagmat(1 / diagvec(Sigma)) * A;
  
  // Log-normalizing constant for the Gaussian with covariance SigmaA
  double lognormconstSigmaA = -log(2 * M_PI) + log(det(invSigmaA)) / 2;
  
  /* 
   * Evaluation of the density reusing the code from the weights of the winding numbers
   * in LogLikWNOUPairs for each data point. Here we sum all the unstandarized weights 
   * for each data point.
   */
  
  // We store the weights in a matrix to skip the null later in the computation of the tpd
  mat weightswindsinitial(N, lk * lk);
  
  // Loop in the data
  for(int i = 0; i < N; i++){
    
    // Compute the factors in the exponent that do not depend on the windings
    vec xmu = x.row(i).t() - mu;
    vec xmuinvSigmaA = invSigmaA * xmu;
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
  
  // The density is the sum of the weights
  vec dens = sum(weightswindsinitial, 1);
  
  return dens;
  
}

