#include <armadillo>

using namespace arma;

mat SampStatWNOU(int N, vec mu, vec alpha, vec sigma) {
  
    /*
     * Description: Sampling from the stationary distribution of a MWN-OU diffusion (with diagonal diffusion matrix)
     * 
     * Arguments:
     *
     * - N: sample size.
     * - mu: a vector of length 2 with the mean parameter of the MWN-OU process. The mean of the MWN stationary distribution.
     * - alpha: vector of length 3 containing the A matrix of the drift of the MWN-OU process in the following codification: 
     *        A = [alpha[0], alpha[2] * sqrt(sigma[0] / sigma[1]); alpha[2] * sqrt(sigma[1] / sigma[0]), alpha[1]]. 
     *        This enforces that A^(-1) * Sigma is symmetric. Positive definiteness is guaranteed if alpha[0] * alpha[1] > alpha[2] * alpha[2].
     *        The function checks for it and, if violated, returns the sample from a close A^(-1) * Sigma that is positive definite.
     * - sigma: vector of length 2 containing the diagonal of Sigma, the diffusion matrix.
     * 
     * Value: 
     * 
     * - x: matrix of size N x 2 containing the samples.
     * 
     * Author: Eduardo García-Portugués (egarcia@math.ku.dk) 
     * 
     */
    
    /*
     * Compute A^(-1) * Sigma and check for positive definiteness
     */
    
    // Create and initialize A
    double quo = sqrt(sigma(0) / sigma(1));
    mat A(2, 2); 
    A(0, 0) = alpha(0); 
    A(1, 1) = alpha(1); 
    A(0, 1) = alpha(2) * quo;
    A(1, 0) = alpha(2) / quo;
    
    // Only positive definiteness can be violated with the parametrization of A
    double testalpha = alpha(0) * alpha(1) - alpha(2) * alpha(2);
    
    // Check positive definiteness 
    if(testalpha <= 0) {
      
      // Update alpha(2) such that testalpha > 0
      alpha(2) = signbit(alpha(2)) * sqrt(alpha(0) * alpha(1)) * 0.9999;
      
      // Reset A to a matrix with positive determinant and continue
      A(0, 1) = alpha(2) * quo;
      A(1, 0) = alpha(2) / quo;
      
    }
    
    // Create invASigma
    mat invASigma = inv_sympd(A) * diagmat(sigma) / 2;
    
    /*
     * Sample correlated normal variables
     */
    
    // Sample of independent N(0, 1)
    mat x = randn(N, 2);
    
    // Cholesky decomposition for correlate random sample
    mat ch = chol(invASigma);
    
    // Correlate random variables
    x = x * ch;
    
    // Recenter
    x.each_row() += mu.t();
    
    // Wrap (convert to [-PI,PI) x [-PI,PI))
    x -= floor((x + M_PI) / (2 * M_PI)) * (2 * M_PI);

    return x;
    
}
