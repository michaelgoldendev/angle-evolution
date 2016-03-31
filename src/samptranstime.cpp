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

// Declarations
arma::mat safeSoftMax(arma::mat logs, double etrunc);
arma::cube rTpdWnOu2D(int n, arma::mat x0, arma::vec t, arma::vec mu, arma::vec alpha, arma::vec sigma, int maxK, double etrunc);

// Main
int main(int argc, char** argv) {
  
  /*
  *  Argument codification in argv: 
  *  N mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x0 t
  *  (where x0 is the nx0 x 2 matrix stored by rows as a vector
  *  and t is a vector of length nx0)
  */
  
  // Set parameters
  double N = atoi(argv[1]);
  arma::vec mu(2); 
  mu(0) = atof(argv[2]); mu(1) = atof(argv[3]);
  arma::vec alpha(3); 
  alpha(0) = atof(argv[4]); alpha(1) = atof(argv[5]); alpha(2) = atof(argv[6]);
  arma::vec sigma(2);
  sigma(0) = atof(argv[7]); sigma(1) = atof(argv[8]);
  int maxK = atoi(argv[9]);
  double etrunc = atof(argv[10]);
  
  // Read matrix
  int count = 11;
  int n_rows = (argc - count)/3;
  arma::mat x0(n_rows, 2);
  for (int i = 0; i < n_rows; i++){
    for (int j = 0; j < 2; j++){
      x0(i, j) = atof(argv[count]);
      count++;
    }
  }
  
  // Read vector
  arma::vec t(n_rows);
  for (int i = 0; i < n_rows; i++){
    t(i) = atof(argv[count]);
    count++;
  }
  
  // Call function
  arma::cube result = rTpdWnOu2D(N, x0, t, mu, alpha, sigma, maxK, etrunc);
  
  // Print result
  result.print();
  
  return 0;
  
}

//' @title Simulation from the approximated transition distribution of a MWN-OU diffusion in 2D
//'
//' @description Simulates from the approximate transition density of the MWN-OU diffusion in 2D.
//'
//' @param n sample size.
//' @param x0 a matrix of dimension \code{c(nx0, 2)} giving the starting values.
//' @param t vector of length \code{nx0} containing the times between observations.
//' @inheritParams dTpdWnOu2D
//' @inheritParams safeSoftMax
//' @return An array of dimension \code{c(n, 2, nx0)} containing the \code{n} samples of the trasition distribution,
//' conditioned on that the process was at \code{x0} at \code{t} instants ago. The samples are all in \eqn{[\pi,\pi)}.
//' @author Eduardo Garcia-Portugues (\email{egarcia@@math.ku.dk})
//' @examples
//' alpha <- c(3, 2, -1)
//' sigma <- c(0.5, 1)
//' mu <- c(pi, pi)
//' x <- seq(-pi, pi, l = 100)
//' t <- 0.5
//' image(x, x, matrix(dTpdWnOu2D(x = as.matrix(expand.grid(x, x)),
//'                               x0 = matrix(rep(0, 100 * 2),
//'                                           nrow = 100 * 100, ncol = 2),
//'                               t = t, mu = mu, alpha = alpha, sigma = sigma,
//'                               maxK = 2, etrunc = 30), nrow = 100, ncol = 100),
//'       zlim = c(0, 0.5))
//' points(rTpdWnOu2D(n = 500, x0 = rbind(c(0, 0)), t = t, mu = mu, alpha = alpha,
//'                   sigma = sigma)[, , 1], col = 3)
//' points(stepAheadWn2D(x0 = c(0, 0), delta = t / 500,
//'                      A = alphaToA(alpha = alpha, sigma = sigma),
//'                      mu = mu, sigma = sigma, N = 500, M = 500, maxK = 2,
//'                      etrunc = 30), col = 4)
//' @export
// [[Rcpp::export]]
arma::cube rTpdWnOu2D(int n, arma::mat x0, arma::vec t, arma::vec mu, arma::vec alpha, arma::vec sigma, int maxK = 2, double etrunc = 30) {
  
  /*
  * Create basic objects
  */
  
  // Number of different starting angles
  int nx0 = x0.n_rows;
  
  // Create and initialize A
  double quo = sigma(0) / sigma(1);
  arma::mat A(2, 2);
  A(0, 0) = alpha(0);
  A(1, 1) = alpha(1);
  A(0, 1) = alpha(2) * quo;
  A(1, 0) = alpha(2) / quo;
  
  // Create and initialize Sigma
  arma::mat Sigma = diagmat(square(sigma));
  
  // Sequence of winding numbers
  const int lk = 2 * maxK + 1;
  arma::vec twokpi = arma::linspace<arma::vec>(-maxK * 2 * M_PI, maxK * 2 * M_PI, lk);
  
  // Bivariate vector (2 * K1 * M_PI, 2 * K2 * M_PI) for weighting
  arma::vec twokepivec(2);
  
  // Bivariate vector (2 * K1 * M_PI, 2 * K2 * M_PI) for wrapping
  arma::vec twokapivec(2);
  
  /*
  * Check if the t is common
  */
  
  int commonTime;
  if(t.n_elem == 1){
    
    commonTime = 0;
    
  }else if(t.n_elem == nx0) {
    
    commonTime = 1;
    
  } else {
    
    //stop("Length of t is neither 1 nor nx0");
    std::exit(EXIT_FAILURE);
    
  }
  
  /*
  * Check for symmetry and positive definiteness of A^(-1) * Sigma
  */
  
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
  
  // A^(-1) * Sigma
  arma::mat AInvSigma(2, 2);
  AInvSigma(0, 0) = alpha(1) * Sigma(0, 0);
  AInvSigma(0, 1) = -alpha(2) * sigma(0) * sigma(1);
  AInvSigma(1, 0) = AInvSigma(0, 1);
  AInvSigma(1, 1) = alpha(0) * Sigma(1, 1);
  AInvSigma = AInvSigma / (alpha(0) * alpha(1) - alpha(2) * alpha(2));
  
  // Inverse of (1/2 * A^(-1) * Sigma): 2 * Sigma^(-1) * A
  arma::mat invSigmaA(2, 2);
  invSigmaA(0, 0) = 2 * alpha(0) / Sigma(0, 0);
  invSigmaA(0, 1) = 2 * alpha(2) / (sigma(0) * sigma(1));
  invSigmaA(1, 0) = invSigmaA(0, 1);
  invSigmaA(1, 1) = 2 * alpha(1) / Sigma(1, 1);
  
  // For normalizing constants
  double l2pi = log(2 * M_PI);
  
  // Log-determinant of invSigmaA (assumed to be positive)
  double logDetInvSigmaA, sign;
  arma::log_det(logDetInvSigmaA, sign, invSigmaA);
  
  // Log-normalizing constant for the Gaussian with covariance SigmaA
  double lognormconstSigmaA = -l2pi + logDetInvSigmaA / 2;
  
  /*
  * Computation of Gammat and exp(-t * A) analytically
  */
  
  // Quantities for computing exp(-t * A)
  double s = sum(alpha.head(2)) / 2;
  double q = sqrt(fabs((alpha(0) - s) * (alpha(1) - s) - alpha(2) * alpha(2)));
  
  // Avoid indetermination in sinh(q * t) / q when q == 0
  if(q == 0){
    
    q = 1e-6;
    
  }
  
  // s1(-t) and s2(-t)
  arma::vec est = exp(-s * t);
  arma::vec eqt = exp(q * t);
  arma::vec eqtinv = 1 / eqt;
  arma::vec cqt = (eqt + eqtinv) / 2;
  arma::vec sqt = (eqt - eqtinv) / (2 * q);
  arma::vec s1t = est % (cqt + s * sqt);
  arma::vec s2t = -est % sqt;
  
  // s1(-2t) and s2(-2t)
  est = est % est;
  eqt = eqt % eqt;
  eqtinv = eqtinv % eqtinv;
  cqt = (eqt + eqtinv) / 2;
  sqt = (eqt - eqtinv) / (2 * q);
  arma::vec s12t = est % (cqt + s * sqt);
  arma::vec s22t = -est % sqt;
  
  /*
  * Weights of the winding numbers for each data point
  */
  
  // We store the weights in a matrix to skip the null later in the computation of the tpd
  arma::mat weightswindsinitial(nx0, lk * lk);
  weightswindsinitial.fill(lognormconstSigmaA);
  
  // Loop in the data
  for(int i = 0; i < nx0; i++){
    
    // Compute the factors in the exponent that do not depend on the windings
    arma::vec xmu = x0.row(i).t() - mu;
    arma::vec xmuinvSigmaA = invSigmaA * xmu;
    double xmuinvSigmaAxmudivtwo = -dot(xmuinvSigmaA, xmu) / 2;
    
    // Loop in the winding weight K1
    for(int wek1 = 0; wek1 < lk; wek1++){
      
      // 2 * K1 * M_PI
      twokepivec(0) = twokpi(wek1);
      
      // Compute once the index
      int wekl1 = wek1 * lk;
      
      // Loop in the winding weight K2
      for(int wek2 = 0; wek2 < lk; wek2++){
        
        // 2 * K2 * M_PI
        twokepivec(1) = twokpi(wek2);
        
        // Negative exponent
        weightswindsinitial(i, wekl1 + wek2) += xmuinvSigmaAxmudivtwo - dot(xmuinvSigmaA, twokepivec) - dot(invSigmaA * twokepivec, twokepivec) / 2;
        
      }
      
    }
    
  }
  
  // Standardize weights for the tpd
  weightswindsinitial = safeSoftMax(weightswindsinitial, etrunc);
  
  /*
  * Sampling
  */
  
  // Probabilities of windings
  arma::mat probs = arma::cumsum(weightswindsinitial, 1);
  
  // Sample uniforms in [0, 1]. There is no weigthed sampling in Armadillo!
  arma::vec runif = arma::randu(n * nx0);
  
  // Matrix of muti for each x
  arma::mat mutix(n, 2);
  
  // Sample of independent N(0, 1)
  arma::cube x = arma::randn(n, 2, nx0);
  
  // Variables inside the commonTime if-block
  arma::mat ExptiA(2, 2), Gammati(2, 2), ch(2, 2);  
  
  // If t is common, compute once
  if(commonTime == 0){
    
    // Exp(-ti * A)
    ExptiA = s2t(0) * A;
    ExptiA.diag() += s1t(0);
    
    // Gammati
    Gammati = ((1 - s12t(0)) * AInvSigma - s22t(0) * Sigma) / 2;
    
    // Cholesky decomposition for correlate random sample
    ch = chol(Gammati);
    
  }
  
  // Loop throught the x0s
  for(int i = 0; i < nx0; i++){
    
    // If t is not common
    if(commonTime != 0){
      
      // Exp(-ti * A)
      ExptiA = s2t(i) * A;
      ExptiA.diag() += s1t(i);
      
      // Gammati
      Gammati = ((1 - s12t(i)) * AInvSigma - s22t(i) * Sigma) / 2;
      
      // Cholesky decomposition for correlate random sample
      ch = chol(Gammati);
      
    }
    
    // Common muti
    mutix.each_row() = (mu + ExptiA * (x0.row(i).t() - mu)).t();
    
    // Compute once the index
    int li = i * n;
    
    // Loop in the number of replicates
    for(int m = 0; m < n; m++){
      
      // Choose windings with probabilities weightswindsinitial
      // Loop in the winding weight K1
      for(int wek1 = 0; wek1 < lk; wek1++){
        
        // Compute once the index
        int wekl1 = wek1 * lk;
        
        // Loop in the winding weight K2
        for(int wek2 = 0; wek2 < lk; wek2++){
          
          // Weighted sampling
          if(runif(li + m) <= probs(i, wekl1 + wek2)) {
            
            // 2 * K1 * M_PI
            twokepivec(0) = twokpi(wek1);
            
            // 2 * K2 * M_PI
            twokepivec(1) = twokpi(wek2);
            
            // Set mut for x
            mutix.row(m) += (ExptiA * twokepivec).t();
            
            // Skip the wek1 and wek2 loops
            goto sample;
            
          }
          
        }
        
      }
      
      // Goto identifier
      sample: ;
      
    }
    
    // Correlate random variables
    x.slice(i) *= ch;
    
    // Recenter depending on mut
    x.slice(i) += mutix;
    
  }
  
  // Wrap (convert to [-M_PI,M_PI) x [-M_PI,M_PI))
  x -= floor((x + M_PI) / (2 * M_PI)) * (2 * M_PI);
  
  return x;
  
}

//' @title Safe softmax function for computing weights
//'
//' @description Computes the weights \eqn{w_i = \frac{e^{p_i}}{\sum_{j=1}^k e^{p_j}}} from \eqn{p_i}, \eqn{i=1,\ldots,k}
//' in a safe way to avoid overflows and to truncate automatically to zero low values of \eqn{w_i}.
//' 
//' @param logs matrix of logarithms where each row contains a set of \eqn{p_1,\ldots,p_k} to compute the weights from.
//' @param etrunc truncation for exponential: \code{exp(x)} with \code{x <= -etrunc} is set to zero. Defaults to \code{30}.
//' @return A matrix of the size as \code{logs} containing the weights for each row.
//' @author Eduardo Garcia-Portugues (\email{egarcia@@math.ku.dk})
//' @details The \code{logs} argument must be always a matrix.
//' @examples
//' # A matrix
//' safeSoftMax(rbind(1:10, 20:11))
//' rbind(exp(1:10) / sum(exp(1:10)), exp(20:11) / sum(exp(20:11)))
//' 
//' # A row-matrix
//' safeSoftMax(rbind(-100:100), etrunc = 30)
//' exp(-100:100) / sum(exp(-100:100))
//' @export
// [[Rcpp::export]]
arma::mat safeSoftMax(arma::mat logs, double etrunc = 30) {
  
  // Maximum of logs by rows to avoid overflows
  arma::vec m = max(logs, 1);
  
  // Recenter by columns
  logs.each_col() -= m;
  
  // Ratios by columns
  logs.each_col() -= log(sum(exp(logs), 1));
  
  // Truncate exponential by using a lambda function - requires C++ 11
  logs.transform([etrunc](double val) { return (val < -etrunc) ? double(0) : double(exp(val)); });
  
  return logs;
  
}
