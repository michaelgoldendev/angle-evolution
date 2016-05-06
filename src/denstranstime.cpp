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
arma::vec dTpdWnOu2D(arma::mat x, arma::mat x0, arma::vec t, arma::vec alpha, arma::vec mu, arma::vec sigma, int maxK, double etrunc);

// Main
int main(int argc, char** argv) {
  
  /*
   *  Argument codification in argv: 
   *  mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc x0 x t
   *  (where x0 and x are the N x 2 matrices of starting and evaluation
   *   points stored by rows as a vector, respectively, and t is the 
   *   vector of times)
   */
  
  // Set parameters
  arma::vec mu(2);
  mu(0) = atof(argv[1]); mu(1) = atof(argv[2]);
  arma::vec alpha(3); 
  alpha(0) = atof(argv[3]); alpha(1) = atof(argv[4]); alpha(2) = atof(argv[5]);
  arma::vec sigma(2);
  sigma(0) = atof(argv[6]); sigma(1) = atof(argv[7]);
  int maxK = atoi(argv[8]);
  double etrunc = atof(argv[9]);
  
  // Read matrix x0
  int count = 10;
  int n_rows = (argc - count)/5;
  arma::mat x0(n_rows, 2);
  for (int i = 0; i < n_rows; i++){
    for (int j = 0; j < 2; j++){
      x0(i, j) = atof(argv[count]);
      count++;
    }
  }
  
  // Read matrix x
  arma::mat x(n_rows, 2);
  for (int i = 0; i < n_rows; i++){
    for (int j = 0; j < 2; j++){
      x(i, j) = atof(argv[count]);
      count++;
    }
  }
  
  // Read vector t
  arma::vec t(n_rows);
  for (int i = 0; i < n_rows; i++){
    t(i) = atof(argv[count]);
    count++;
  }
  
  // Call function
  arma::vec result = dTpdWnOu2D(x, x0, t, alpha, mu, sigma, maxK, etrunc);

  // Print result
  result.t().print();

  return 0;
    
}

//' @title Approximation of the tpd of the MWN-OU process in 2D
//'
//' @description Computation of the transition probability density (tpd) for a MWN-OU diffusion (with diagonal diffusion matrix)
//'
//' @param x a matrix of dimension \code{c(n, 2)} containing angles. They all must be in \eqn{[\pi,\pi)} so that the truncated wrapping by \code{maxK} windings is able to capture periodicity.
//' @param x0 a matrix of of dimension \code{c(n, 2)} containing the starting angles. They all must be in \eqn{[\pi,\pi)}.
//' @param t either a scalar or a vector of length \code{n} containing the times separating \code{x} and \code{x0}. If \code{t} is a scalar, a common time is assumed.
//' @param alpha vector of length \code{3} parametrizing the \code{A} matrix of the drift of the MWN-OU process in the following codification:
//'
//' \code{A = rbind(c(alpha[1], alpha[3] * sqrt(sigma[1] / sigma[2])),
//'                 c(alpha[3] * sqrt(sigma[2] / sigma[1]), alpha[2]))}.
//'
//' This enforces that \code{solve(A) \%*\% diag(sigma)} is symmetric. Positive definiteness is guaranteed if \code{alpha[1] * alpha[2] > alpha[3]^2}.
//' The function checks for it and, if violated, resets \code{A} such that \code{solve(A) \%*\% diag(sigma)} is positive definite.
//' @param mu a vector of length \code{2} with the mean parameter of the MWN-OU process.
//' @param sigma vector of length \code{2} containing the \strong{square root} of the diagonal of \eqn{\Sigma}, the diffusion matrix.
//' @param maxK maximum number of winding number considered in the computation of the approximated transition probability density.
//' @inheritParams safeSoftMax
//' @return A vector of size \code{n} containing the tpd evaluated at \code{x}.
//' @author Eduardo Garcia-Portugues (\email{egarcia@@math.ku.dk})
//' @examples
//' set.seed(345567)
//' alpha <- c(2, 1, -1)
//' sigma <- c(1.5, 2)
//' Sigma <- diag(sigma^2)
//' A <- alphaToA(alpha = alpha, sigma = sigma)
//' mu <- c(pi, pi)
//' x <- t(eulerWn2D(x0 = matrix(c(0, 0), nrow = 1), A = A, mu = mu,
//'                  sigma = sigma, N = 500, delta = 0.1)[1, , ])
//' sum(sapply(1:49, function(i) log(dTpdMwou(x = matrix(x[i + 1, ], ncol = 2),
//'                                           x0 = x[i, ], t = 1.5, A = A,
//'                                           Sigma = Sigma, mu = mu, K = 2,
//'                                           N.int = 2000))))
//' sum(log(dTpdWnOu2D(x = matrix(x[2:50, ], ncol = 2),
//'                    x0 = matrix(x[1:49, ], ncol = 2), t = 1.5, alpha = alpha,
//'                    mu = mu, sigma = sigma)))
//' \dontrun{
//' lgrid <- 56
//' grid <- seq(-pi, pi, l = lgrid + 1)[-(lgrid + 1)]
//' image(grid, grid, matrix(dTpdMwou(x = as.matrix(expand.grid(grid, grid)),
//'                                   x0 = c(0, 0), t = 0.5, A = A, Sigma = Sigma,
//'                                   mu = mu, K = 2, N.int = 2000),
//'                          nrow = 56, ncol = 56), zlim = c(0, 0.25),
//'       main = "dTpdMwou")
//' image(grid, grid, matrix(dTpdWnOu2D(x = as.matrix(expand.grid(grid, grid)),
//'                                     x0 = matrix(0, nrow = 56^2, ncol = 2),
//'                                     t = 0.5, alpha = alpha, sigma = sigma,
//'                                     mu = mu),
//'                          nrow = 56, ncol = 56), zlim = c(0, 0.25),
//'       main = "dTpdWnOu2D")
//'
//' dr <- driftWn2D(x = as.matrix(expand.grid(grid, grid)), A = A, mu = mu,
//'                 sigma = sigma, maxK = 2, etrunc = 30)
//' b1 <- matrix(dr[, 1], nrow = lgrid, ncol = lgrid)
//' b2 <- matrix(dr[, 2], nrow = lgrid, ncol = lgrid)
//' parms <- list(b1 = b1, b2 = b2, sigma2.1 = Sigma[1, 1], sigma2.2 = Sigma[2, 2],
//'               len.grid = lgrid, delx = grid[2] - grid[1])
//' image(grid, grid, matrix(tpd.2D(x0i = which.min(2 - 2 * cos(grid - 0)),
//'                                  y0i = which.min(2 - 2 * cos(grid - 0)),
//'                                  times = seq(0, .5, l = 100), parms = parms,
//'                                  method = "lsodes", atol = 1e-10,
//'                                  lrw = 7e+07)[100, ], nrow = lgrid,
//'                                  ncol = lgrid),
//'       zlim = c(0, 0.25), main = "tpd.2D")
//'
//' x <- seq(-pi, pi, l = 100)
//' t <- 1
//' image(x, x, matrix(dTpdWnOu2D(x = as.matrix(expand.grid(x, x)),
//'                               x0 = matrix(rep(0, 100 * 2), nrow = 100 * 100,
//'                                           ncol = 2),
//'                               t = t, alpha = alpha, mu = mu, sigma = sigma,
//'                               maxK = 2, etrunc = 30), 100, 100),
//'       zlim = c(0, 0.25))
//' points(stepAheadWn2D(x0 = c(0, 0), delta = t / 500,
//'                      A = alphaToA(alpha = alpha, sigma = sigma), mu = mu,
//'                      sigma = sigma, N = 500, M = 1000, maxK = 2,
//'                      etrunc = 30))
//' }
//' @export
// [[Rcpp::export]]
arma::vec dTpdWnOu2D(arma::mat x, arma::mat x0, arma::vec t, arma::vec alpha, arma::vec mu, arma::vec sigma, int maxK = 2, double etrunc = 30) {
  
  /*
  * Create basic objects
  */
  
  // Number of pairs
  int N = x.n_rows;
  
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
    
  }else if(t.n_elem == N) {
    
    commonTime = 1;
    
  } else {
    
    //stop("Length of t is neither 1 nor N");
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
  arma::mat weightswindsinitial(N, lk * lk);
  weightswindsinitial.fill(lognormconstSigmaA);
  
  // Loop in the data
  for(int i = 0; i < N; i++){
    
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
  * Computation of the tpd: wrapping + weighting
  */
  
  // The evaluations of the tpd are stored in a vector, no need to keep track of wrappings
  arma::vec tpdfinal(N); tpdfinal.zeros();
  
  // Variables inside the commonTime if-block
  arma::mat ExptiA(2, 2), invGammati(2, 2);  
  double logDetInvGammati, lognormconstGammati;
  
  // If t is common, compute once
  if(commonTime == 0){
    
    // Exp(-ti * A)
    ExptiA = s2t(0) * A;
    ExptiA.diag() += s1t(0);
    
    // Inverse and log-normalizing constant for the Gammat
    invGammati = 2 * inv_sympd((1 - s12t(0)) * AInvSigma - s22t(0) * Sigma);
    
    // Log-determinant of invGammati (assumed to be positive)
    arma::log_det(logDetInvGammati, sign, invGammati);
    
    // Log-normalizing constant for the Gaussian with covariance Gammati
    lognormconstGammati = -l2pi + logDetInvGammati / 2;
    
  }
  
  // Loop in the data
  for(int i = 0; i < N; i++){
    
    // Initial point x0 varying with i
    arma::vec x00 = x0.row(i).t();
    
    // Evaluation point x varying with i
    arma::vec xx = x.row(i).t();
    
    // If t is not common
    if(commonTime != 0){
      
      // Exp(-ti * A)
      ExptiA = s2t(i) * A;
      ExptiA.diag() += s1t(i);
      
      // Inverse and log-normalizing constant for the Gammati
      invGammati = 2 * inv_sympd((1 - s12t(i)) * AInvSigma - s22t(i) * Sigma);
      
      // Log-determinant of invGammati (assumed to be positive)
      arma::log_det(logDetInvGammati, sign, invGammati);
      
      // Log-normalizing constant for the Gaussian with covariance Gammati
      lognormconstGammati = -l2pi + logDetInvGammati / 2;
      
    }
    
    // Common muti
    arma::vec muti = mu + ExptiA * (x00 - mu);
    
    // Loop on the winding weight K1
    for(int wek1 = 0; wek1 < lk; wek1++){
      
      // 2 * K1 * M_PI
      twokepivec(0) = twokpi(wek1);
      
      // Compute once the index
      int wekl1 = wek1 * lk;
      
      // Loop on the winding weight K2
      for(int wek2 = 0; wek2 < lk; wek2++){
        
        // Skip zero weights
        if(weightswindsinitial(i, wekl1 + wek2) > 0){
          
          // 2 * K1 * M_PI
          twokepivec(1) = twokpi(wek2);
          
          // Compute the factors in the exponent that do not depend on the windings
          arma::vec xmuti = xx - muti - ExptiA * twokepivec;
          arma::vec xmutiInvGammati = invGammati * xmuti;
          double xmutiInvGammatixmutidiv2 = -dot(xmutiInvGammati, xmuti) / 2;
          
          // Loop in the winding wrapping K1
          for(int wak1 = 0; wak1 < lk; wak1++){
            
            // 2 * K1 * M_PI
            twokapivec(0) = twokpi(wak1);
            
            // Loop in the winding wrapping K2
            for(int wak2 = 0; wak2 < lk; wak2++){
              
              // 2 * K2 * M_PI
              twokapivec(1) = twokpi(wak2);
              
              // Decomposition of the negative exponent
              double exponent = xmutiInvGammatixmutidiv2 - dot(xmutiInvGammati, twokapivec) - dot(invGammati * twokapivec, twokapivec) / 2 + lognormconstGammati;
              
              // Tpd
              tpdfinal(i) += exp(exponent) * weightswindsinitial(i, wekl1 + wek2);
              
            }
            
          }
          
        }
        
      }
      
    }
    
  }
  
  return tpdfinal;
  
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
