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
double logLikWnOuPairs(arma::mat x, arma::vec t, arma::vec alpha, arma::vec mu, arma::vec sigma, int maxK, double etrunc);

// Main
int main(int argc, char** argv) {
  
  /*
   *  Argument codification in argv: 
   *  mu1 mu2 alpha1 alpha2 alpha3 sigma1 sigma2 maxK etrunc data t
   *  (where data is the N x 4 matrix of pairs stored by rows as a vector
   *  and t is a vector of length N)
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
  
  // Read matrix
  int count = 10;
  int n_rows = (argc - count)/5;
  arma::mat x(n_rows, 4);
  for (int i = 0; i < n_rows; i++){
    for (int j = 0; j < 4; j++){
      x(i, j) = atof(argv[count]);
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
  double result = logLikWnOuPairs(x, t, alpha, mu, sigma, maxK, etrunc);

  // Print result
  printf("%.4f", result);

  return 0;

}

//' @title Loglikelihood of MWN-OU in 2D when only the initial and final points are observed
//'
//' @description Computation of the loglikelihood for a MWN-OU diffusion (with diagonal diffusion matrix) from a sample of initial and final pairs of angles.
//'
//' @param x a matrix of dimension \code{c(n, 4)} of initial and final pairs of angles. Each row is an observation containing \eqn{(\phi_0, \psi_0, \phi_t, \psi_t)}.
//' They all must be in \eqn{[\pi,\pi)} so that the truncated wrapping by \code{maxK} windings is able to capture periodicity.
//' @param t either a scalar or a vector of length \code{n} containing the times the initial and final dihedrals. If \code{t} is a scalar, a common time is assumed.
//' @inheritParams dTpdWnOu2D
//' @inheritParams safeSoftMax
//' @return A scalar giving the final loglikelihood, defined as the sum of the loglikelihood of the initial angles according to the stationary density
//' and the loglikelihood of the transitions from initial to final angles.
//' @details A negative penalty is added if positive definiteness is violated. If the output value is Inf, -100 * N is returned instead.
//' @author Eduardo Garcia-Portugues (\email{egarcia@@math.ku.dk})
//' @examples
//' set.seed(345567)
//' x <- radToPiInt(matrix(rnorm(200, mean = pi), ncol = 4, nrow = 50))
//' alpha <- c(2, 1, -0.5)
//' mu <- c(0, pi)
//' sigma <- sqrt(c(2, 1))
//'
//' # The same
//' logLikWnOuPairs(x = x, t = 0.5, alpha = alpha, mu = mu, sigma = sigma)
//' sum(
//'   log(dStatWnOu2D(x = x[, 1:2], alpha = alpha, mu = mu, sigma = sigma)) +
//'   log(dTpdWnOu2D(x = x[, 3:4], x0 = x[, 1:2], t = 0.5, alpha = alpha, mu = mu,
//'                  sigma = sigma))
//' )
//' 
//' # Different times
//' logLikWnOuPairs(x = x, t = (1:50) / 50, alpha = alpha, mu = mu, sigma = sigma)
//' @export
// [[Rcpp::export]]
double logLikWnOuPairs(arma::mat x, arma::vec t, arma::vec alpha, arma::vec mu, arma::vec sigma, int maxK = 2, double etrunc = 30) {
  
  /*
  * Create basic objects
  */
  
  // Number of pairs
  int N = x.n_rows;
  
  // Create log-likelihoods
  double loglikinitial = 0;
  double logliktpd = 0;
  double loglik = 0;
  
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
  
  // Add a penalty to the loglikelihood in case any assumption is violated
  double penalty = 0;
  
  // Only positive definiteness can be violated with the parametrization of A
  double testalpha = alpha(0) * alpha(1) - alpha(2) * alpha(2);
  
  // Check positive definiteness
  if(testalpha <= 0) {
    
    // Add a penalty
    penalty = -testalpha * 10000 + 10;
    
    // Update alpha(2) such that testalpha > 0
    alpha(2) = std::signbit(alpha(2)) * sqrt(alpha(0) * alpha(1)) * 0.9999;
    
    // Reset A to a matrix with positive determinant
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
    arma::vec xmu = x.submat(i, 0, i, 1).t() - mu;
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
  
  // The unstandardized weights of the tpd give the required wrappings for the initial loglikelihood
  loglikinitial = accu(log(sum(exp(weightswindsinitial), 1)));
  
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
    arma::vec x00 = x.submat(i, 0, i, 1).t();;
    
    // Evaluation point x varying with i
    arma::vec xx = x.submat(i, 2, i, 3).t();;
    
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
  
  // Logarithm of tpd
  tpdfinal = log(tpdfinal);
  
  // Set log(0) to -trunc, as this is the truncation of the negative exponentials
  tpdfinal.elem(find_nonfinite(tpdfinal)).fill(-etrunc);
  
  // Log-likelihood from tpd
  logliktpd = sum(tpdfinal);
  
  // Final loglikelihood
  loglik = loglikinitial + logliktpd;
  
  // Check if it is finite
  if(!std::isfinite(loglik)) loglik = -100 * N;
  
  return loglik - penalty;
  
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
