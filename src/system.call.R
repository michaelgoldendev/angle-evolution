# Set directory
setwd("/Path/to/cpp/binaries")

# Parameters
t <- 1
time <- rep(1, 2)
mu <- c(0, -0.57)
alpha <- c(1, 2, 0)
sigma <- c(1, 1.0)
pairs <- matrix(c(-1, 1, 0, 1, 0, 1, -1, -1), nrow = 2, ncol = 4, byrow = TRUE)
x <- matrix(c(-1, 1, 0, 1), nrow = 2, ncol = 2, byrow = TRUE)
N <- 10
maxK <- 1
etrunc <- 25

## First versions

# Call logliktime
loglik <- as.double(system(paste("./loglik", t, paste(mu, collapse = " "), paste(alpha, collapse = " "), paste(sigma, collapse = " "), maxK, etrunc, paste(as.vector(t(pairs)), collapse = " ")), intern = TRUE))
loglik

# Call sampstat
samp <- system(paste("./sampstat", N, paste(mu, collapse = " "), paste(alpha, collapse = " "), paste(sigma, collapse = " "), paste(as.vector(t(pairs)), collapse = " ")), intern = TRUE)
samp <- matrix(scan(text = samp, what = numeric(), quiet = TRUE), nrow = N, ncol = 4, byrow = TRUE)
samp

# Call samptrans
dens <- system(paste("./samptrans", paste(mu, collapse = " "), paste(alpha, collapse = " "), paste(sigma, collapse = " "), maxK, etrunc, paste(as.vector(t(x)), collapse = " ")), intern = TRUE)
dens <- scan(text = dens, what = numeric(), quiet = TRUE)
dens

## Newer versions with vectorized times - C++ 11 dependent!

# Call logliktime
loglik <- as.double(system(paste("./logliktime", paste(mu, collapse = " "), paste(alpha, collapse = " "), paste(sigma, collapse = " "), maxK, etrunc, paste(as.vector(t(pairs)), collapse = " "), paste(time, collapse = " ")), intern = TRUE))
loglik



