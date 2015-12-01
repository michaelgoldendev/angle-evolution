# Set directory
setwd("/Path/to/cpp/binaries")

# Parameters
t <- 1
mu <- c(0, -0.57)
alpha <- c(1, 2, 0)
sigma <- c(1, 1.0)
pairs <- matrix(c(-1, 1, 0, 1, 0, 1, -1, -1), nrow = 2, ncol = 4, byrow = TRUE)
x <- matrix(c(-1, 1, 0, 1), nrow = 2, ncol = 2, byrow = TRUE)
N <- 10
maxK <- 1
etrunc <- 25

# Call tpd
loglik <- as.double(system(paste("./maintpd", t, paste(mu, collapse = " "), paste(alpha, collapse = " "), paste(sigma, collapse = " "), maxK, etrunc, paste(as.vector(t(pairs)), collapse = " ")), intern = TRUE))
loglik

# Call sample
samp <- system(paste("./mainsamp", N, paste(mu, collapse = " "), paste(alpha, collapse = " "), paste(sigma, collapse = " "), paste(as.vector(t(pairs)), collapse = " ")), intern = TRUE)
samp <- matrix(scan(text = samp, what = numeric(), quiet = TRUE), nrow = N, ncol = 4, byrow = TRUE)
samp

# Call dens
dens <- system(paste("./maindens", paste(mu, collapse = " "), paste(alpha, collapse = " "), paste(sigma, collapse = " "), maxK, etrunc, paste(as.vector(t(x)), collapse = " ")), intern = TRUE)
dens <- scan(text = dens, what = numeric(), quiet = TRUE)
dens

