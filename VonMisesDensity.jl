type VonMisesDensity
  mu::Float64
  kappa::Float64
  log_denominator::Float64

  function VonMisesDensity()
    new(-Inf, -Inf, -Inf)
  end

  function VonMisesDensity(mu::Float64, kappa::Float64)
    new(mu, kappa, log(2.0 * pi * besseli(0, kappa)))
  end

  function VonMisesDensity(vmd::VonMisesDensity)
    new(vmd.mu, vmd.kappa, vmd.log_denominator)
  end

end

function set_parameters(density::VonMisesDensity, mu::Float64, kappa::Float64)
    if kappa != density.kappa
      density.kappa = kappa
      density.log_denominator = log(2.0 * pi * besseli(0, kappa))
    end
    density.mu = mu
end

export logdensity
function logdensity(density::VonMisesDensity, angle::Float64, mu::Float64, kappa::Float64)
    set_parameters(density, mu, kappa)
    log_numerator = kappa * cos(angle-mu)
    return log_numerator - density.log_denominator
end

function logdensity(density::VonMisesDensity, angle::Float64)
    log_numerator = density.kappa * cos(angle-density.mu)
    return log_numerator - density.log_denominator
end

function sample(density::VonMisesDensity, rng::AbstractRNG)
  # Returns a sample from the Von Mises distribution.
  # Based on the algorithm of Best & Fisher (1979), as described in: Jupp PE
	# and Mardia KV,  "Directional Statistics",  John Wiley & Sons, 1999.

	# For kappas close to zero return a sample from the uniform distribution on the circle
  kappa = density.kappa
  mu = density.mu
	if kappa < 1e-6
		return 2.0 * pi * rand(rng)  # This should be a random number in the interval [0,1)
  end

  a = 1.0 + sqrt(1.0 + 4.0 * kappa * kappa)
  b = (a - sqrt(2.0 * a)) / (2.0 * kappa)
  r = (1.0 + b * b) / (2.0 * b)

  U1 = 0.0
  U2 = 0.0
  U3 = 0.0
  z = 0.0
  f = 0.0
  c = 0.0
  theta = 0.0

  while true
		U1 = rand(rng)
		z = cos(pi * U1)
		f = (1.0 + r * z) / (r + z)
		c = kappa * (r - f)

		U2 = rand(rng)

    !(( c * (2.0 - c) - U2 <= 0.0 ) && ( c * exp(1.0 - c) - U2 < 0.0 )) && break
  end

	U3 = rand(rng)

	if U3 - 0.5 > 0.0
		theta = mu + acos(f)
	else
		theta = mu - acos(f)
  end

  theta = mod2pi(theta)
  if theta > pi
    return theta -2.0*pi
  else
    return theta
  end
end
