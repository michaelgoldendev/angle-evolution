include("UtilsModule.jl")
using UtilsModule

export WrappedNormalOUNode
type WrappedNormalOUNode
  x::Array{Float64,1}
  t::Float64
  mu::Array{Float64,1}
  alpha::Array{Float64,1}
  sigma::Array{Float64,1}
  maxK::Int
  etrunc::Float64
  vstores::Array{Float64,1}
  vstoret::Array{Float64,1}
  weightswindsinitial::Array{Float64,1}
  logweightswindsinitial::Array{Float64,1}
  A::Array{Float64,2}
  Sigmamat::Array{Float64,2}
  oneoverSigmamat::Array{Float64,2}
  invGammat::Array{Float64,2}

  lk::Int
  twokpi::Array{Float64,1}
  twokepivec::Array{Float64,1}
  twokapivec::Array{Float64,1}
  penalty::Float64
  invSigmaA::Array{Float64,2}
  xmuinvSigmaA::Array{Float64,1}
  lognormconstSigmaA::Float64
  ASigma::Array{Float64,2}
  ASigmaA::Array{Float64,2}
  lognormconstGammat::Float64
  Gammat::Array{Float64,2}
  ExptA::Array{Float64,2}
  s::Float64
  q::Float64

  function WrappedNormalOUNode(maxK::Int)
    lk = 2*maxK+1
    new(zeros(Float64,4), 0.0, zeros(Float64,2), zeros(Float64,3), zeros(Float64,2), maxK, 100.0, zeros(Float64,lk*lk), zeros(Float64,lk*lk), zeros(Float64,lk*lk), zeros(Float64,lk*lk), zeros(Float64,2,2), zeros(Float64,2,2), zeros(Float64,2,2), zeros(Float64,2,2),lk, zeros(Float64,lk*lk), zeros(Float64,2),zeros(Float64,2))
  end

  function WrappedNormalOUNode(node::WrappedNormalOUNode)
    lk = 2*node.maxK+1
    #=
    maxK = 1
    lk = 2*maxK+1
    node.maxK = maxK=#
    new(copy(node.x), node.t, copy(node.mu), copy(node.mu), copy(node.alpha), copy(node.sigma), node.maxK, node.etrunc, zeros(Float64,lk*lk), zeros(Float64,lk*lk), zeros(Float64,lk*lk), zeros(Float64,2,2), zeros(Float64,2,2), zeros(Float64,2,2), zeros(Float64,2,2),lk, zeros(Float64,lk*lk), zeros(Float64,2),zeros(Float64,2))
  end
end

function set_parameters(node::WrappedNormalOUNode, t::Float64, muphi::Float64, mupsi::Float64, alpha_phi::Float64, alpha_psi::Float64, alpha_rho::Float64, sigma_phi::Float64, sigma_psi::Float64)
  #rho = alpha_rho*alpha_phi*alpha_psi
  #if node.t != t || node.mu[1] != muphi || node.mu[2] != mupsi || node.alpha[1] != alpha_phi || node.alpha[2] != alpha_psi || node.alpha[3] != alpha_rho || node.sigma[1] != sigma_phi || node.sigma[2] != sigma_psi
    #println("notcached")
    #node.t = t
    node.mu[1] = muphi
    node.mu[2] = mupsi
    node.alpha[1] = alpha_phi
    node.alpha[2] = alpha_psi
    node.alpha[3] = alpha_rho
    node.sigma[1] = sigma_phi
    node.sigma[2] = sigma_psi
    set_parameters(node, node.mu, node.alpha, node.sigma)
    set_parameters(node, t)
  #end
end

log2pi = log(2.0 * pi)
function set_parameters(node::WrappedNormalOUNode, mu::Array{Float64,1}, alpha::Array{Float64,1}, sigma::Array{Float64,1})
    node.mu = mu
    node.alpha = alpha
    node.sigma = sigma

    quo = sqrt(sigma[1] / sigma[2])
    node.A[1,1] = alpha[1]
    node.A[2,2] = alpha[2]
    node.A[1,2] = alpha[3]*quo
    node.A[2,1] = alpha[3]/quo

    testalpha = alpha[1] * alpha[2] - alpha[3] * alpha[3]
    node.penalty = 0.0
    if testalpha <= 0.0
      node.penalty = -testalpha*100000.0 + 100.0

      alpha[3] = signbit(alpha[3]) * sqrt(alpha[1] * alpha[2]) * 0.9999
      node.A[1,2] = alpha[3] * quo
      node.A[2,1] =  alpha[3] / quo
      #return
    end


    node.Sigmamat[1,1] = sigma[1]
    node.Sigmamat[2,2] = sigma[2]
    node.lk = 2 * node.maxK + 1
    node.twokpi = linspace(-node.maxK*2.0*pi,node.maxK*2.0*pi,node.lk)

    node.oneoverSigmamat[1,1] = 1.0/sigma[1]
    node.oneoverSigmamat[2,2] = 1.0/sigma[2]
    node.invSigmaA = 2.0 * node.oneoverSigmamat * node.A
    node.lognormconstSigmaA  = -log(2.0*pi) + logdet(node.invSigmaA) / 2.0

    node.ASigma = node.A * node.Sigmamat
    node.ASigmaA = node.ASigma * transpose(node.A)
    node.ASigma += transpose(node.ASigma)

    node.s = trace(node.A) / 2.0
    s = node.s
    node.q = sqrt(abs(det(node.A - s * eye(2))))

    if node.q == 0.0
      node.q = 1e-6
    end
    q = node.q

    fill!(node.weightswindsinitial, 0.0)
    for wek1=1:node.lk
      for wek2=1:node.lk
        node.vstores[(wek1-1)*node.lk+wek2] = ((node.invSigmaA[1,1]*node.twokpi[wek1] + node.invSigmaA[1,2]*node.twokpi[wek2])*node.twokpi[wek1] + (node.invSigmaA[2,1]*node.twokpi[wek1] + node.invSigmaA[2,2]*node.twokpi[wek2])*node.twokpi[wek2])/2.0
      end
    end
end

function set_parameters(node::WrappedNormalOUNode, t::Float64)
    if node.t != t
      node.t = t
      s = node.s
      q = node.q

      q2 = q * q
      s2 = s * s
      est = exp(s * t)
      e2st = est * est
      inve2st = 1.0 / e2st
      c2st = exp(2.0 * q * t)
      s2st = (c2st - 1.0/c2st) / 2.0
      c2st = (c2st + 1.0/c2st) / 2.0

      cte = inve2st / (4.0 * q2 * s * (s2 - q2))
      integral1 = cte * (- s2 * (3.0 * q2 + s2) * c2st - q * s * (q2 + 3.0 * s2) * s2st - q2 * (q2 - 5.0 * s2) * e2st + (q2 - s2) *  (q2 - s2))
      integral2 = cte * s * ((q2 + s2) * c2st + 2.0 * q * s * s2st - 2.0 * q2 * e2st + q2 - s2)
      integral3 = cte * (- s * (s * c2st + q * s2st) + (e2st - 1.0) * q2 + s2)

      node.Gammat = integral1 * node.Sigmamat + integral2 * node.ASigma + integral3 * node.ASigmaA

      eqt = exp(q * t)
      cqt = (eqt + 1.0/eqt) / 2.0
      sqt = (eqt - 1.0/eqt) / 2.0
      node.ExptA = ((cqt + s * sqt / q) * eye(2) - (sqt / q * node.A)) / est

      z  = 1.0 / (node.Gammat[1,1]*node.Gammat[2,2]-node.Gammat[1,2]*node.Gammat[2,1])
      node.invGammat[1,1] = z*node.Gammat[2,2]
      node.invGammat[1,2] = -z*node.Gammat[1,2]
      node.invGammat[2,1] = -z*node.Gammat[2,1]
      node.invGammat[2,2] = z*node.Gammat[1,1]
      node.lognormconstGammat = -log2pi + logdet(node.invGammat) / 2.0

      fill!(node.weightswindsinitial, 0.0)
      for wek1=1:node.lk
        for wek2=1:node.lk
          node.vstoret[(wek1-1)*node.lk+wek2] = ((node.invGammat[1,1]*node.twokpi[wek1] + node.invGammat[1,2]*node.twokpi[wek2])*node.twokpi[wek1] + (node.invGammat[2,1]*node.twokpi[wek1] + node.invGammat[2,2]*node.twokpi[wek2])*node.twokpi[wek2])/2.0
        end
      end
  end
end

export loglikwndtpd
function loglikwndtpd(node::WrappedNormalOUNode, phi0::Float64, psi0::Float64, phit::Float64, psit::Float64)
  if node.penalty > 0.0
    return -Inf
    #return -node.penalty
  end
  node.x[1] = phi0
  node.x[2] = psi0
  node.x[3] = phit
  node.x[4] = psit

  xmu = node.x[1:2] - node.mu
  node.xmuinvSigmaA = node.invSigmaA * xmu
  xmuinvSigmaAxmudivtwo = (node.xmuinvSigmaA[1]*xmu[1] + node.xmuinvSigmaA[2]*xmu[2]) / 2.0

  logtpdfinal = -Inf
  x0 = node.x[1:2]
  for wek1=1:node.lk
    node.twokepivec[1] = node.twokpi[wek1]
    for wek2=1:node.lk # Loop on the winding weight K2
      index = (wek1-1) * (node.lk) + wek2

      exponent = xmuinvSigmaAxmudivtwo + (node.xmuinvSigmaA[1]*node.twokpi[wek1]+node.xmuinvSigmaA[2]*node.twokpi[wek2]) + node.vstores[index] - node.lognormconstSigmaA
      if exponent <= node.etrunc
        node.logweightswindsinitial[index] = -exponent
      else
        node.logweightswindsinitial[index] = -Inf
      end

      if node.logweightswindsinitial[index] > -Inf
        node.twokepivec[2] = node.twokpi[wek2]
        mut = node.mu + node.ExptA * (x0 + node.twokepivec - node.mu)
        xmut = node.x[3:4] - mut
        xmutinvGammat = node.invGammat * xmut;
        xmutinvGammatxmutdiv2 = (xmutinvGammat[1]*xmut[1] + xmutinvGammat[2]*xmut[2]) / 2.0

        logtpdintermediate = -Inf
        for wak1=1:node.lk # Loop in the winding wrapping K1
          for wak2=1:node.lk # Loop in the winding wrapping K2
            # Decomposition of the exponent
             exponent = xmutinvGammatxmutdiv2 + (xmutinvGammat[1]*node.twokpi[wak1]+xmutinvGammat[2]*node.twokpi[wak2]) + node.vstoret[(wak1-1)*node.lk+wak2] - node.lognormconstGammat
             #if exponent < node.etrunc # Truncate the negative exponential
                logtpdintermediate = logsumexp(logtpdintermediate,-exponent)
            #end
          end
        end
        logtpdfinal = logsumexp(logtpdfinal,logtpdintermediate+node.logweightswindsinitial[index])
      end
    end
  end

  ll = logtpdfinal - node.penalty
  if isnan(ll) || ll == -Inf
    #println(node.logweightswindsinitial)
    #println(weightswindsinitialsum)
    #println(tpdfinal)
    #println(node.penalty)
    #println(logtpdfinal)
    #println("A",ll,"\t",phi0,"\t",phit,"\t",psi0,"\t",psit,"\t",length(node.logweightswindsinitial))
    return -1e10
  else
    return ll
  end

end

export loglikwndstat
function loglikwndstat(node::WrappedNormalOUNode, phi::Float64, psi::Float64)
  if node.penalty > 0.0
    return -1e10
    #return -node.penalty
  end

  node.x[1] = phi
  node.x[2] = psi
  xmu = node.x[1:2] - node.mu
  node.xmuinvSigmaA = node.invSigmaA * xmu
  xmuinvSigmaAxmudivtwo = (node.xmuinvSigmaA[1]*xmu[1] + node.xmuinvSigmaA[2]*xmu[2]) / 2.0

  logweightswindsinitialsum = -Inf
  for wek1=1:node.lk
    for wek2=1:node.lk
      exponent = xmuinvSigmaAxmudivtwo + (node.xmuinvSigmaA[1]*node.twokpi[wek1]+node.xmuinvSigmaA[2]*node.twokpi[wek2]) + node.vstores[(wek1-1)*node.lk + wek2] - node.lognormconstSigmaA
      logweightswindsinitialsum = logsumexp(logweightswindsinitialsum,-exponent)
    end
  end
  ll = logweightswindsinitialsum - node.penalty
  if isnan(ll)  || ll == -Inf
    #println("B",ll,"\t",x,"\t",length(node.weightswindsinitial))
    return -1e10
  else
    return ll
  end
end

function sampstat(node::WrappedNormalOUNode, rng::AbstractRNG)
    #invASigma = inv_sympd(A) * diagmat(sigma) / 2;
    invASigma = inv(node.A)*diagm(node.sigma)/2.0

    x = randn(rng,2)
    #x = transpose(x)*chol(node.invSigmaA)
    x = transpose(x)*chol(invASigma)
    x += transpose(node.mu)
    x -= floor((x + pi) / (2.0 * pi)) * (2.0 * pi)
    return x
end

function samptrans(node::WrappedNormalOUNode, rng::AbstractRNG, x0::Array{Float64,1}, t::Float64)
   if (node.t != t)
    set_parameters(node, t)
  end

  mut = zeros(Float64, node.lk*node.lk, 2)
  for i=1:node.lk*node.lk
    mut[i,:] = node.mu + node.ExptA * (x0 - node.mu)
  end

  xmu = x0 - node.mu
  node.xmuinvSigmaA = node.invSigmaA * xmu
  xmuinvSigmaAxmudivtwo = (node.xmuinvSigmaA[1]*xmu[1] + node.xmuinvSigmaA[2]*xmu[2]) / 2.0

  weightswindsinitialsum = 0.0
  tpdfinal = 0.0
  for wek1=1:node.lk
    node.twokepivec[1] = node.twokpi[wek1]
    for wek2=1:node.lk
      node.twokepivec[2] = node.twokpi[wek2]
      index = (wek1-1) * (node.lk) + wek2

      exponent = xmuinvSigmaAxmudivtwo + (node.xmuinvSigmaA[1]*node.twokpi[wek1]+node.xmuinvSigmaA[2]*node.twokpi[wek2]) + node.vstores[index] - node.lognormconstSigmaA
      if exponent <= node.etrunc
        v = exp(-exponent)
        node.weightswindsinitial[index] = v
        weightswindsinitialsum += v
        mut[index,:]  += transpose(node.ExptA * node.twokepivec)
      else
        node.weightswindsinitial[index] = 0.0
        mut[index,:] = 0.0
      end
    end
  end
  node.weightswindsinitial /= weightswindsinitialsum


  r = UtilsModule.sample(rng,node.weightswindsinitial)
  mutx = mut[r,:]

  x = randn(rng,2)
  x = transpose(x)*chol(node.Gammat)
  x += mutx
  x -= floor((x + pi) / (2.0 * pi)) * (2.0 * pi)

  return x

end

#=
x = zeros(Float64,2,4)
x[1,1] = 0.5
x[1,2] = 3.0
x[1,3] = 1.0
x[1,4] = 3.5
x[2,1] = 0.5
x[2,2] = 3.0
x[2,3] = 1.0
x[2,4] = 3.5
t = 1.0
mu = Float64[0.0, 1.5]
alpha = Float64[1.0, 1.0, 0.0]
sigma = Float64[1.0, 1.0]
maxK = 1
etrunc = 20.0

me = WrappedNormalOUNode(1)

set_parameters(me, 0.35, mu[1], mu[2], alpha[1], alpha[2], alpha[3],sigma[1], sigma[2])
println(me)
println((me.lk*me.lk))
=#
#set_parameters(me, t, mu, alpha, sigma, maxK, etrunc)
#=
for i=1:1000000
  x = (rand(Float64,4) - 0.5)*2.0*pi
  res = loglikwndtpd(me, x)
  if res > -2.2
    println(x,"\t", res)
  end
end
=#

#=
y = Float64[0.5, 3.0, 1.0, 3.5]
set_parameters(me, 0.35)
println(loglikwndtpd(me, y[1], y[2], y[3], y[4]))
println(loglikwndstat(me, y[1], y[2]))
set_parameters(me, 1.0)
println(loglikwndtpd(me, y[1], y[2], y[3], y[4]))
println(loglikwndstat(me, y[1], y[2]))

rng = MersenneTwister(3142049220144)
s = transpose(zeros(Float64,2))
for i=1:5
  v = samptrans(me, rng, y[1:2], 0.5)
  s += v
  println(v)

end

println("avg=",s/100.0)=#

t = 1.0
mu = Float64[-2.0, 0.0]
alpha = Float64[0.75, 1.0, 0.2]
sigma = Float64[0.25, 0.5]
etrunc = 100.0
0
me = WrappedNormalOUNode(1)
rng = MersenneTwister(3142049220144)
set_parameters(me, 0.35, mu[1], mu[2], alpha[1], alpha[2], alpha[3],sigma[1], sigma[2])
set_parameters(me, 0.35)
println(loglikwndstat(me,1.0,2.0))
println(loglikwndstat(me,0.1,0.75))
println(loglikwndstat(me,0.1,0.1))
println(loglikwndtpd(me,0.1, 0.75, -0.1, 0.8))
println(loglikwndtpd(me,1.0, 0.5, 0.4, -1.0))
println(length(me.weightswindsinitial))

println(samptrans(me,rng,Float64[-0.5,-0.5],10.0))
println(samptrans(me,rng,Float64[-0.5,-0.5],10.0))
println(samptrans(me,rng,Float64[-0.5,-0.5],10.0))
println(samptrans(me,rng,Float64[-0.5,-0.5],10.0))
println(samptrans(me,rng,Float64[-0.5,-0.5],10.0))

println(samptrans(me,rng,Float64[-0.5,-0.5],1.0))
println(samptrans(me,rng,Float64[-0.5,-0.5],1.0))
println(samptrans(me,rng,Float64[-0.5,-0.5],1.0))
println(samptrans(me,rng,Float64[-0.5,-0.5],1.0))
println(samptrans(me,rng,Float64[-0.5,-0.5],1.0))


println(samptrans(me,rng,Float64[-0.5,-0.5],0.1))
println(samptrans(me,rng,Float64[-0.5,-0.5],0.1))
println(samptrans(me,rng,Float64[-0.5,-0.5],0.1))
println(samptrans(me,rng,Float64[-0.5,-0.5],0.001))
println(samptrans(me,rng,Float64[-0.5,-0.5],0.001))
