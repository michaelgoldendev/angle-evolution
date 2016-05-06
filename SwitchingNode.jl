using UtilsModule

include("AAPairNode.jl")
include("DiffusionNode.jl")
include("SecondaryStructureNode.jl")
include("Sequence.jl")

export SwitchingNode
type SwitchingNode
  aapairnode_r1::AAPairNode
  aapairnode_r2::AAPairNode
  diffusion_r1::DiffusionNode
  diffusion_r2::DiffusionNode
  alpha::Float64
  pi_r1::Float64
  ss_r1::SecondaryStructureNode
  ss_r2::SecondaryStructureNode

  function SwitchingNode()
      aapairnode_r1 = AAPairNode()
      load_parameters(aapairnode_r1, "resources/lg_LG.PAML.txt")

      aapairnode_r2 = AAPairNode()
      load_parameters(aapairnode_r2, "resources/lg_LG.PAML.txt")

      new(aapairnode_r1, aapairnode_r2, DiffusionNode(), DiffusionNode(), 2.0, 0.5, SecondaryStructureNode(), SecondaryStructureNode())
  end

  function SwitchingNode(node::SwitchingNode)
    new(AAPairNode(node.aapairnode_r1),AAPairNode(node.aapairnode_r2),DiffusionNode(node.diffusion_r1),DiffusionNode(node.diffusion_r2),node.alpha, node.pi_r1, SecondaryStructureNode(node.ss_r1), SecondaryStructureNode(node.ss_r2))
  end
end

export set_parameters
function set_parameters(node::SwitchingNode, x::Array{Float64,1})
  set_parameters(node.aapairnode_r1, x[1:20]/sum(x[1:20]), 1.0)
  set_parameters(node.aapairnode_r2, x[21:40]/sum(x[21:40]), 1.0)
  set_parameters(node.diffusion_r1, x[41], mod2pi(x[42]+pi)-pi, x[43], x[44], mod2pi(x[45]+pi)-pi, x[46], x[47], 1.0)
  set_parameters(node.diffusion_r2, x[48], mod2pi(x[49]+pi)-pi, x[50], x[51], mod2pi(x[52]+pi)-pi, x[53], x[54], 1.0)
  node.alpha = x[55]
  node.pi_r1 = x[56]
  set_parameters(node.ss_r1.ctmc, x[57:59]/sum(x[57:59]), 1.0)
  set_parameters(node.ss_r2.ctmc, x[60:62]/sum(x[60:62]), 1.0)
end

export get_parameters
function get_parameters(node::SwitchingNode)
  initial = zeros(Float64,62)
  for i=1:20
    initial[i] = node.aapairnode_r1.eqfreqs[i]
    initial[20+i] = node.aapairnode_r2.eqfreqs[i]
  end
  initial[41] = node.diffusion_r1.alpha_phi
  initial[42] = node.diffusion_r1.mu_phi
  initial[43] = node.diffusion_r1.sigma_phi
  initial[44] = node.diffusion_r1.alpha_psi
  initial[45] = node.diffusion_r1.mu_psi
  initial[46] = node.diffusion_r1.sigma_psi
  initial[47] = node.diffusion_r1.alpha_rho

  initial[48] = node.diffusion_r2.alpha_phi
  initial[49] = node.diffusion_r2.mu_phi
  initial[50] = node.diffusion_r2.sigma_phi
  initial[51] = node.diffusion_r2.alpha_psi
  initial[52] = node.diffusion_r2.mu_psi
  initial[53] = node.diffusion_r2.sigma_psi
  initial[54] = node.diffusion_r2.alpha_rho
  initial[55] = node.alpha
  initial[56] = node.pi_r1
  for i=1:3
    initial[56+i] = node.ss_r1.ctmc.eqfreqs[i]
    initial[59+i] = node.ss_r2.ctmc.eqfreqs[i]
  end

  return initial
end

export get_data_lik_x0
function get_data_lik_x0(node::SwitchingNode, x0::Int, phi_x0::Float64, psi_x0::Float64, ss0::Int, t::Float64)

  ll1 = get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0) + get_data_lik(node.ss_r1.ctmc, ss0)
  ll2 = get_data_lik(node.aapairnode_r2, x0) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0) + get_data_lik(node.ss_r2.ctmc, ss0)
  pi_r1 = node.pi_r1
  pi_r2 = 1.0-pi_r1
  return logsumexp(log(pi_r1)+ll1, log(pi_r2)+ll2)
end

export get_data_lik_xt
function get_data_lik_xt(node::SwitchingNode, x0::Int, phi_x0::Float64, psi_x0::Float64, ss0::Int, t::Float64)
  ll1 = get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0) + get_data_lik(node.ss_r1.ctmc, ss0)
  ll2 = get_data_lik(node.aapairnode_r2, x0) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0) + get_data_lik(node.ss_r2.ctmc, ss0)
  pi_r1 = node.pi_r1
  pi_r2 = 1.0-pi_r1
  return logsumexp(log(pi_r1)+ll1, log(pi_r2)+ll2)
end

export get_data_lik
function get_data_lik(node::SwitchingNode, x0::Int, xt::Int, phi_x0::Float64, psi_x0::Float64, phi_xt::Float64, psi_xt::Float64, ss_x0::Int, ss_xt::Int, t::Float64)
  pi_r1 = node.pi_r1
  pi_r2 = 1.0-pi_r1
  wt = exp(-node.alpha*t)
  r1r1 = log(pi_r1*(wt+pi_r1*(1.0-wt))) + get_data_lik(node.aapairnode_r1, x0, xt, t) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0, phi_xt, psi_xt, t) + get_data_lik(node.ss_r1.ctmc, ss_x0, ss_xt, t)
  logtransprob = log(pi_r1*pi_r2*(1.0-wt))
  r1r2 = logtransprob + get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0) + get_data_lik(node.ss_r1.ctmc, ss_x0) + get_data_lik(node.aapairnode_r2, xt) + get_data_lik(node.diffusion_r2, phi_xt, psi_xt) + get_data_lik(node.ss_r2.ctmc, ss_xt)
  r2r1 = logtransprob + get_data_lik(node.aapairnode_r2, x0) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0) + get_data_lik(node.ss_r2.ctmc, ss_x0) + get_data_lik(node.aapairnode_r1, xt) + get_data_lik(node.diffusion_r1, phi_xt, psi_xt) + get_data_lik(node.ss_r1.ctmc, ss_xt)
  r2r2 = log(pi_r2*(wt+pi_r2*(1.0-wt))) + get_data_lik(node.aapairnode_r2, x0, xt, t) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0, phi_xt, psi_xt, t) + get_data_lik(node.ss_r2.ctmc, ss_x0, ss_xt, t)
  v = Float64[r1r1, r1r2, r2r1, r2r2]

  return logsumexp(v)
end

export get_regime_logprobs
function get_regime_logprobs(node::SwitchingNode, x0::Int, xt::Int, phi_x0::Float64, psi_x0::Float64, phi_xt::Float64, psi_xt::Float64, ss_x0::Int, ss_xt::Int, t::Float64)
  pi_r1 = node.pi_r1
  pi_r2 = 1.0-pi_r1
  wt = exp(-node.alpha*t)
  r1r1 = log(pi_r1*(wt+pi_r1*(1.0-wt))) + get_data_lik(node.aapairnode_r1, x0, xt, t) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0, phi_xt, psi_xt, t) + get_data_lik(node.ss_r1.ctmc, ss_x0, ss_xt, t)
  logtransprob = log(pi_r1*pi_r2*(1.0-wt))
  r1r2 = logtransprob + get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0) + get_data_lik(node.ss_r1.ctmc, ss_x0) + get_data_lik(node.aapairnode_r2, xt) + get_data_lik(node.diffusion_r2, phi_xt, psi_xt) + get_data_lik(node.ss_r2.ctmc, ss_xt)
  r2r1 = logtransprob + get_data_lik(node.aapairnode_r2, x0) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0) + get_data_lik(node.ss_r2.ctmc, ss_x0) + get_data_lik(node.aapairnode_r1, xt) + get_data_lik(node.diffusion_r1, phi_xt, psi_xt) + get_data_lik(node.ss_r1.ctmc, ss_xt)
  r2r2 = log(pi_r2*(wt+pi_r2*(1.0-wt))) + get_data_lik(node.aapairnode_r2, x0, xt, t) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0, phi_xt, psi_xt, t) + get_data_lik(node.ss_r2.ctmc, ss_x0, ss_xt, t)
  v = Float64[r1r1, r1r2, r2r1, r2r2]
end

export get_regime_pair_lik
function get_regime_pair_lik(node::SwitchingNode, r1::Int, r2::Int, t::Float64)
  pi_r1 = node.pi_r1
  pi_r2 = 1.0-pi_r1
  wt = exp(-node.alpha*t)
  if r1 == 1 && r2 == 1
    return log(pi_r1*(wt+pi_r1*(1.0-wt)))
  elseif (r1 == 1 && r2 == 2) || (r1 == 2 && r2 == 1)
    return log(pi_r1*pi_r2*(1.0-wt))
  else
    return log(pi_r2*(wt+pi_r2*(1.0-wt)))
  end
end

export sample_regimes
function sample_regimes(node::SwitchingNode, rng::AbstractRNG, seq1::Sequence, seq2::Sequence, i::Int, j::Int, t::Float64)
  x0 = 0
  phi_x0 = -1000.0
  psi_x0 = -1000.0
  ss_x0 = 0
  if i > 0
    x0 = seq1.seq[i]
    phi_x0 = seq1.phi[i]
    psi_x0 = seq1.psi[i]
    ss_x0 = seq1.ss[i]
  end

  xt = 0
  phi_xt = -1000.0
  psi_xt = -1000.0
  ss_xt = 0
  if j > 0
    xt = seq2.seq[j]
    phi_xt = seq2.phi[j]
    psi_xt = seq2.psi[j]
    ss_xt = seq2.ss[j]
  end
  return GumbelSample(rng, get_regime_logprobs(node, x0, xt, phi_x0, psi_x0, phi_xt, psi_xt, ss_x0, ss_xt, t))
end

export sample
function sample(rng::AbstractRNG, node::SwitchingNode, x0::Int, xt::Int, phi_x0::Float64, psi_x0::Float64, phi_xt::Float64, psi_xt::Float64, ss_x0::Int, ss_xt::Int, t::Float64)

  v = get_regime_logprobs(node, x0, xt, phi_x0, psi_x0, phi_xt, psi_xt, ss_x0, ss_xt, t)
  c = UtilsModule.GumbelSample(rng,v)

  ret2 = (x0,xt,(phi_x0,phi_xt),(psi_x0,psi_xt),ss_x0,ss_xt)
  ret = ret2

  if c == 1
    a,b = sample(node.aapairnode_r1, rng, x0, xt, t)
    phi,psi = sample_phi_psi(node.diffusion_r1, rng, phi_x0, phi_xt, psi_x0, psi_xt, t)
    c,d = sample(node.ss_r1.ctmc, rng, ss_x0, ss_xt, t)
    ret = (a,b,phi,psi,c,d)
  elseif c == 2
    a = sample(node.aapairnode_r1, rng, x0, 0, t)[1]
    b = sample(node.aapairnode_r2, rng, xt, 0, t)[1]
    phi1, psi1 = sample_phi_psi(node.diffusion_r1, rng, phi_x0, -1000.0, psi_x0, -1000.0, t)
    phi2, psi2 = sample_phi_psi(node.diffusion_r2, rng, -1000.0, phi_xt, -1000.0, psi_xt, t)
    phi = (phi1[1],phi2[2])
    psi = (psi1[1],psi2[2])
    c = sample(node.ss_r1.ctmc, rng, ss_x0, 0, t)[1]
    d = sample(node.ss_r2.ctmc, rng, ss_xt, 0, t)[1]
    ret = (a,b,phi,psi,c,d)
  elseif c == 3
    a = sample(node.aapairnode_r2, rng, x0, 0, t)[1]
    b = sample(node.aapairnode_r1, rng, xt, 0, t)[1]
    phi1, psi1 = sample_phi_psi(node.diffusion_r2, rng, phi_x0, -1000.0, psi_x0, -1000.0, t)
    phi2, psi2 = sample_phi_psi(node.diffusion_r1, rng, -1000.0, phi_xt, -1000.0, psi_xt, t)
    phi = (phi1[1],phi2[2])
    psi = (psi1[1],psi2[2])
    c = sample(node.ss_r2.ctmc, rng, ss_x0, 0, t)[1]
    d = sample(node.ss_r1.ctmc, rng, ss_xt, 0, t)[1]
    ret = (a,b,phi,psi,c,d)
  else
    a,b = sample(node.aapairnode_r2, rng, x0, xt, t)
    phi,psi = sample_phi_psi(node.diffusion_r2, rng, phi_x0, phi_xt, psi_x0, psi_xt, t)
    c,d = sample(node.ss_r2.ctmc, rng, ss_x0, ss_xt, t)
    ret = (a,b,phi,psi,c,d)
  end
  return ret
end

