include("Utils.jl")
include("AAPairNode.jl")
include("DiffusionNode.jl")

type SwitchingNode
  aapairnode_r1::AAPairNode
  aapairnode_r2::AAPairNode
  diffusion_r1::DiffusionNode
  diffusion_r2::DiffusionNode
  alpha::Float64
  pi_r1::Float64

  function SwitchingNode()
      aapairnode_r1 = AAPairNode()
      load_parameters(aapairnode_r1, "/home/michael/dev/moju/lg_LG.PAML.txt")

      aapairnode_r2 = AAPairNode()
      load_parameters(aapairnode_r2, "/home/michael/dev/moju/lg_LG.PAML.txt")

      new(aapairnode_r1, aapairnode_r2, DiffusionNode(), DiffusionNode(), 2.0, 0.5)
  end

  function SwitchingNode(node::SwitchingNode)
    new(AAPairNode(node.aapairnode_r1),AAPairNode(node.aapairnode_r2),DiffusionNode(node.diffusion_r1),DiffusionNode(node.diffusion_r2),node.alpha, node.pi_r1)
  end
end

function get_data_lik_x0(node::SwitchingNode, x0::Int, phi_x0::Float64, psi_x0::Float64, t::Float64)
  #return get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0)
  ll1 = get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0)
  ll2 = get_data_lik(node.aapairnode_r2, x0) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0)
  pi_r1 = node.pi_r1
  pi_r2 = 1.0-pi_r1
  return logsumexp(log(pi_r1)+ll1, log(pi_r2)+ll2)
end

function get_data_lik_xt(node::SwitchingNode, x0::Int, phi_x0::Float64, psi_x0::Float64, t::Float64)
  #=ll1 = get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0)
  ll2 = get_data_lik(node.aapairnode_r2, x0) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0)
  probr1 = exp(-node.alpha*t)
  probr2 = 1.0 - probr1

  return logsumexp(log(probr1)+ll1, log(probr2)+ll2)=#
  #=
  ll1 = get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0)
  ll2 = get_data_lik(node.aapairnode_r2, x0) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0)
  probr1 = exp(-node.alpha*t)
  probr2 = 1.0 - probr1

  return logsumexp(log(probr1)+ll1, log(probr2)+ll2)=#
  ll1 = get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0)
  ll2 = get_data_lik(node.aapairnode_r2, x0) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0)
  pi_r1 = node.pi_r1
  pi_r2 = 1.0-pi_r1
  return logsumexp(log(pi_r1)+ll1, log(pi_r2)+ll2)


end

function get_data_lik(node::SwitchingNode, x0::Int, xt::Int, phi_x0::Float64, psi_x0::Float64, phi_xt::Float64, psi_xt::Float64, t::Float64)
  #=
  translik =  get_data_lik(node.aapairnode_r1, x0, xt, t) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0, phi_xt, psi_xt, t)

  switchll = get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0)
  switchll += get_data_lik(node.aapairnode_r2, xt) + get_data_lik(node.diffusion_r2, phi_xt, psi_xt)

  probr1 = exp(-node.alpha*t)
  probr2 = 1.0 - probr1

  return logsumexp(log(probr1)+translik, log(probr2)+switchll)  =#
  pi_r1 = node.pi_r1
  pi_r2 = 1.0-pi_r1
  wt = exp(-node.alpha*t)
  r1r1 = log(pi_r1*pi_r2*wt) + get_data_lik(node.aapairnode_r1, x0, xt, t) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0, phi_xt, psi_xt, t)
  r1r2 = log(pi_r1*(1.0-pi_r2*wt)) + get_data_lik(node.aapairnode_r1, x0) + get_data_lik(node.diffusion_r1, phi_x0, psi_x0) + get_data_lik(node.aapairnode_r2, xt) + get_data_lik(node.diffusion_r2, phi_xt, psi_xt)
  r2r1 = log(pi_r2*(1.0-pi_r1*wt)) + get_data_lik(node.aapairnode_r2, x0) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0) + get_data_lik(node.aapairnode_r1, xt) + get_data_lik(node.diffusion_r1, phi_xt, psi_xt)
  r2r2 = log(pi_r2*pi_r1*wt) + get_data_lik(node.aapairnode_r2, x0, xt, t) + get_data_lik(node.diffusion_r2, phi_x0, psi_x0, phi_xt, psi_xt, t)
  v = Float64[r1r1, r1r2, r2r1, r2r2]
  return logsumexp(v)

end

function sample(rng::AbstractRNG, node::SwitchingNode, x0::Int, xt::Int, phi_x0::Float64, psi_x0::Float64, phi_xt::Float64, psi_xt::Float64, t::Float64)
  wt = exp(-node.alpha*t)
  pi_r1 = node.pi_r1
  pi_r2 = 1.0-pi_r1
  v = [pi_r1*pi_r2*wt, pi_r1*(1.0-pi_r2*wt), pi_r2*(1.0-pi_r1*wt), pi_r1*pi_r2*wt]
  c = sample(rng, v)

  if c == 1
    a,b = sample(node.aapairnode_r1, rng, x0, xt, t)
    phi,psi = sample_phi_psi(node.diffusion_r1, rng, phi_x0, phi_xt, psi_x0, psi_xt, t)
    return a,b,phi,psi
  elseif c == 2
    a = sample(node.aapairnode_r1, rng, x0, 0, t)[1]
    b = sample(node.aapairnode_r2, rng, xt, 0, t)[1]
    phi = sample_phi_psi(node.diffusion_r1, rng, phi_x0, -1000.0, psi_x0, -1000.0, t)[1]
    psi = sample_phi_psi(node.diffusion_r2, rng, -1000.0, phi_xt, -1000.0, psi_xt, t)[2]
    return a,b,phi,psi
  elseif c == 3
    a = sample(node.aapairnode_r2, rng, x0, 0, t)[1]
    b = sample(node.aapairnode_r1, rng, xt, 0, t)[1]
    phi = sample_phi_psi(node.diffusion_r2, rng, phi_x0, -1000.0, psi_x0, -1000.0, t)[1]
    psi = sample_phi_psi(node.diffusion_r1, rng, -1000.0, phi_xt, -1000.0, psi_xt, t)[2]
    return a,b,phi,psi
  else
    a,b = sample(node.aapairnode_r2, rng, x0, xt, t)
    phi,psi = sample_phi_psi(node.diffusion_r2, rng, phi_x0, phi_xt, psi_x0, psi_xt, t)
    return a,b,phi,psi
  end

  #=
  probr1 = exp(-node.alpha*t)
  if rand(rng) < probr1
    a,b = sample(node.aapairnode_r1, rng, x0, xt, t)
    phi,psi = sample_phi_psi(node.diffusion_r1, rng, phi_x0, phi_xt, psi_x0, psi_xt, t)
    return a,b,phi,psi
  else
    a = sample(node.aapairnode_r1, rng, x0, 0, t)[1]
    b = sample(node.aapairnode_r2, rng, xt, 0, t)[1]
    phi = sample_phi_psi(node.diffusion_r1, rng, phi_x0, -1000.0, psi_x0, -1000.0, t)[1]
    psi = sample_phi_psi(node.diffusion_r2, rng, -1000.0, phi_xt, -1000.0, psi_xt, t)[2]
    return a,b,phi,psi
  end
  =#
end

#=
rng = MersenneTwister(1313048419911)

switching = SwitchingNode()
set_parameters(switching.diffusion_r1, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
set_parameters(switching.diffusion_r2, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)

switching.alpha = 0.0
println(sample(rng, switching, 1,2, 0.1, 0.1, -1000.0, -1000.0, 1.0))
println(sample(rng, switching, 1,2, 0.1, 0.1, -1000.0, -1000.0, 1.0))
println(sample(rng, switching, 1,2, 0.1, 0.1, -1000.0, -1000.0, 1.0))
println(sample(rng, switching, 1,2, 0.1, 0.1, -1000.0, -1000.0, 1.0))
println(sample(rng, switching, 1,2, 0.1, 0.1, -1000.0, -1000.0, 1.0))
println(sample(rng, switching, 1,2, 0.1, 0.1, -1000.0, -1000.0, 1.0))
=#
