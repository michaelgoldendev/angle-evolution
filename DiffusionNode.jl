include("VonMisesDensity.jl")

include("WrappedNormalOUNode.jl")
#using WrappedNormalOU

export DiffusionNode
type DiffusionNode
  vm_phi::VonMisesDensity
  vm_psi::VonMisesDensity
  vm_phi_stat::VonMisesDensity
  vm_psi_stat::VonMisesDensity
  ounode::WrappedNormalOUNode

  alpha_phi::Float64
  mu_phi::Float64
  sigma_phi::Float64
  alpha_psi::Float64
  mu_psi::Float64
  sigma_psi::Float64
  alpha_rho::Float64


  t::Float64
  branch_scale::Float64
  useou::Bool

  function DiffusionNode()
    vm_phi = VonMisesDensity()
    vm_psi = VonMisesDensity()
    vm_phi_stat = VonMisesDensity()
    vm_psi_stat = VonMisesDensity()
    ounode = WrappedNormalOUNode(1)
    new(vm_phi, vm_psi, vm_phi_stat, vm_psi_stat, ounode, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,true)
  end

  function DiffusionNode(node::DiffusionNode)
    ounode = WrappedNormalOUNode(node.ounode)

    #set_parameters(ounode, node.mu, node.alpha, node.sigma)
    #set_parameters(ounode, 1.0)
    new(VonMisesDensity(node.vm_phi), VonMisesDensity(node.vm_psi), VonMisesDensity(node.vm_phi_stat), VonMisesDensity(node.vm_psi_stat), ounode, node.alpha_phi, node.mu_phi, node.sigma_phi, node.alpha_psi, node.mu_psi, node.sigma_psi, node.alpha_rho, node.t, node.branch_scale, node.useou)

  end
end

export get_parameters
function get_parameters(node::DiffusionNode)
  return Float64[node.alpha_phi, node.mu_phi, node.sigma_phi, node.alpha_psi, node.mu_psi, node.sigma_psi, node.alpha_rho, node.branch_scale]
end

export set_parameters
function set_parameters(node::DiffusionNode, x::Array{Float64,1})
  set_parameters(node,x[1], x[2], x[3], x[4],x[5], x[6], x[7], 1.0, x[8])
  #return Float64[node.alpha_phi, node.mu_phi, node.sigma_phi, node.alpha_psi, node.mu_psi, node.sigma_psi, node.alpha_rho, node.branch_scale]
end

export set_parameters
function set_parameters(node::DiffusionNode, alpha_phi::Float64, mu_phi::Float64, sigma_phi::Float64, alpha_psi::Float64, mu_psi::Float64, sigma_psi::Float64, alpha_rho::Float64, t::Float64, branch_scale::Float64=1.0)
  node.alpha_phi = alpha_phi
  node.mu_phi =  mod2pi(mu_phi+pi)-pi
  node.sigma_phi = sigma_phi
  node.alpha_psi = alpha_psi
  node.mu_psi = mod2pi(mu_psi+pi)-pi
  node.sigma_psi = sigma_psi
  node.alpha_rho = alpha_rho
  node.t = t
  node.branch_scale = 1.0
  set_parameters(node.vm_phi_stat, node.mu_phi, min(700.0, 2.0*node.alpha_phi/(node.sigma_phi*node.sigma_phi)))
  set_parameters(node.vm_psi_stat, node.mu_psi, min(700.0, 2.0*node.alpha_psi/(node.sigma_psi*node.sigma_psi)))
  set_parameters(node.ounode, 1.0, node.mu_phi,  node.mu_psi, alpha_phi, alpha_psi, alpha_rho, sigma_phi, sigma_psi)
end

export get_data_lik_phi
function get_data_lik_phi(node::DiffusionNode, phi_x0::Float64)
  phi_ll = 0.0
  if phi_x0 > -100.0
    phi_ll = logdensity(node.vm_phi_stat, phi_x0)
  end
  return phi_ll
end

export get_data_lik_psi
function get_data_lik_psi(node::DiffusionNode, psi_x0::Float64)
  psi_ll = 0.0
  if psi_x0 > -100.0
    psi_ll = logdensity(node.vm_psi_stat, psi_x0)
  end
  return psi_ll
end

export get_data_lik
function get_data_lik(node::DiffusionNode, phi_x0::Float64, psi_x0::Float64)
  if node.useou
    ll = 0.0
    if phi_x0 > -100.0 && psi_x0 > -100.0
      ll = loglikwndstat(node.ounode, phi_x0, psi_x0)
    end
    return ll
  else
    return get_data_lik_phi(node, phi_x0) + get_data_lik_psi(node, psi_x0)
  end
end

function get_data_lik(node::DiffusionNode, phi_x0::Float64, psi_x0::Float64, phi_xt::Float64, psi_xt::Float64, t2::Float64)

  t = t2*node.branch_scale
  if node.useou
    ll = 0.0
    if phi_x0 > -100.0 && phi_xt > -100.0 && psi_x0 > -100.0 && psi_xt > -100.0
      set_parameters(node.ounode, t)
      ll = loglikwndtpd(node.ounode, phi_x0, psi_x0, phi_xt, psi_xt)
    elseif phi_x0 > -100.0 && psi_x0 > -100.0
      ll = loglikwndstat(node.ounode, phi_x0, psi_x0)
    elseif phi_xt > -100.0 && psi_xt > -100.0
      ll = loglikwndstat(node.ounode, phi_xt, psi_xt)
    end
    return ll
  else
    phi_ll = 0.0
    if phi_x0 > -100.0 && phi_xt > -100.0
      mut_phi::Float64 = node.mu_phi + 2.0*atan(tan((phi_x0-node.mu_phi)/2.0)*exp(-node.alpha_phi*t))
      kt_phi::Float64  = (2.0*node.alpha_phi)/(node.sigma_phi*node.sigma_phi*(1.0-exp(-2.0*node.alpha_phi*t)))
      phi_ll = get_data_lik_phi(node, phi_x0) + logdensity(node.vm_phi, phi_xt, mut_phi, min(700.0, kt_phi))
    elseif phi_x0 > -100.0
      phi_ll = get_data_lik_phi(node, phi_x0)
    elseif phi_xt > -100.0
      phi_ll = get_data_lik_phi(node, phi_xt)
    end

    psi_ll = 0.0
    if psi_x0 > -100.0 && psi_xt > -100.0
      mut_psi::Float64 = node.mu_psi + 2.0*atan(tan((psi_x0-node.mu_psi)/2.0)*exp(-node.alpha_psi*t))
      kt_psi::Float64  = (2.0*node.alpha_psi)/(node.sigma_psi*node.sigma_psi*(1.0-exp(-2.0*node.alpha_psi*t)))
      psi_ll = get_data_lik_psi(node, psi_x0) + logdensity(node.vm_psi, psi_xt, mut_psi, min(700.0, kt_psi ))
    elseif psi_x0 > -100.0
      psi_ll = get_data_lik_psi(node, psi_x0)
    elseif psi_xt > -100.0
      psi_ll = get_data_lik_psi(node, psi_xt)
    end

    return phi_ll + psi_ll
  end
end


export sample_phi_psi
function sample_phi_psi(node::DiffusionNode, rng::AbstractRNG, phi_x0::Float64, phi_xt::Float64, psi_x0::Float64, psi_xt::Float64, t2::Float64)
  t = t2*node.branch_scale

  if node.useou
    if phi_x0 < -100.0 && phi_xt < -100.0 && psi_x0 < -100.0 && psi_xt < -100.0
      a = sampstat(node.ounode,rng)
      x0 = Float64[a[1], a[2]]
      xt = samptrans(node.ounode, rng, x0, t)
      return (x0[1],xt[1]), (x0[2], xt[2])
    end

    if phi_x0 >= -100.0 && phi_xt >= -100.0 && psi_x0 >= -100.0 && psi_xt >= -100.0
      return (phi_x0, phi_xt), (psi_x0, psi_xt)
    end

    if phi_x0 >= -100.0 && phi_xt < -100.0 && psi_x0 >= -100.0 && psi_xt < -100.0
      x0 = Float64[phi_x0, psi_x0]
      xt = samptrans(node.ounode, rng, x0, t)
      return (phi_x0, xt[1]), (psi_x0, xt[2])
    end

    if phi_x0 < -100.0 && phi_xt >= -100.0 && psi_x0 < -100.0 && psi_xt >= -100.0
      x0 = Float64[phi_xt, psi_xt]
      xt = samptrans(node.ounode, rng, x0, t)
      return (xt[1], phi_xt), (xt[2], psi_xt)
    end

    return (phi_x0, phi_xt), (psi_x0, psi_xt)
  else
    phi = sample(rng, node.alpha_phi, node.mu_phi, node.sigma_phi, phi_x0, phi_xt, t)
    psi = sample(rng, node.alpha_psi, node.mu_psi, node.sigma_psi, psi_x0, psi_xt, t)

    return phi, psi
  end

end


export sample
function sample(rng::AbstractRNG, alpha::Float64, mu::Float64, sigma::Float64, x0::Float64, xt::Float64, t::Float64)
  a = x0
  b = xt
  if a >= -100.0 && b >= -100.0
    return a,b
  end

  if a < -100.0 && b < -100.0
    vm = VonMisesDensity(mu, min(700.0, (2.0*alpha)/(sigma*sigma)))
    a = sample(vm, rng)
  end

  swap = false
  if a < -100.0
    swap = true
    a, b = b, a
  end

  mut::Float64 = mu + 2.0*atan(tan((a-mu)/2.0)*exp(-alpha*t))
  kt::Float64 = (2.0*alpha)/(sigma*sigma*(1.0-exp(-2.0*alpha*t)))
  vm = VonMisesDensity(mut, min(700.0, kt))
  b = sample(vm, rng)

  if swap
    return b,a
  else
    return a,b
  end
end


#=
rng=MersenneTwister(138411101308)

println(sample(rng, 100.0, 1.0, 1.0, 0.0, -1000.0, 1.0))
println(sample(rng, 1.0, 0.0, 1.0, 0.0, -1000.0, 1.0))
println(sample(rng, 1.0, 0.0, 1.0, 0.0, -1000.0, 1.0))
println(sample(rng, 1.0, 0.0, 1.0, 0.0, -1000.0, 1.0))
println(sample(rng, 1.0, 0.0, 1.0, 0.0, -1000.0, 1.0))
node =  DiffusionNode()
println(sample_phi_psi(node, rng, 0.0, 0.0, -1000.0, -1000.0, 0.01))
println(sample_phi_psi(node, rng, 0.0, 0.0, -1000.0, -1000.0, 0.1))
println(sample_phi_psi(node, rng, 0.0, 0.0, -1000.0, -1000.0, 0.1))
println(sample_phi_psi(node, rng, 0.0, 0.0, -1000.0, -1000.0, 1.0))
println(sample_phi_psi(node, rng, 0.0, 0.0, -1000.0, -1000.0, 1.0))

node =  DiffusionNode()
node.alpha_rho = 0.0
println("BB",get_data_lik(node, 1.0, 1.0, 1.0, 1.0, 1.0))
=#
