include("AAPairNode.jl")
include("DiffusionNode.jl")
include("SwitchingNode.jl")
include("SecondaryStructureNode.jl")
include("Sequence.jl")

type ObservationNode
  aapairnode::AAPairNode
  diffusion::DiffusionNode
  ss::SecondaryStructureNode
  switching::SwitchingNode
  useswitching::Bool

  function ObservationNode()
    aapairnode = AAPairNode()
    load_parameters(aapairnode, "resources/lg_LG.PAML.txt")
    ss = SecondaryStructureNode(Float64[0.5,0.25,0.25],1.0,1.0,1.0,1.0)
    new(aapairnode, DiffusionNode(),ss, SwitchingNode(),false)
  end

  function ObservationNode(node::ObservationNode)
    new(AAPairNode(node.aapairnode), DiffusionNode(node.diffusion), SecondaryStructureNode(node.ss), SwitchingNode(node.switching), node.useswitching)
  end
end

function get_data_lik(obsnode::ObservationNode, seq1::Sequence, seq2::Sequence, i::Int, j::Int, t::Float64)
  if obsnode.useswitching
    ##obsll = logdensity(seq1.error_distribution, seq1.phi_error[i]-seq1.phi[i]) + logdensity(seq1.error_distribution, seq1.psi_error[i]-seq1.psi[i])
    ##obsll += logdensity(seq2.error_distribution, seq2.phi_error[j]-seq2.phi[j]) + logdensity(seq2.error_distribution, seq2.psi_error[j]-seq2.psi[j])
    obsll = 0.0
    #return obsll + get_data_lik(obsnode.switching, seq1.seq[i], seq2.seq[j], seq1.phi_error[i], seq1.psi_error[i], seq2.phi_error[j], seq2.psi_error[j], t)
    return obsll + get_data_lik(obsnode.switching, seq1.seq[i], seq2.seq[j], seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], t)
  else
    aapairll = get_data_lik(obsnode.aapairnode, seq1.seq[i], seq2.seq[j], t)
    #diffusionll = get_data_lik(obsnode.diffusion, seq1.phi_error[i], seq1.psi_error[i], seq2.phi_error[j], seq2.psi_error[j], t)
    diffusionll = get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], t)

    ssll = get_data_lik(obsnode.ss.ctmc, seq1.ss[i], seq2.ss[j], t)

    return aapairll + diffusionll + ssll
  end
end

function get_data_lik_x0(obsnode::ObservationNode, seq1::Sequence, i::Int, t::Float64)
  if obsnode.useswitching
    #obsll = logdensity(seq1.error_distribution, seq1.phi_error[i]-seq1.phi[i]) + logdensity(seq1.error_distribution, seq1.psi_error[i]-seq1.psi[i])
    obsll = 0.0
    #return obsll + get_data_lik_x0(obsnode.switching, seq1.seq[i], seq1.phi_error[i], seq1.psi_error[i], t)
    return obsll + get_data_lik_x0(obsnode.switching, seq1.seq[i], seq1.phi[i], seq1.psi[i], t)
  else
    aapairll =  get_data_lik(obsnode.aapairnode, seq1.seq[i])
    #diffusionll = get_data_lik(obsnode.diffusion, seq1.phi_error[i], seq1.psi_error[i])
    diffusionll = get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i])
    ssll = get_data_lik(obsnode.ss.ctmc, seq1.ss[i])
    return aapairll + diffusionll + ssll
  end
end

function get_data_lik_xt(obsnode::ObservationNode, seq1::Sequence, i::Int, t::Float64)
  if obsnode.useswitching
    #obsll = logdensity(seq1.error_distribution, seq1.phi_error[i]-seq1.phi[i]) + logdensity(seq1.error_distribution, seq1.psi_error[i]-seq1.psi[i])
    obsll = 0.0
    #return obsll + get_data_lik_xt(obsnode.switching, seq1.seq[i], seq1.phi_error[i], seq1.psi_error[i], t)
    return obsll + get_data_lik_xt(obsnode.switching, seq1.seq[i], seq1.phi[i], seq1.psi[i], t)
  else
    aapairll =  get_data_lik(obsnode.aapairnode, seq1.seq[i])
    #diffusionll = get_data_lik(obsnode.diffusion, seq1.phi_error[i], seq1.psi_error[i])
    diffusionll = get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i])
    ssll = get_data_lik(obsnode.ss.ctmc, seq1.ss[i])
    return aapairll + diffusionll + ssll
  end
end

function get_data_lik(obsnode::ObservationNode, seq1::Sequence, seq2::Sequence, i::Int, j::Int, t::Float64, nodetype::Int)
  #obsll = logdensity(seq1.error_distribution, seq1.phi_error[i]-seq1.phi[i]) + logdensity(seq1.error_distribution, seq1.psi_error[i]-seq1.psi[i])
  #obsll += logdensity(seq2.error_distribution, seq2.phi_error[j]-seq2.phi[j]) + logdensity(seq2.error_distribution, seq2.psi_error[j]-seq2.psi[j])
  obsll = 0.0
  if nodetype == 1
    return obsll+get_data_lik(obsnode.aapairnode, seq1.seq[i], seq2.seq[j], t)
  elseif nodetype == 2
    #return obsll+get_data_lik(obsnode.diffusion, seq1.phi_error[i], seq1.psi_error[i], seq2.phi_error[j], seq2.psi_error[j], t)
    return obsll+get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], t)
  else
    return obsll+get_data_lik(obsnode.ss.ctmc, seq1.ss[i], seq2.ss[j], t)
  end
end

function get_data_lik(obsnode::ObservationNode, seq1::Sequence, i::Int, nodetype::Int)
  #obsll = logdensity(seq1.error_distribution, seq1.phi_error[i]-seq1.phi[i]) + logdensity(seq1.error_distribution, seq1.psi_error[i]-seq1.psi[i])
  #obsll += logdensity(seq2.error_distribution, seq2.phi_error[j]-seq2.phi[j]) + logdensity(seq2.error_distribution, seq2.psi_error[j]-seq2.psi[j])
  obsll = 0.0
  if nodetype == 1
    return obsll+get_data_lik(obsnode.aapairnode, seq1.seq[i])
  elseif nodetype == 2
    #return obsll+get_data_lik(obsnode.diffusion, seq1.phi_error[i], seq1.psi_error[i])
    return obsll+get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i])
  else
    return obsll+get_data_lik(obsnode.ss.ctmc, seq1.ss[i])
  end
end

function sample(obsnode::ObservationNode, rng::AbstractRNG, x0::Int, xt::Int, phi_x0::Float64, phi_xt::Float64, psi_x0::Float64, psi_xt::Float64, ss0::Int, sst::Int, t::Float64)

  if obsnode.useswitching
    return sample(rng, obsnode.switching, x0, xt, phi_x0, psi_x0, phi_xt, psi_xt, t)
  else
    a, b = sample(obsnode.aapairnode, rng, x0, xt, t)
    phi,psi = sample_phi_psi(obsnode.diffusion, rng, phi_x0, phi_xt, psi_x0, psi_xt,t)
    ss1, ss2 = sample(obsnode.ctmc, rng, ss0, sst, t)
    return a, b, phi, psi, ss1, ss2
  end
end



#=
seq1 = Sequence("SLYEMAVEQFNRAASLMDLESDLAEVLRRPKRVLIVEFP")
seq2 = Sequence("SKYVDRVIAEVEKKYADEPEFVQTVEEVLSSLGPVVDAHPEYEEVALLERMVIPERVIEFRVPWED")

obsnode = ObservationNode()
println(get_data_lik(obsnode, seq1, seq2, 1,1,1.0))
println(get_data_lik(obsnode, seq1, seq2, 1,1,0.1))
println(get_data_lik(obsnode, seq1, seq2, 1,1,0.01))
println(get_data_lik(obsnode, seq1, seq2, 1,1, 10.0))
println(get_data_lik(obsnode, seq1, 1, 2.0))
=#
