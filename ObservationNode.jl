#include("AAPairNode.jl")
#include("DiffusionNode.jl")
#include("SwitchingNode.jl")
#include("SecondaryStructureNode.jl")
#include("Sequence.jl")

export ObservationNode

type ObservationNode
  aapairnode::AAPairNode
  diffusion::DiffusionNode
  ss::SecondaryStructureNode
  usesecondarystructure::Bool
  switching::SwitchingNode
  useswitching::Bool
  branch_scale::Float64

  function ObservationNode()
    aapairnode = AAPairNode()
    load_parameters(aapairnode, "resources/lg_LG.PAML.txt")
    ss = SecondaryStructureNode(Float64[0.5,0.25,0.25],1.0,1.0,1.0,1.0)
    new(aapairnode, DiffusionNode(),ss,false, SwitchingNode(),false, 1.0)
  end

  function ObservationNode(node::ObservationNode)
    new(AAPairNode(node.aapairnode), DiffusionNode(node.diffusion), SecondaryStructureNode(node.ss), node.usesecondarystructure, SwitchingNode(node.switching), node.useswitching, node.branch_scale)
  end
end

export get_data_lik
function get_data_lik(obsnode::ObservationNode, seq1::Sequence, seq2::Sequence, i::Int, j::Int, t::Float64)
  if obsnode.useswitching
    obsll = 0.0
    return obsll + get_data_lik(obsnode.switching, seq1.seq[i], seq2.seq[j], seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], seq1.ss[i], seq2.ss[j], t)
  else
    aapairll = get_data_lik(obsnode.aapairnode, seq1.seq[i], seq2.seq[j], t)
    diffusionll = get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], t)
    ssll = 0.0
    if obsnode.usesecondarystructure
      ssll = get_data_lik(obsnode.ss.ctmc, seq1.ss[i], seq2.ss[j], t)
    end
    return aapairll + diffusionll + ssll
  end
end

export get_data_lik_x0
function get_data_lik_x0(obsnode::ObservationNode, seq1::Sequence, i::Int, t::Float64)
  if obsnode.useswitching
    obsll = 0.0
    return obsll + get_data_lik_x0(obsnode.switching, seq1.seq[i], seq1.phi[i], seq1.psi[i], seq1.ss[i], t)
  else
    aapairll =  get_data_lik(obsnode.aapairnode, seq1.seq[i])
    diffusionll = get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i])
    ssll = 0.0
    if obsnode.usesecondarystructure
      ssll = get_data_lik(obsnode.ss.ctmc, seq1.ss[i])
    end
    return aapairll + diffusionll + ssll
  end
end

export get_data_lik_xt
function get_data_lik_xt(obsnode::ObservationNode, seq1::Sequence, i::Int, t::Float64)
  if obsnode.useswitching
    obsll = 0.0
    return obsll + get_data_lik_xt(obsnode.switching, seq1.seq[i], seq1.phi[i], seq1.psi[i], seq1.ss[i], t)
  else
    aapairll =  get_data_lik(obsnode.aapairnode, seq1.seq[i])
    diffusionll = get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i])
    ssll = 0.0
    if obsnode.usesecondarystructure
      ssll = get_data_lik(obsnode.ss.ctmc, seq1.ss[i])
    end
    return aapairll + diffusionll + ssll
  end
end

function get_data_lik(obsnode::ObservationNode, seq1::Sequence, seq2::Sequence, i::Int, j::Int, t::Float64, nodetype::Int)
  obsll = 0.0
  if nodetype == 1
    return obsll+get_data_lik(obsnode.aapairnode, seq1.seq[i], seq2.seq[j], t)
  elseif nodetype == 2
    return obsll+get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], t)
  else
    return obsll+get_data_lik(obsnode.ss.ctmc, seq1.ss[i], seq2.ss[j], t)
  end
end

function get_data_lik(obsnode::ObservationNode, seq1::Sequence, i::Int, nodetype::Int)
  obsll = 0.0
  if nodetype == 1
    return obsll+get_data_lik(obsnode.aapairnode, seq1.seq[i])
  elseif nodetype == 2
    return obsll+get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i])
  else
    ssll = 0.0
    if obsnode.usesecondarystructure
      return obsll+get_data_lik(obsnode.ss.ctmc, seq1.ss[i])
    end
  end
end

export sample
function sample(obsnode::ObservationNode, rng::AbstractRNG, x0::Int, xt::Int, phi_x0::Float64, phi_xt::Float64, psi_x0::Float64, psi_xt::Float64, ss0::Int, sst::Int, t::Float64)

  if obsnode.useswitching
    return sample(rng, obsnode.switching, x0, xt, phi_x0, psi_x0, phi_xt, psi_xt, ss0, sst, t)
  else
    a, b = sample(obsnode.aapairnode, rng, x0, xt, t)
    phi,psi = sample_phi_psi(obsnode.diffusion, rng, phi_x0, phi_xt, psi_x0, psi_xt, t)
    ss1 = 0
    ss2 = 0
    if obsnode.usesecondarystructure
      ss1, ss2 = sample(obsnode.ss.ctmc, rng, ss0, sst, t)
    end
    return a, b, phi, psi, ss1, ss2
  end
end
