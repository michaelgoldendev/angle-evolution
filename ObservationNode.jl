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
  angle_error_kappa::Float64

  function ObservationNode()
    aapairnode = AAPairNode()
    load_parameters(aapairnode, "resources/lg_LG.PAML.txt")
    ss = SecondaryStructureNode(Float64[0.5,0.25,0.25],1.0,1.0,1.0,1.0)
    new(aapairnode, DiffusionNode(),ss,false, SwitchingNode(),false, 1.0,600.0)
  end

  function ObservationNode(node::ObservationNode)
    new(AAPairNode(node.aapairnode), DiffusionNode(node.diffusion), SecondaryStructureNode(node.ss), node.usesecondarystructure, SwitchingNode(node.switching), node.useswitching, node.branch_scale, node.angle_error_kappa)
  end
end

export computeobsll
function computeobsll(current_sample::SequencePairSample, obsnodes::Array{ObservationNode,1}, a::Int)
    #=i = current_sample.align1[a]
    j = current_sample.align2[a]
    h = current_sample.states[a]
    t = current_sample.params.t
    obsnode = obsnodes[h]

    vmerror1 = current_sample.seqpair.seq1.error_distribution
    vmerror2 = current_sample.seqpair.seq2.error_distribution
    ll  = 0.0
    if i == 0
      ll += logdensity(vmerror2,current_sample.seqpair.seq2.phi[j],current_sample.seqpair.seq2.phi_obs[j],obsnode.angle_error_kappa)
      ll += logdensity(vmerror2,current_sample.seqpair.seq2.psi[j],current_sample.seqpair.seq2.psi_obs[j],obsnode.angle_error_kappa)
      ll += get_data_lik_xt(obsnode, current_sample.seqpair.seq2,j,t)
    elseif j == 0
      ll += logdensity(vmerror1,current_sample.seqpair.seq1.phi[i],current_sample.seqpair.seq1.phi_obs[i],obsnode.angle_error_kappa)
      ll += logdensity(vmerror1,current_sample.seqpair.seq1.psi[i],current_sample.seqpair.seq1.psi_obs[i],obsnode.angle_error_kappa)
      ll += get_data_lik_x0(obsnode, current_sample.seqpair.seq1,i,t)
    else
      ll += logdensity(vmerror1,current_sample.seqpair.seq1.phi[i],current_sample.seqpair.seq1.phi_obs[i],obsnode.angle_error_kappa)
      ll += logdensity(vmerror1,current_sample.seqpair.seq1.psi[i],current_sample.seqpair.seq1.psi_obs[i],obsnode.angle_error_kappa)
      ll += logdensity(vmerror2,current_sample.seqpair.seq2.phi[j],current_sample.seqpair.seq2.phi_obs[j],obsnode.angle_error_kappa)
      ll += logdensity(vmerror2,current_sample.seqpair.seq2.psi[j],current_sample.seqpair.seq2.psi_obs[j],obsnode.angle_error_kappa)
      ll += get_data_lik(obsnode, current_sample.seqpair.seq1, current_sample.seqpair.seq2, i, j, t)
    end
    return ll=#

    i = current_sample.align1[a]
    j = current_sample.align2[a]
    h = current_sample.states[a]
    t = current_sample.params.t
    obsnode = obsnodes[h]

    error1 = TDist(5)
    ll  = 0.0
    if i == 0
      d1 = (current_sample.seqpair.seq2.phi[j]-current_sample.seqpair.seq2.phi_obs[j])*(1.0/obsnode.angle_error_kappa)
      d2 = (current_sample.seqpair.seq2.psi[j]-current_sample.seqpair.seq2.psi_obs[j])*(1.0/obsnode.angle_error_kappa)
      ll += logpdf(error1, d1)+logpdf(error1,d2)
      ll += get_data_lik_xt(obsnode, current_sample.seqpair.seq2,j,t)
    elseif j == 0
      d1 = (current_sample.seqpair.seq1.phi[i]-current_sample.seqpair.seq1.phi_obs[i])*(1.0/obsnode.angle_error_kappa)
      d2 = (current_sample.seqpair.seq1.psi[i]-current_sample.seqpair.seq1.psi_obs[i])*(1.0/obsnode.angle_error_kappa)
      ll += logpdf(error1, d1)+logpdf(error1,d2)
      ll += get_data_lik_x0(obsnode, current_sample.seqpair.seq1,i,t)
    else
      d1 = (current_sample.seqpair.seq2.phi[j]-current_sample.seqpair.seq2.phi_obs[j])*(1.0/obsnode.angle_error_kappa)
      d2 = (current_sample.seqpair.seq2.psi[j]-current_sample.seqpair.seq2.psi_obs[j])*(1.0/obsnode.angle_error_kappa)
      d3 = (current_sample.seqpair.seq1.phi[i]-current_sample.seqpair.seq1.phi_obs[i])*(1.0/obsnode.angle_error_kappa)
      d4 = (current_sample.seqpair.seq1.psi[i]-current_sample.seqpair.seq1.psi_obs[i])*(1.0/obsnode.angle_error_kappa)
      ll += logpdf(error1, d1)+logpdf(error1,d2)+logpdf(error1, d3)+logpdf(error1,d4)
      ll += get_data_lik(obsnode, current_sample.seqpair.seq1, current_sample.seqpair.seq2, i, j, t)
    end
    return ll
end

export get_data_lik
function get_data_lik(obsnode::ObservationNode, seq1::Sequence, seq2::Sequence, i::Int, j::Int, t::Float64)
  if obsnode.useswitching
    return get_data_lik(obsnode.switching, seq1.seq[i], seq2.seq[j], seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], seq1.ss[i], seq2.ss[j], t)
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

function get_data_lik(obsnode::ObservationNode, seq1::Sequence, seq2::Sequence, i::Int, j::Int, t::Float64, nodetype::Int, regime::Int=1)
  if obsnode.useswitching
    if regime == 1
      if nodetype == 1
        return get_data_lik(obsnode.switching.aapairnode_r1, seq1.seq[i], seq2.seq[j], t)
      elseif nodetype == 2
        return get_data_lik(obsnode.switching.diffusion_r1, seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], t)
      else
        return get_data_lik(obsnode.switching.ss_r1.ctmc, seq1.ss[i], seq2.ss[j], t)
      end
    elseif regime == 2
      if nodetype == 1
        return get_data_lik(obsnode.switching.aapairnode_r2, seq1.seq[i], seq2.seq[j], t)
      elseif nodetype == 2
        return get_data_lik(obsnode.switching.diffusion_r2, seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], t)
      else
        return get_data_lik(obsnode.switching.ss_r2.ctmc, seq1.ss[i], seq2.ss[j], t)
      end
    end
  else
    obsll = 0.0
    if nodetype == 1
      return get_data_lik(obsnode.aapairnode, seq1.seq[i], seq2.seq[j], t)
    elseif nodetype == 2
      return get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], t)
    else
      return get_data_lik(obsnode.ss.ctmc, seq1.ss[i], seq2.ss[j], t)
    end
  end
end

function get_data_lik(obsnode::ObservationNode, seq1::Sequence, i::Int, nodetype::Int, regime::Int=1)
  if obsnode.useswitching
    if regime == 1
      if nodetype == 1
        return get_data_lik(obsnode.switching.aapairnode_r1, seq1.seq[i])
      elseif nodetype == 2
        return get_data_lik(obsnode.switching.diffusion_r1, seq1.phi[i], seq1.psi[i])
      else
        return get_data_lik(obsnode.switching.ss_r1.ctmc, seq1.ss[i])
      end
    elseif regime == 2
      if nodetype == 1
        return get_data_lik(obsnode.switching.aapairnode_r2, seq1.seq[i])
      elseif nodetype == 2
        return get_data_lik(obsnode.switching.diffusion_r2, seq1.phi[i], seq1.psi[i])
      else
        return get_data_lik(obsnode.switching.ss_r2.ctmc, seq1.ss[i])
      end
    end
  else
    if nodetype == 1
      return get_data_lik(obsnode.aapairnode, seq1.seq[i])
    elseif nodetype == 2
      return get_data_lik(obsnode.diffusion, seq1.phi[i], seq1.psi[i])
    else
      ssll = 0.0
      if obsnode.usesecondarystructure
        return get_data_lik(obsnode.ss.ctmc, seq1.ss[i])
      end
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
