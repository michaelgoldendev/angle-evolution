export FullNode
type FullNode
  obsnodes::Array{ObservationNode,1}
  switchingctmc::CTMC
  branchfactors::Array{Float64,1}
  numHiddenStates::Int
  localNumHiddenStates::Int
  S::Array{Float64,2}
  logfreq::Float64

  function FullNode(obsnodes::Array{ObservationNode,1}, numHiddenStates::Int)
    localNumHiddenStates = length(obsnodes)
    eqfreqs = ones(Float64, localNumHiddenStates) / Float64(localNumHiddenStates)
    S = ones(Float64, localNumHiddenStates, localNumHiddenStates)*50.0
    for h=1:localNumHiddenStates
      for k=1:localNumHiddenStates
        if k == 1
          S[h,h] = 0.0
        end
        if h != k
          S[h,h] -= S[h,k]
        end
      end
    end

    switchingctmc =  CTMC(eqfreqs, S, 1.0)

    branchfactors = ones(Float64,numHiddenStates)
    for i=2:numHiddenStates
      branchfactors[i] = 2.0^i
    end

    new(obsnodes, switchingctmc, branchfactors, numHiddenStates, localNumHiddenStates, S, log(1.0/localNumHiddenStates))
  end

  function FullNode(node::FullNode)
    new(deepcopy(node.obsnodes), deepcopy(node.switchingctmc), copy(node.branchfactors), node.numHiddenStates, node.localNumHiddenStates, copy(node.S), node.logfreq)
  end
end

export computeobsll
function computeobsll(current_sample::SequencePairSample, fullnode::FullNode, a::Int)
    i = current_sample.align1[a]
    j = current_sample.align2[a]
    ah = current_sample.states[a]
    t = current_sample.params.t
    obsnode = fullnode.obsnodes[ah]

    vmerror1 = current_sample.seqpair.seq1.error_distribution
    vmerror2 = current_sample.seqpair.seq2.error_distribution
    ll  = 0.0
    if i == 0
      ll += logdensity(vmerror2,current_sample.seqpair.seq2.phi[j],current_sample.seqpair.seq2.phi_obs[j],obsnode.angle_error_kappa)
      ll += logdensity(vmerror2,current_sample.seqpair.seq2.psi[j],current_sample.seqpair.seq2.psi_obs[j],obsnode.angle_error_kappa)
      ll += get_data_lik(fullnode, ah, current_sample.seqpair.seq2,j,t)
    elseif j == 0
      ll += logdensity(vmerror1,current_sample.seqpair.seq1.phi[i],current_sample.seqpair.seq1.phi_obs[i],obsnode.angle_error_kappa)
      ll += logdensity(vmerror1,current_sample.seqpair.seq1.psi[i],current_sample.seqpair.seq1.psi_obs[i],obsnode.angle_error_kappa)
      ll += get_data_lik(fullnode, ah, current_sample.seqpair.seq1,i,t)
    else
      ll += logdensity(vmerror1,current_sample.seqpair.seq1.phi[i],current_sample.seqpair.seq1.phi_obs[i],obsnode.angle_error_kappa)
      ll += logdensity(vmerror1,current_sample.seqpair.seq1.psi[i],current_sample.seqpair.seq1.psi_obs[i],obsnode.angle_error_kappa)
      ll += logdensity(vmerror2,current_sample.seqpair.seq2.phi[j],current_sample.seqpair.seq2.phi_obs[j],obsnode.angle_error_kappa)
      ll += logdensity(vmerror2,current_sample.seqpair.seq2.psi[j],current_sample.seqpair.seq2.psi_obs[j],obsnode.angle_error_kappa)
      ll += get_data_lik(fullnode, ah, current_sample.seqpair.seq1, current_sample.seqpair.seq2, i, j, t)
    end
    return ll
end

export get_data_lik
function get_data_lik(node::FullNode, x0::Int, phi_x0::Float64, psi_x0::Float64, ss0::Int, t::Float64, h::Int)
  #transitionll = get_data_lik(node.switchingctmc, h)
  #transitionll = 0.0
  #println(transitionll)
  transitionll = get_trans_lik(node, h)
  return transitionll+get_data_lik(node.obsnodes[h].aapairnode, x0) + get_data_lik(node.obsnodes[h].diffusion, phi_x0, psi_x0) + get_data_lik(node.obsnodes[h].ss.ctmc, ss0)
end

export get_data_lik
function get_data_lik(fullnode::FullNode, ah::Int, seq1::Sequence, i::Int, t2::Float64)
  t = fullnode.branchfactors[ah]*t2
  return  get_data_lik(fullnode, seq1.seq[i], seq1.phi[i], seq1.psi[i], seq1.ss[i], t, seq1.localstates[i])
end

export get_data_lik
function get_data_lik(node::FullNode, x0::Int, xt::Int, phi_x0::Float64, psi_x0::Float64, phi_xt::Float64, psi_xt::Float64, ss_x0::Int, ss_xt::Int, t::Float64, h0::Int, ht::Int)
   #transitionll = get_trans_lik(node.switchingctmc, h0, ht, t)

  #transitionll = get_data_lik(node.switchingctmc, h0, ht, t)
  transitionll = get_trans_lik(node, h0, ht, t)
  if h0 == ht
    return transitionll + get_data_lik(node.obsnodes[h0].aapairnode, x0, xt, t) + get_data_lik(node.obsnodes[h0].diffusion, phi_x0, psi_x0, phi_xt, psi_xt, t) + get_data_lik(node.obsnodes[h0].ss.ctmc, ss_x0, ss_xt, t)
  else
    llr1 = get_data_lik(node.obsnodes[h0].aapairnode, x0) + get_data_lik(node.obsnodes[h0].diffusion, phi_x0, psi_x0) + get_data_lik(node.obsnodes[h0].ss.ctmc, ss_x0)
    llr2 =  get_data_lik(node.obsnodes[ht].aapairnode, xt) + get_data_lik(node.obsnodes[ht].diffusion, phi_xt, psi_xt) + get_data_lik(node.obsnodes[ht].ss.ctmc, ss_xt)
    return transitionll + llr1 + llr2
  end
end



export get_data_lik
function get_data_lik(fullnode::FullNode, ah::Int, seq1::Sequence, seq2::Sequence, i::Int, j::Int, t2::Float64)
  t = fullnode.branchfactors[ah]*t2
  return get_data_lik(fullnode, seq1.seq[i], seq2.seq[j], seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], seq1.ss[i], seq2.ss[j], t, seq1.localstates[i], seq2.localstates[j])
end

function get_data_lik(node::FullNode, ah::Int, seq1::Sequence, i::Int, t2::Float64, nodetype::Int)
  t = node.branchfactors[ah]*t2

  h = seq1.localstates[i]
  if nodetype == 1
    return get_data_lik(node.obsnodes[h].aapairnode, seq1.seq[i])
  elseif nodetype == 2
    return get_data_lik(node.obsnodes[h].diffusion, seq1.phi[i], seq1.psi[i])
  else
    if node.obsnodes[h].usesecondarystructure
      return get_data_lik(node.obsnodes[h].ss.ctmc, seq1.ss[i])
    end
  end

  return 0.0
end

function get_data_lik(node::FullNode, ah::Int, seq1::Sequence, seq2::Sequence, i::Int, j::Int, t2::Float64, nodetype::Int)
  t = node.branchfactors[ah]*t2

  h0 = seq1.localstates[i]
  ht = seq2.localstates[j]
  if h0 == ht
      if nodetype == 1
        return get_data_lik(node.obsnodes[h0].aapairnode, seq1.seq[i], seq2.seq[j], t)
      elseif nodetype == 2
        return get_data_lik(node.obsnodes[h0].diffusion, seq1.phi[i], seq1.psi[i], seq2.phi[j], seq2.psi[j], t)
      else
        return get_data_lik(node.obsnodes[h0].ss.ctmc, seq1.ss[i], seq2.ss[j], t)
      end
  else
    if nodetype == 1
      llr1 = get_data_lik(node.obsnodes[h0].aapairnode, seq1.seq[i])
      llr2 =  get_data_lik(node.obsnodes[ht].aapairnode, seq2.seq[j])
      return llr1 + llr2
    elseif nodetype == 2
      llr1 = get_data_lik(node.obsnodes[h0].diffusion, seq1.phi[i], seq1.psi[i])
      llr2 = get_data_lik(node.obsnodes[ht].diffusion, seq2.phi[j], seq2.psi[j])
      return llr1 + llr2
    else
      llr1 = get_data_lik(node.obsnodes[h0].ss.ctmc, seq1.ss[i])
      llr2 = get_data_lik(node.obsnodes[ht].ss.ctmc, seq2.ss[j])
      return ll1r1 + llr2
    end
  end
end


export get_trans_lik
function get_trans_lik(fullnode::FullNode, x0::Int)
  #=node = fullnode.switchingctmc
  if !node.enabled
    return 0.0
  end

  if x0 > 0
    return node.logeqfreqs[x0]
  else
    return 0.0
  end=#

  if x0 > 0
    return fullnode.logfreq
  else
    println("NOTHING")
  end
end


export get_trans_lik
function get_trans_lik(fullnode::FullNode, x0::Int, xt::Int, t::Float64)
  #=
  node = fullnode.switchingctmc
  if !node.enabled
    return 0.0
  end

  if x0 > 0 && xt > 0
    set_parameters(node, t)
    return node.logeqfreqs[x0] + node.logPt[x0,xt]
  end

  return 0.0=#

  if x0 > 0 && xt > 0
    if x0 == xt
      return fullnode.logfreq+fullnode.S[x0,x0]*t
      #return log1p(-exp(fullnode.S[x0,x0]*t))
    else
      return fullnode.logfreq+log1p(-exp(fullnode.S[x0,x0]*t)) + log(fullnode.S[x0,xt]/(-fullnode.S[x0,x0]))
      #return fullnode.S[x0,x0]*t + log(fullnode.S[x0,xt]/(-fullnode.S[x0,x0]))
    end
  else
    println("EHERAA")
  end
  return 0.0
end




export sample
function sample(rng::AbstractRNG, node::FullNode, ah::Int, x0::Int, xt::Int, phi_x0::Float64, psi_x0::Float64, phi_xt::Float64, psi_xt::Float64, ss_x0::Int, ss_xt::Int, t2::Float64, h0::Int, ht::Int)
  t = node.branchfactors[ah]*t2

  if h0 == ht
    a,b = sample(node.obsnodes[h0].aapairnode, rng, x0, xt, t)
    phi,psi = sample_phi_psi(node.obsnodes[h0].diffusion, rng, phi_x0, phi_xt, psi_x0, psi_xt, t)
    c,d = sample(node.obsnodes[h0].ss.ctmc, rng, ss_x0, ss_xt, t)
    return (a,b,phi,psi,c,d)
  else
    a = sample(node.obsnodes[h0].aapairnode, rng, x0, 0, t)[1]
    b = sample(node.obsnodes[ht].aapairnode, rng, xt, 0, t)[1]
    phi1, psi1 = sample_phi_psi(node.obsnodes[h0].diffusion, rng, phi_x0, -1000.0, psi_x0, -1000.0, t)
    phi2, psi2 = sample_phi_psi(node.obsnodes[ht].diffusion, rng, -1000.0, phi_xt, -1000.0, psi_xt, t)
    phi = (phi1[1],phi2[2])
    psi = (psi1[1],psi2[2])
    c = sample(node.obsnodes[h0].ss.ctmc, rng, ss_x0, 0, t)[1]
    d = sample(node.obsnodes[ht].ss.ctmc, rng, ss_xt, 0, t)[1]
    return (a,b,phi,psi,c,d)
  end
end

export set_parameters_S
function set_parameters_S(node::FullNode, x::Array{Float64,1})
  index = 1
  len = size(node.S, 1)
  for i=1:len
    node.S[i,i] = 0.0
     for j=1:len
        if j > i
          node.S[i,j] = x[index]
          node.S[j,i] = x[index]
          index += 1
        end
    end
  end

  for i=1:len
     for j=1:len
      if i != j
        node.S[i,i] -= node.S[i,j]
      end
    end
  end
end
