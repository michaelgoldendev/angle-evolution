using NLopt

function switchll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1}, store::Array{Float64, 1})
  aapairnode_r1_eqfreqs = x[1:20]/sum(x[1:20])
  if(!(0.999 < sum(aapairnode_r1_eqfreqs) < 1.001))
    aapairnode_r1_eqfreqs = ones(Float64, 20)*0.05
  end

  aapairnode_r2_eqfreqs = x[21:40]/sum(x[21:40])
  if(!(0.999 < sum(aapairnode_r2_eqfreqs) < 1.001))
    aapairnode_r2_eqfreqs = ones(Float64, 20)*0.05
  end

  set_parameters(obsnodes[h].switching.aapairnode_r1, aapairnode_r1_eqfreqs, 1.0)
  set_parameters(obsnodes[h].switching.aapairnode_r2, aapairnode_r2_eqfreqs, 1.0)

  d1 = x[41:46]
  set_parameters(obsnodes[h].switching.diffusion_r1, d1[1], mod2pi(d1[2]+pi)-pi, d1[3], d1[4], mod2pi(d1[5]+pi)-pi, d1[6], 1.0)
  d2 = x[47:52]
  set_parameters(obsnodes[h].switching.diffusion_r2, d2[1], mod2pi(d2[2]+pi)-pi, d2[3], d2[4], mod2pi(d2[5]+pi)-pi, d2[6], 1.0)

  obsnodes[h].switching.alpha = x[53]
  obsnodes[h].switching.pi_r1 = x[54]

  # dirichlet prior
  concentration_param = 1.025
  ll = sum((concentration_param-1.0)*log(aapairnode_r1_eqfreqs))
  ll += sum((concentration_param-1.0)*log(aapairnode_r2_eqfreqs))

  for (s,a) in zip(seqindices,hindices)
    sample = samples[s]
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    i = align1[a]
    j = align2[a]
    t = sample.params.t
    if i == 0
      ll += get_data_lik_xt(obsnodes[h], seqpair.seq2,j,t)
    elseif j == 0
      ll += get_data_lik_x0(obsnodes[h], seqpair.seq1,i,t)
    else
      ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t)
    end
  end

  if ll > store[1]
    store[1] = ll
    for i=1:54
      store[i+1]  = x[i]
    end
  end

  return ll
end

function switchopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  seqindices,hindices = getindices(samples, h)
  store = ones(Float64,55)*(-1e20)
  localObjectiveFunction = ((param, grad) -> switchll(param, h, samples,seqindices,hindices, obsnodes, store))
  opt = Opt(:LN_COBYLA, 54)
  lower = zeros(Float64, 54)
  lower[41] = 1e-5
  lower[42] = -1000000.0
  lower[43] = 1e-5
  lower[44] = 1e-5
  lower[45] = -1000000.0
  lower[46] = 1e-5

  lower[47] = 1e-5
  lower[48] = -1000000.0
  lower[49] = 1e-5
  lower[50] = 1e-5
  lower[51] = -1000000.0
  lower[52] = 1e-5

  lower[53] = 1e-3
  lower[54] = 0.0
  lower_bounds!(opt, lower)

  upper = ones(Float64, 54)
  upper[41] = 1e5
  upper[42] = 1000000.0
  upper[43] = 1e5
  upper[44] = 1e5
  upper[45] = 1000000.0
  upper[46] = 1e5

  upper[47] = 1e5
  upper[48] = 1000000.0
  upper[49] = 1e5
  upper[50] = 1e5
  upper[51] = 1000000.0
  upper[52] = 1e5

  upper[53] = 1e3
  upper[54] = 1.0
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 500)
  max_objective!(opt, localObjectiveFunction)
  initial = zeros(Float64,54)
  for i=1:20
    initial[i] = obsnodes[h].switching.aapairnode_r1.eqfreqs[i]
    initial[20+i] = obsnodes[h].switching.aapairnode_r2.eqfreqs[i]
  end
  initial[41] = obsnodes[h].switching.diffusion_r1.alpha_phi
  initial[42] = obsnodes[h].switching.diffusion_r1.mu_phi
  initial[43] = obsnodes[h].switching.diffusion_r1.sigma_phi
  initial[44] = obsnodes[h].switching.diffusion_r1.alpha_psi
  initial[45] = obsnodes[h].switching.diffusion_r1.mu_psi
  initial[46] = obsnodes[h].switching.diffusion_r1.sigma_psi
  initial[47] = obsnodes[h].switching.diffusion_r2.alpha_phi
  initial[48] = obsnodes[h].switching.diffusion_r2.mu_phi
  initial[49] = obsnodes[h].switching.diffusion_r2.sigma_phi
  initial[50] = obsnodes[h].switching.diffusion_r2.alpha_psi
  initial[51] = obsnodes[h].switching.diffusion_r2.mu_psi
  initial[52] = obsnodes[h].switching.diffusion_r2.sigma_psi
  initial[53] = obsnodes[h].switching.alpha
  initial[54] = obsnodes[h].switching.pi_r1

  (minf,minx,ret) = optimize(opt, initial)
  optx = store[2:55]


  set_parameters(obsnodes[h].switching.aapairnode_r1, optx[1:20]/sum(optx[1:20]), 1.0)
  set_parameters(obsnodes[h].switching.aapairnode_r2, optx[21:40]/sum(optx[21:40]), 1.0)
  set_parameters(obsnodes[h].switching.diffusion_r1, optx[41], mod2pi(optx[42]+pi)-pi, optx[43], optx[44], mod2pi(optx[45]+pi)-pi, optx[46], 1.0)
  set_parameters(obsnodes[h].switching.diffusion_r2, optx[47], mod2pi(optx[48]+pi)-pi, optx[49], optx[50], mod2pi(optx[51]+pi)-pi, optx[52], 1.0)
  obsnodes[h].switching.alpha = optx[53]
  obsnodes[h].switching.pi_r1 = optx[54]

  return optx
end

function switchllswitchingparams(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1})

  obsnodes[h].switching.alpha = x[1]
  obsnodes[h].switching.pi_r1 = x[2]
  ll = 0.0
  for (s,a) in zip(seqindices,hindices)
    sample = samples[s]
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    i = align1[a]
    j = align2[a]
    t = sample.params.t
    if i == 0
      ll += get_data_lik_xt(obsnodes[h], seqpair.seq2,j,t)
    elseif j == 0
      ll += get_data_lik_x0(obsnodes[h], seqpair.seq1,i,t)
    else
      ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t)
    end
  end

  return ll
end

function switchoptswitchingparams(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  seqindices,hindices = getindices(samples, h)
  store = ones(Float64,3)*(-1e20)
  localObjectiveFunction = ((param, grad) -> switchllswitchingparams(param, h, samples,seqindices,hindices, obsnodes))
  opt = Opt(:LN_COBYLA, 2)
  lower = zeros(Float64, 2)
  lower[1] = 1e-3
  lower[2] = 0.0
  lower_bounds!(opt, lower)

  upper = ones(Float64, 2)
  upper[1] = 1e3
  upper[2] = 1.0
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 80)
  max_objective!(opt, localObjectiveFunction)
  initial = zeros(Float64,2)
  initial[1] = obsnodes[h].switching.alpha
  initial[2] = obsnodes[h].switching.pi_r1

  (minf,minx,ret) = optimize(opt, initial)
  obsnodes[h].switching.alpha = minx[1]
  obsnodes[h].switching.pi_r1 = minx[2]

  return minx
end


function aapairll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1})
  neweqfreqs = x/sum(x)
  if(!(0.999 < sum(neweqfreqs) < 1.001))
    neweqfreqs = ones(Float64, 20)*0.05
  end

  set_parameters(obsnodes[h].aapairnode, neweqfreqs, 1.0)

  # dirichlet prior
  concentration_param = 1.025
  ll = sum((concentration_param-1.0)*log(neweqfreqs))

  for (s,a) in zip(seqindices,hindices)
    sample = samples[s]
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    i = align1[a]
    j = align2[a]
    t = sample.params.t
    if i == 0
      ll += get_data_lik(obsnodes[h], seqpair.seq2,j,1)
    elseif j == 0
      ll += get_data_lik(obsnodes[h], seqpair.seq1,i,1)
    else
      ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,1)
    end
  end

  return ll
end

function getindices(samples::Array{SequencePairSample,1}, h::Int)
  seqindices = Int[]
  hindices = Int[]
  nsamples = length(samples)
  for i=1:nsamples
    sample = samples[i]
    slen = length(sample.states)
    for j=1:slen
      if sample.states[j] == h
          push!(seqindices, i)
          push!(hindices, j)
      end
    end
  end

  return seqindices,hindices
end


function aapairopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  seqindices,hindices = getindices(samples, h)
  localObjectiveFunction = ((param, grad) -> aapairll(param, h, samples, seqindices ,hindices, obsnodes))
  opt = Opt(:LN_COBYLA, 20)
  lower_bounds!(opt, zeros(Float64, 20))
  upper_bounds!(opt, ones(Float64, 20))
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 200)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, obsnodes[h].aapairnode.eqfreqs)
  set_parameters(obsnodes[h].aapairnode, minx/sum(minx), 1.0)
  return minx/sum(minx)
end

function diffusionll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1}, store::Array{Float64, 1})
  #println("XX", x)
  set_parameters(obsnodes[h].diffusion, x[1], mod2pi(x[2]+pi)-pi, x[3], x[4], mod2pi(x[5]+pi)-pi, x[6], 1.0)

  ll = 0.0

  for (s,a) in zip(seqindices,hindices)
    sample = samples[s]
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2

    i = align1[a]
    j = align2[a]
    t = sample.params.t
    if i == 0
      ll += get_data_lik(obsnodes[h], seqpair.seq2,j, 2)
    elseif j == 0
      ll += get_data_lik(obsnodes[h], seqpair.seq1,i, 2)
    else
      ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t, 2)
    end
  end
  if ll > store[1]
    store[1] = ll
    for i=1:6
      store[i+1]  = x[i]
    end
  end

  return ll
end

function diffusionopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  seqindices,hindices = getindices(samples, h)
  store = ones(Float64,7)*(-1e20)
  localObjectiveFunction = ((param, grad) -> diffusionll(param, h, samples,seqindices,hindices, obsnodes, store))
  opt = Opt(:LN_COBYLA, 6)
  lower = zeros(Float64, 6)
  lower[1] = 1e-5
  lower[2] = -1000000.0
  lower[3] = 1e-5
  lower[4] = 1e-5
  lower[5] = -1000000.0
  lower[6] = 1e-5
  lower_bounds!(opt, lower)

  upper = ones(Float64, 6)
  upper[1] = 1e5
  upper[2] = 1000000.0
  upper[3] = 1e5
  upper[4] = 1e5
  upper[5] = 1000000.0
  upper[6] = 1e5
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 100)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, get_parameters(obsnodes[h].diffusion))
  optx = store[2:7]
  #println(optx)

  set_parameters(obsnodes[h].diffusion, optx[1], mod2pi(optx[2]+pi)-pi, optx[3], optx[4], mod2pi(optx[5]+pi)-pi, optx[6], 1.0)

  return optx
end

function mlopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
 aares = aapairopt(h, samples,obsnodes)
 diffusionres = diffusionopt(h, samples, obsnodes)
 return aares, diffusionres
end

function hmmopt(samples::Array{SequencePairSample,1}, numHiddenStates::Int)
  hmminitprobs = ones(Float64, numHiddenStates)*1e-4
  hmmtransprobs = ones(Float64, numHiddenStates, numHiddenStates)*1e-2
  for sample in samples
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    states = sample.states
    hmminitprobs[states[1]] += 1
    for j=2:length(states)
      hmmtransprobs[states[j-1],states[j]] += 1
    end
  end
  hmminitprobs /= sum(hmminitprobs)
  for i=1:numHiddenStates
    s = 0.0
    for j=1:numHiddenStates
          s += hmmtransprobs[i,j]
    end
    for j=1:numHiddenStates
          hmmtransprobs[i,j] /=   s
    end
  end

  return hmminitprobs,hmmtransprobs
end

function prioropt(samples::Array{SequencePairSample,1}, prior::PriorDistribution)
  localObjectiveFunction = ((param, grad) -> logprior(PriorDistribution(param), samples))
  opt = Opt(:LN_COBYLA, 8)
  lower_bounds!(opt, ones(Float64, 8)*1e-10)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 800)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, ones(Float64, 8))
  return PriorDistribution(minx)
end
