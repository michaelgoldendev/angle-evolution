using NLopt

freqprior = Beta(1.5, 1.5)
alpha_prior = 0.01
#rho_prior = Beta(5.0, 5.0)
function switchll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1}, store::Array{Float64, 1})
  set_parameters(obsnodes[h].switching, x)

  # dirichlet prior
  concentration_param = 1.025
  ll = sum((concentration_param-1.0)*log(obsnodes[h].switching.aapairnode_r1.eqfreqs))
  ll += sum((concentration_param-1.0)*log(obsnodes[h].switching.aapairnode_r2.eqfreqs))
  ll += logpdf(freqprior, obsnodes[h].switching.pi_r1)
  ll += -alpha_prior*obsnodes[h].switching.alpha
  #ll += logpdf(rho_prior, obsnodes[h].switching.diffusion_r1.alpha_rho/2.0 + 0.5)
  #ll += logpdf(rho_prior, obsnodes[h].switching.diffusion_r2.alpha_rho/2.0 + 0.5)

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
    for i=1:62
      store[i+1]  = x[i]
    end
  end

  return ll
end

export switchopt
function switchopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  seqindices,hindices = getindices(samples, h)
  store = ones(Float64,63)*(-1e20)
  localObjectiveFunction = ((param, grad) -> switchll(param, h, samples,seqindices,hindices, obsnodes, store))
  opt = Opt(:LN_COBYLA, 62)
  lower = zeros(Float64, 62)
  for i=1:40
    lower[i] = 1e-10
  end
  lower[41] = 1e-5
  lower[42] = -1000000.0
  lower[43] = 1e-5
  lower[44] = 1e-5
  lower[45] = -1000000.0
  lower[46] = 1e-5
  lower[47] = -100.0

  lower[48] = 1e-5
  lower[49] = -1000000.0
  lower[50] = 1e-5
  lower[51] = 1e-5
  lower[52] = -1000000.0
  lower[53] = 1e-5
  lower[54] = -100.0

  lower[55] = 1e-3
  lower[56] = 0.0
  for i=57:62
    lower[i] = 1e-10
  end
  lower_bounds!(opt, lower)

  upper = ones(Float64, 62)
  upper[41] = 1e5
  upper[42] = 1000000.0
  upper[43] = 1e5
  upper[44] = 1e5
  upper[45] = 1000000.0
  upper[46] = 1e5
  upper[47] = 100.0

  upper[48] = 1e5
  upper[49] = 1000000.0
  upper[50] = 1e5
  upper[51] = 1e5
  upper[52] = 1000000.0
  upper[53] = 1e5
  upper[54] = 100.0

  upper[55] = 1e3
  upper[56] = 1.0
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 600)
  max_objective!(opt, localObjectiveFunction)
  initial = get_parameters(obsnodes[h].switching)
  (minf,minx,ret) = optimize(opt, initial)
  optx = store[2:63]

  set_parameters(obsnodes[h].switching, optx)
  return optx
end

function ssll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1})
  neweqfreqs = x/sum(x)
  if(!(0.999 < sum(neweqfreqs) < 1.001))
    neweqfreqs = ones(Float64, 3)/3.0
  end

  set_parameters(obsnodes[h].ss, neweqfreqs,1.0)

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
      ll += get_data_lik(obsnodes[h], seqpair.seq2,j,3)
    elseif j == 0
      ll += get_data_lik(obsnodes[h], seqpair.seq1,i,3)
    else
      ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,3)
    end
  end

  return ll
end

function ssopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  seqindices,hindices = getindices(samples, h)
  localObjectiveFunction = ((param, grad) -> ssll(param, h, samples, seqindices ,hindices, obsnodes))
  opt = Opt(:LN_COBYLA, 3)
  lower = ones(Float64, 3)*1e-8
  lower_bounds!(opt, lower)
  upper = ones(Float64, 3)
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 200)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, Float64[obsnodes[h].ss.ctmc.eqfreqs[1], obsnodes[h].ss.ctmc.eqfreqs[2], obsnodes[h].ss.ctmc.eqfreqs[3]])
  outfreqs = minx[1:3]

  set_parameters(obsnodes[h].ss, outfreqs/sum(outfreqs), 1.0)
  return minx
end



function ssll_all(a::Float64, b::Float64, c::Float64, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  for obsnode in obsnodes
      set_parameters(obsnode.ss, a, b, c, 1.0)
      set_parameters(obsnode.switching.ss_r1, a, b, c, 1.0)
      set_parameters(obsnode.switching.ss_r2, a, b, c, 1.0)
  end

  ll = 0.0
  for sample in samples
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    t = sample.params.t
    for a=1:length(align1)
      i = align1[a]
      j = align2[a]
      h = sample.states[a]
      if i == 0
        #ll += get_data_lik(obsnodes[h], seqpair.seq2,j,3
        ll += get_data_lik_x0(obsnodes[h], seqpair.seq2,j, t)
      elseif j == 0
        ll += get_data_lik_xt(obsnodes[h], seqpair.seq1,i, t)
        #ll += get_data_lik(obsnodes[h], seqpair.seq1,i,3)
      else
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t)
        #ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,3)
      end
    end
  end

  return ll
end

export ssoptrates
function ssoptrates(samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  localObjectiveFunction = ((param, grad) -> ssll_all(param[1], param[2], param[3], samples, obsnodes))
  opt = Opt(:LN_COBYLA, 3)
  lower = ones(Float64, 3)*1e-5
  lower_bounds!(opt, lower)
  upper = ones(Float64, 3)*200.0
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 100)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, Float64[obsnodes[1].ss.ctmc.S[1,2], obsnodes[1].ss.ctmc.S[1,3], obsnodes[1].ss.ctmc.S[2,3]])

  for obsnode in obsnodes
      set_parameters(obsnode.ss, minx[1], minx[2], minx[3], 1.0)
      set_parameters(obsnode.switching.ss_r1, minx[1], minx[2], minx[3], 1.0)
      set_parameters(obsnode.switching.ss_r2, minx[1], minx[2], minx[3], 1.0)
  end

  return minx
end


function diffusion_all(branch_scale::Float64, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  for obsnode in obsnodes
      obsnode.diffusion.branch_scale = branch_scale
      obsnode.switching.diffusion_r1.branch_scale = branch_scale
      obsnode.switching.diffusion_r2.branch_scale = branch_scale
  end

  ll = 0.0
  for sample in samples
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    t = sample.params.t
    for a=1:length(align1)
      i = align1[a]
      j = align2[a]
      h = sample.states[a]
      if i == 0
        ll += get_data_lik_x0(obsnodes[h], seqpair.seq2,j, t)
      elseif j == 0
        ll += get_data_lik_xt(obsnodes[h], seqpair.seq1,i,t)
      else
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t)
      end
    end
  end

  return ll
end

export optimize_diffusion_branch_scale
function optimize_diffusion_branch_scale(samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  localObjectiveFunction = ((param, grad) -> diffusion_all(param[1], samples, obsnodes))
  opt = Opt(:LN_COBYLA, 1)
  lower = ones(Float64, 1)*1e-5
  lower_bounds!(opt, lower)
  upper = ones(Float64, 1)*1e5
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 100)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, Float64[obsnodes[1].branch_scale])

  for obsnode in obsnodes
      obsnode.diffusion.branch_scale = minx[1]
      obsnode.switching.diffusion_r1.branch_scale = minx[1]
      obsnode.switching.diffusion_r2.branch_scale = minx[1]
  end
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
  set_parameters(obsnodes[h].diffusion, x[1], mod2pi(x[2]+pi)-pi, x[3], x[4], mod2pi(x[5]+pi)-pi, x[6], 1.0, x[7])

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
    for i=1:7
      store[i+1]  = x[i]
    end
  end

  return ll
end

function diffusionopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  seqindices,hindices = getindices(samples, h)
  store = ones(Float64,8)*(-1e20)
  localObjectiveFunction = ((param, grad) -> diffusionll(param, h, samples,seqindices,hindices, obsnodes, store))
  opt = Opt(:LN_COBYLA, 7)
  lower = zeros(Float64, 7)
  lower[1] = 1e-5
  lower[2] = -1000000.0
  lower[3] = 1e-5
  lower[4] = 1e-5
  lower[5] = -1000000.0
  lower[6] = 1e-5
  lower[7] = 1e-5
  lower_bounds!(opt, lower)

  upper = ones(Float64, 7)
  upper[1] = 1e5
  upper[2] = 1000000.0
  upper[3] = 1e5
  upper[4] = 1e5
  upper[5] = 1000000.0
  upper[6] = 1e5
  upper[7] = 1e5
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 200)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, get_parameters(obsnodes[h].diffusion))
  optx = store[2:8]

  set_parameters(obsnodes[h].diffusion, optx[1], mod2pi(optx[2]+pi)-pi, optx[3], optx[4], mod2pi(optx[5]+pi)-pi, optx[6], 1.0, optx[7])

  return optx
end


export mlopt
function mlopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
 aares = aapairopt(h, samples,obsnodes)
 diffusionres = diffusionopt(h, samples, obsnodes)
 ssres = Float64[]
 if obsnodes[h].usesecondarystructure
   ssres =  ssopt(h, samples, obsnodes)
  end
 return aares, diffusionres, ssres
end

export hmmopt
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

export prioropt
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


export aa_all
function aa_all(x::Array{Float64,1}, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  for obsnode in obsnodes
      set_aaratematrix(obsnode.aapairnode, x)
      set_aaratematrix(obsnode.switching.aapairnode_r1, x)
      set_aaratematrix(obsnode.switching.aapairnode_r2, x)
  end

  ll = 0.0
  for sample in samples
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    t = sample.params.t
    for a=1:length(align1)
      i = align1[a]
      j = align2[a]
      h = sample.states[a]
      if i == 0
        ll += get_data_lik_x0(obsnodes[h], seqpair.seq2,j, t)
      elseif j == 0
        ll += get_data_lik_xt(obsnodes[h], seqpair.seq1,i,t)
      else
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t)
      end
    end
  end

  return ll
end

export optimize_aaratematrix
function optimize_aaratematrix(samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  localObjectiveFunction = ((param, grad) -> aa_all(param, samples, obsnodes))
  opt = Opt(:LN_COBYLA, 190)
  lower = ones(Float64, 190)*1e-4
  lower_bounds!(opt, lower)
  upper = ones(Float64, 190)*1e4
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 1330)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, get_aaratematrixparameters(obsnodes[1].aapairnode))

   for obsnode in obsnodes
      set_aaratematrix(obsnode.aapairnode, minx)
      set_aaratematrix(obsnode.switching.aapairnode_r1, minx)
      set_aaratematrix(obsnode.switching.aapairnode_r2, minx)
  end
  return minx
end
