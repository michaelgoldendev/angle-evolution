using NLopt


freqprior = Beta(4.0, 4.0)
alpha_prior = 0.05

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

function fullswitchingll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1}, store::Array{Float64, 1})
  set_parameters(obsnodes[h].switching, x)

  # dirichlet prior
  concentration_param = 1.01
  ll = sum((concentration_param-1.0)*log(obsnodes[h].switching.aapairnode_r1.eqfreqs))
  ll += sum((concentration_param-1.0)*log(obsnodes[h].switching.aapairnode_r2.eqfreqs))
  ll += logpdf(freqprior, obsnodes[h].switching.pi_r1)
  ll += -alpha_prior*obsnodes[h].switching.alpha

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

  #println("switching",ll)
  if ll > store[1]
    store[1] = ll
    for i=1:62
      store[i+1]  = x[i]
    end
  end

  return ll
end

export fullswitchingopt
function fullswitchingopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  seqindices,hindices = getindices(samples, h)
  store = ones(Float64,63)*(-1e20)
  localObjectiveFunction = ((param, grad) -> fullswitchingll(param, h, samples,seqindices,hindices, obsnodes, store))
  opt = Opt(:LN_COBYLA, 62)
  lower = zeros(Float64, 62)
  for i=1:40
    lower[i] = 0.0
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

  lower[55] = 1e-5
  lower[56] = 0.0
  for i=57:62
    lower[i] = 0.0
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

  upper[55] = 1e5
  upper[56] = 1.0
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 900)
  max_objective!(opt, localObjectiveFunction)

  try
    (minf,minx,ret) = optimize(opt, get_parameters(obsnodes[h].switching))
  catch
    println("ERROR",get_parameters(obsnodes[h].switching))
  end
  optx = store[2:63]

  set_parameters(obsnodes[h].switching, optx)
  return optx
end

function ssll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1}, regime::Int=0)
  neweqfreqs = x/sum(x)
  if(!(0.999 < sum(neweqfreqs) < 1.001))
    neweqfreqs = ones(Float64, 3)/3.0
  end
  concentration_param = 1.025
  ll = sum((concentration_param-1.0)*log(neweqfreqs))

  if obsnodes[1].useswitching
    if regime == 1
      set_parameters(obsnodes[h].switching.ss_r1, neweqfreqs,1.0)
    elseif regime == 2
      set_parameters(obsnodes[h].switching.ss_r2, neweqfreqs,1.0)
    end
    for (s,a) in zip(seqindices,hindices)
      sample = samples[s]
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      i = align1[a]
      j = align2[a]
      t = sample.params.t
      r = sample.regimes[a]
      r1 = div(r-1, 2) + 1
      r2 = (r-1) % 2 + 1
      #=
      if i == 0 && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq2,j,3,regime)
      elseif j == 0 && r1 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1,i,3,regime)
      elseif r1 == regime && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,3,regime)
      end=#
      if i > 0 && j > 0 && r1 == regime && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,3,regime)
      elseif j > 0 && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq2,j,3,regime)
      elseif i > 0 && r1 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1,i,3,regime)
      end
    end
  else
    set_parameters(obsnodes[h].ss, neweqfreqs,1.0)
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
  end

  return ll
end

function switchingll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1})
  obsnodes[h].switching.pi_r1 = x[1]
  obsnodes[h].switching.alpha = x[2]

  ll = 0.0
  ll += logpdf(freqprior, obsnodes[h].switching.pi_r1)
  ll += -alpha_prior*obsnodes[h].switching.alpha


  for (s,a) in zip(seqindices,hindices)
    sample = samples[s]
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    i = align1[a]
    j = align2[a]
    t = sample.params.t

    #=
    r = sample.regimes[a]
    r1 = div(r-1, 2) + 1
    r2 = (r-1) % 2 + 1
    ll += get_regime_pair_lik(obsnodes[h].switching, r1, r2, t)
    =#
    if i == 0
      ll += get_data_lik_x0(obsnodes[h], seqpair.seq2,j, t)
    elseif j == 0
      ll += get_data_lik_xt(obsnodes[h], seqpair.seq1,i, t)
    else
      ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t)
    end
  end
  return ll
end

export switchingopt
function switchingopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  seqindices,hindices = getindices(samples, h)
  localObjectiveFunction = ((param, grad) -> switchingll(param, h, samples, seqindices ,hindices, obsnodes))
  opt = Opt(:LN_COBYLA, 2)
  lower = ones(Float64, 2)
  lower[1] = 0.0
  lower[2] = 1e-5
  lower_bounds!(opt, lower)

  upper = ones(Float64, 2)
  upper[1] = 1.0
  upper[2] = 1e5
  upper_bounds!(opt, upper)

  xtol_rel!(opt,1e-4)
  maxeval!(opt, 200)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, [obsnodes[h].switching.pi_r1, obsnodes[h].switching.alpha])
  obsnodes[h].switching.pi_r1 = minx[1]
  obsnodes[h].switching.alpha = minx[2]

  return minx
end

export ssopt
function ssopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1}, regime::Int=0)
  seqindices,hindices = getindices(samples, h)
  localObjectiveFunction = ((param, grad) -> ssll(param, h, samples, seqindices ,hindices, obsnodes, regime))
  opt = Opt(:LN_COBYLA, 3)
  lower = ones(Float64, 3)*0.0
  lower_bounds!(opt, lower)
  upper = ones(Float64, 3)
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 200)
  max_objective!(opt, localObjectiveFunction)
  if regime == 0
    (minf,minx,ret) = optimize(opt, obsnodes[h].ss.ctmc.eqfreqs)
    set_parameters(obsnodes[h].ss, minx[1:3]/sum(minx[1:3]), 1.0)
  elseif regime == 1
    (minf,minx,ret) = optimize(opt, obsnodes[h].switching.ss_r1.ctmc.eqfreqs)
    set_parameters(obsnodes[h].switching.ss_r1, minx[1:3]/sum(minx[1:3]), 1.0)
  elseif regime == 2
    (minf,minx,ret) = optimize(opt, obsnodes[h].switching.ss_r2.ctmc.eqfreqs)
    set_parameters(obsnodes[h].switching.ss_r2, minx[1:3]/sum(minx[1:3]), 1.0)
  end
  return minx[1:3]/sum(minx[1:3])
end



function ssll_all(a::Float64, b::Float64, c::Float64, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  ll = 0.0
  if obsnodes[1].useswitching
    for h=1:length(obsnodes)
      set_parameters(obsnodes[h].switching.ss_r1, a, b, c, 1.0)
      set_parameters(obsnodes[h].switching.ss_r2, a, b, c, 1.0)
    end
    for sample in samples
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      t = sample.params.t
      for a=1:length(align1)
        i = align1[a]
        j = align2[a]
        h = sample.states[a]
        r = sample.regimes[a]
        r1 = div(r-1, 2) + 1
        r2 = (r-1) % 2 + 1
        for regime=1:2
          if i > 0 && j > 0 && r1 == regime && r2 == regime
            ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,3,regime)
          elseif j > 0 && r2 == regime
            ll += get_data_lik(obsnodes[h], seqpair.seq2,j,3,regime)
          elseif i > 0 && r1 == regime
            ll += get_data_lik(obsnodes[h], seqpair.seq1,i,3,regime)
          end
        end
      end
    end
  else
    for h=1:length(obsnodes)
      set_parameters(obsnodes[h].ss, a, b, c, 1.0)
    end
     for sample in samples
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      for a=1:length(align1)
        i = align1[a]
        j = align2[a]
        h = sample.states[a]
        t = sample.params.t
        if i == 0
          ll += get_data_lik(obsnodes[h], seqpair.seq2,j,3)
        elseif j == 0
          ll += get_data_lik(obsnodes[h], seqpair.seq1,i,3)
        else
          ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,3)
        end
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

function diffusion_error(angle_error_kappa::Float64, samples::Array{SequencePairSample,1}, modelparams::ModelParameters)
  obsnodes = modelparams.obsnodes
  for obsnode in obsnodes
      obsnode.angle_error_kappa = angle_error_kappa
  end

  ll = 0.0
  for sample in samples
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    t = sample.params.t
    for a=1:length(align1)
      ll += computeobsll(sample, modelparams.obsnodes, a)
    end
  end

  return ll
end

export optimize_diffusion_error
function optimize_diffusion_error(samples::Array{SequencePairSample,1}, modelparams::ModelParameters)
  localObjectiveFunction = ((param, grad) -> diffusion_error(param[1], samples, modelparams))
  opt = Opt(:LN_COBYLA, 1)
  lower = ones(Float64, 1)*1e-5
  lower_bounds!(opt, lower)
  upper = ones(Float64, 1)*600.0
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 100)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, Float64[modelparams.obsnodes[1].angle_error_kappa])

  for obsnode in modelparams.obsnodes
      obsnode.angle_error_kappa = minx[1]
  end
  return minx
end


function aapairll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1}, regime::Int=0)
  neweqfreqs = x[1:20]/sum(x[1:20])
  if(!(0.999 < sum(neweqfreqs) < 1.001))
    neweqfreqs = ones(Float64, 20)*0.05
  end

  concentration_param = 1.025
  ll = sum((concentration_param-1.0)*log(neweqfreqs))
  ll += logpdf(Gamma(obsnodes[1].aapairnode.diagsum/branch_gamma_scale + 1.0, branch_gamma_scale), x[21])

  if obsnodes[1].useswitching
    if regime == 1
      set_parameters(obsnodes[h].switching.aapairnode_r1, neweqfreqs, 1.0)
      obsnodes[h].switching.aapairnode_r1.branchscale = x[21]
    elseif regime == 2
      set_parameters(obsnodes[h].switching.aapairnode_r2, neweqfreqs, 1.0)
      obsnodes[h].switching.aapairnode_r2.branchscale = x[21]
    end

    for (s,a) in zip(seqindices,hindices)
      sample = samples[s]
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      i = align1[a]
      j = align2[a]
      t = sample.params.t
      r = sample.regimes[a]
      r1 = div(r-1, 2) + 1
      r2 = (r-1) % 2 + 1
      #=
      if i == 0 && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq2,j,1,regime)
      elseif j == 0 && r1 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1,i,1,regime)
      elseif r1 == regime && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,1,regime)
      end=#
      if i > 0 && j > 0 && r1 == regime && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,1,regime)
      elseif j > 0 && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq2,j,1,regime)
      elseif i > 0 && r1 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1,i,1,regime)
      end
    end
  else
    set_parameters(obsnodes[h].aapairnode, neweqfreqs, 1.0)
    obsnodes[h].aapairnode.branchscale = x[21]
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
  end

  return ll
end

export aapairopt
function aapairopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1}, regime::Int=0)
  seqindices,hindices = getindices(samples, h)
  localObjectiveFunction = ((param, grad) -> aapairll(param, h, samples, seqindices ,hindices, obsnodes, regime))
  opt = Opt(:LN_COBYLA, 21)
  lower_bounds!(opt, zeros(Float64, 21))
  upper = ones(Float64, 21)
  upper[21] = 1e5
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 400)
  max_objective!(opt, localObjectiveFunction)
  if regime == 0
    initial = Float64[]
    for i=1:20
      push!(initial, obsnodes[h].aapairnode.eqfreqs[i])
    end
    push!(initial, obsnodes[h].aapairnode.branchscale)
    (minf,minx,ret) = optimize(opt, initial)
    set_parameters(obsnodes[h].aapairnode, minx[1:20]/sum(minx[1:20]), 1.0)
    obsnodes[h].aapairnode.branchscale = minx[21]
  elseif regime == 1
    initial = Float64[]
    for i=1:20
      push!(initial, obsnodes[h].switching.aapairnode_r1.eqfreqs[i])
    end
    push!(initial, obsnodes[h].switching.aapairnode_r1.branchscale)
    (minf,minx,ret) = optimize(opt, initial)
    set_parameters(obsnodes[h].switching.aapairnode_r1, minx[1:20]/sum(minx[1:20]), 1.0)
    obsnodes[h].switching.aapairnode_r1.branchscale = minx[21]
  elseif regime == 2
    initial = Float64[]
    for i=1:20
      push!(initial, obsnodes[h].switching.aapairnode_r2.eqfreqs[i])
    end
    push!(initial, obsnodes[h].switching.aapairnode_r2.branchscale)
    (minf,minx,ret) = optimize(opt, initial)
    set_parameters(obsnodes[h].switching.aapairnode_r2, minx[1:20]/sum(minx[1:20]), 1.0)
    obsnodes[h].switching.aapairnode_r2.branchscale = minx[21]
  end

  return minx[1:20]/sum(minx[1:20]), minx[21]
end

function diffusionll(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1}, store::Array{Float64, 1}, regime::Int=0)


  ll = 0.0
  if obsnodes[1].useswitching
    if regime == 1
      set_parameters(obsnodes[h].switching.diffusion_r1, x[1], mod2pi(x[2]+pi)-pi, x[3], x[4], mod2pi(x[5]+pi)-pi, x[6], x[7], 1.0, x[8])
    elseif regime == 2
      set_parameters(obsnodes[h].switching.diffusion_r2, x[1], mod2pi(x[2]+pi)-pi, x[3], x[4], mod2pi(x[5]+pi)-pi, x[6], x[7], 1.0, x[8])
    end
    for (s,a) in zip(seqindices,hindices)
      sample = samples[s]
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      i = align1[a]
      j = align2[a]
      t = sample.params.t
      r = sample.regimes[a]
      r1 = div(r-1, 2) + 1
      r2 = (r-1) % 2 + 1
      #=
      if i == 0 && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq2,j,2,regime)
      elseif j == 0 && r1 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1,i,2,regime)
      elseif r1 == regime && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,2,regime)
      end=#
      if i > 0 && j > 0 && r1 == regime && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,2,regime)
      elseif j > 0 && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq2,j,2,regime)
      elseif i > 0 && r1 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1,i,2,regime)
      end
    end
  else
    set_parameters(obsnodes[h].diffusion, x[1], mod2pi(x[2]+pi)-pi, x[3], x[4], mod2pi(x[5]+pi)-pi, x[6], x[7], 1.0, x[8])
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
  end


  if ll > store[1]
    store[1] = ll
    for i=1:8
      store[i+1]  = x[i]
    end
  end

  return ll
end

export diffusionopt
function diffusionopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1}, regime::Int=0)
  seqindices,hindices = getindices(samples, h)
  store = ones(Float64,9)*(-1e20)
  localObjectiveFunction = ((param, grad) -> diffusionll(param, h, samples,seqindices,hindices, obsnodes, store, regime))
  opt = Opt(:LN_COBYLA, 8)
  lower = zeros(Float64, 8)
  lower[1] = 1e-5
  lower[2] = -1000000.0
  lower[3] = 1e-5
  lower[4] = 1e-5
  lower[5] = -1000000.0
  lower[6] = 1e-5
  lower[7] = -100.0
  lower[8] = 1e-5
  lower_bounds!(opt, lower)

  upper = ones(Float64, 8)
  upper[1] = 1e5
  upper[2] = 1000000.0
  upper[3] = 1e5
  upper[4] = 1e5
  upper[5] = 1000000.0
  upper[6] = 1e5
  upper[7] = 100.0
  upper[8] = 1e5
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 200)
  max_objective!(opt, localObjectiveFunction)
  if regime == 0
    (minf,minx,ret) = optimize(opt, get_parameters(obsnodes[h].diffusion))
  elseif regime == 1
    (minf,minx,ret) = optimize(opt, get_parameters(obsnodes[h].switching.diffusion_r1))
  elseif regime == 2
    (minf,minx,ret) = optimize(opt, get_parameters(obsnodes[h].switching.diffusion_r2))
  end
  optx = store[2:9]
  optx[2] = mod2pi(optx[2]+pi)-pi
  optx[5] = mod2pi(optx[5]+pi)-pi

  if regime == 0
    set_parameters(obsnodes[h].diffusion, optx[1], optx[2], optx[3], optx[4], optx[5], optx[6], optx[7], 1.0, optx[8])
  elseif regime == 1
    set_parameters(obsnodes[h].switching.diffusion_r1, optx[1], optx[2], optx[3], optx[4], optx[5], optx[6], optx[7], 1.0, optx[8])
  elseif regime == 2
    set_parameters(obsnodes[h].switching.diffusion_r2, optx[1], optx[2], optx[3], optx[4], optx[5], optx[6], optx[7], 1.0, optx[8])
  end


  return optx
end


export mlopt
function mlopt(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  if obsnodes[1].useswitching
    aares1 = aapairopt(h, samples, obsnodes, 1)
    aares2 = aapairopt(h, samples, obsnodes, 2)
    diffusionres1 = diffusionopt(h, samples, obsnodes, 1)
    diffusionres2 = diffusionopt(h, samples, obsnodes, 2)
    ssres1 = ssopt(h, samples, obsnodes, 1)
    ssres2 = ssopt(h, samples, obsnodes, 2)
    switching = switchingopt(h, samples, obsnodes)
    #branchscale1 = optimize_aabranchscale_hiddenstate(h, samples, obsnodes, 1)
    #branchscale2 = optimize_aabranchscale_hiddenstate(h, samples, obsnodes, 2)

    return aares1, aares2, diffusionres1, diffusionres2, ssres1, ssres2, switching
  else
   aares = aapairopt(h, samples,obsnodes)
   diffusionres = diffusionopt(h, samples, obsnodes)
   ssres = Float64[]
   if obsnodes[h].usesecondarystructure
     ssres =  ssopt(h, samples, obsnodes)
    end
   branchscale = optimize_aabranchscale_hiddenstate(h, samples, obsnodes)
   return aares, diffusionres, ssres
  end
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
      if obsnode.useswitching
        set_aaratematrix(obsnode.switching.aapairnode_r1, x)
        set_aaratematrix(obsnode.switching.aapairnode_r2, x)
      else
        set_aaratematrix(obsnode.aapairnode, x)
      end
  end

  ll = 0.0
  if obsnodes[1].useswitching
    for sample in samples
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      t = sample.params.t
      for a=1:length(align1)
        i = align1[a]
        j = align2[a]
        h = sample.states[a]
        r = sample.regimes[a]
        r1 = div(r-1, 2) + 1
        r2 = (r-1) % 2 + 1
        for regime=1:2
          if i > 0 && j > 0 && r1 == regime && r2 == regime
            ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,1,regime)
          elseif j > 0 && r2 == regime
            ll += get_data_lik(obsnodes[h], seqpair.seq2,j,1,regime)
          elseif i > 0 && r1 == regime
            ll += get_data_lik(obsnodes[h], seqpair.seq1,i,1,regime)
          end
        end
      end
    end
  else
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
            ll += get_data_lik(obsnodes[h], seqpair.seq2,j,1)
          elseif j == 0
            ll += get_data_lik(obsnodes[h], seqpair.seq1,i,1)
          else
            ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,1)
          end
      end
    end
  end

  return ll
end

export optimize_aaratematrix
function optimize_aaratematrix(samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  localObjectiveFunction = ((param, grad) -> aa_all(param, samples, obsnodes))
  opt = Opt(:LN_COBYLA, 190)
  lower = zeros(Float64, 190)*1e-5
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

function aa_all_h(x::Array{Float64,1}, samples::Array{SequencePairSample,1}, obsnode::ObservationNode, h::Int)
  seqindices,hindices = getindices(samples,h)
  ll = 0.0
  if obsnode.useswitching
    set_aaratematrix(obsnode.switching.aapairnode_r1, x)
    set_aaratematrix(obsnode.switching.aapairnode_r2, x)
    for (s,a) in zip(seqindices,hindices)
      sample = samples[s]
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      i = align1[a]
      j = align2[a]
      t = sample.params.t
      h = sample.states[a]
      r = sample.regimes[a]
      r1 = div(r-1, 2) + 1
      r2 = (r-1) % 2 + 1
      for regime=1:2
        if i > 0 && j > 0 && r1 == regime && r2 == regime
          ll += get_data_lik(obsnode, seqpair.seq1, seqpair.seq2, i, j, t,1,regime)
        elseif j > 0 && r2 == regime
          ll += get_data_lik(obsnode, seqpair.seq2,j,1,regime)
        elseif i > 0 && r1 == regime
          ll += get_data_lik(obsnode, seqpair.seq1,i,1,regime)
        end
      end
    end
  else
    set_aaratematrix(obsnode.aapairnode, x)
    for (s,a) in zip(seqindices,hindices)
      sample = samples[s]
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      i = align1[a]
      j = align2[a]
      t = sample.params.t
      if i == 0
        ll += get_data_lik(obsnode, seqpair.seq2,j,1)
      elseif j == 0
        ll += get_data_lik(obsnode, seqpair.seq1,i,1)
      else
        ll += get_data_lik(obsnode, seqpair.seq1, seqpair.seq2, i, j, t,1)
      end
    end
  end
  return ll
end

export aa_all_parallel
function aa_all_parallel(x::Array{Float64,1}, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  numHiddenStates = length(obsnodes)
  refs = RemoteRef[]
  for h=1:numHiddenStates
    obsnodecopy = deepcopy(obsnodes[h])
    ref = @spawn aa_all_h(x, samples, obsnodecopy, h)
    push!(refs, ref)
  end

  totalll = 0.0
  for h=1:numHiddenStates
    totalll += fetch(refs[h])
  end

  return totalll
end

export optimize_aaratematrix_parallel
function optimize_aaratematrix_parallel(samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  localObjectiveFunction = ((param, grad) -> aa_all_parallel(param, samples, obsnodes))
  opt = Opt(:LN_COBYLA, 190)
  lower = zeros(Float64, 190)*1e-5
  lower_bounds!(opt, lower)
  upper = ones(Float64, 190)*1e4
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 1130)
  max_objective!(opt, localObjectiveFunction)
  initial = get_aaratematrixparameters(obsnodes[1].aapairnode)
  if obsnodes[1].useswitching
    initial = get_aaratematrixparameters(obsnodes[1].switching.aapairnode_r1)
  end
  (minf,minx,ret) = optimize(opt, initial)

  for obsnode in obsnodes
    if obsnode.useswitching
      set_aaratematrix(obsnode.switching.aapairnode_r1, minx)
      set_aaratematrix(obsnode.switching.aapairnode_r2, minx)
    else
      set_aaratematrix(obsnode.aapairnode, minx)
    end
  end
  return minx
end


export aabranchscale
function aabranchscale(x::Array{Float64,1}, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  for obsnode in obsnodes
      obsnode.aapairnode.branchscale = x[1]
      obsnode.switching.aapairnode_r1.branchscale = x[1]
      obsnode.switching.aapairnode_r2.branchscale = x[1]
  end

  ll = 0.0
  if obsnodes[1].useswitching
    for sample in samples
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      t = sample.params.t
      for a=1:length(align1)
        i = align1[a]
        j = align2[a]
        h = sample.states[a]
        r = sample.regimes[a]
        r1 = div(r-1, 2) + 1
        r2 = (r-1) % 2 + 1
        for regime=1:2
          if i > 0 && j > 0 && r1 == regime && r2 == regime
            ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,1,regime)
          elseif j > 0 && r2 == regime
            ll += get_data_lik(obsnodes[h], seqpair.seq2,j,1,regime)
          elseif i > 0 && r1 == regime
            ll += get_data_lik(obsnodes[h], seqpair.seq1,i,1,regime)
          end
        end
      end
    end
  else
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
            ll += get_data_lik(obsnodes[h], seqpair.seq2,j,1)
          elseif j == 0
            ll += get_data_lik(obsnodes[h], seqpair.seq1,i,1)
          else
            ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,1)
          end
      end
    end
  end

  return ll
end

export optimize_aabranchscale
function optimize_aabranchscale(samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1})
  localObjectiveFunction = ((param, grad) -> aabranchscale(param, samples, obsnodes))
  opt = Opt(:LN_COBYLA, 1)
  lower = zeros(Float64, 1)*1e-5
  lower_bounds!(opt, lower)
  upper = ones(Float64, 1)*1e5
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 150)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, Float64[obsnodes[1].aapairnode.branchscale])

   for obsnode in obsnodes
      obsnode.aapairnode.branchscale = minx[1]
      obsnode.switching.aapairnode_r1.branchscale = minx[1]
      obsnode.switching.aapairnode_r2.branchscale = minx[1]
  end
  return minx
end


branch_gamma_scale = 0.25
export aabranchscalell_hiddenstate
function aabranchscalell_hiddenstate(x::Array{Float64,1}, h::Int, samples::Array{SequencePairSample,1}, seqindices::Array{Int,1}, hindices::Array{Int,1}, obsnodes::Array{ObservationNode, 1}, regime::Int=0)
  ll = logpdf(Gamma(obsnodes[1].aapairnode.diagsum/branch_gamma_scale + 1.0, branch_gamma_scale), x[1])
  if obsnodes[1].useswitching
    if regime == 1
      obsnodes[h].switching.aapairnode_r1.branchscale = x[1]
    elseif regime == 2
      obsnodes[h].switching.aapairnode_r2.branchscale = x[1]
    end

    for (s,a) in zip(seqindices,hindices)
      sample = samples[s]
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      i = align1[a]
      j = align2[a]
      t = sample.params.t
      r = sample.regimes[a]
      r1 = div(r-1, 2) + 1
      r2 = (r-1) % 2 + 1
      if i > 0 && j > 0 && r1 == regime && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t,1,regime)
      elseif j > 0 && r2 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq2,j,1,regime)
      elseif i > 0 && r1 == regime
        ll += get_data_lik(obsnodes[h], seqpair.seq1,i,1,regime)
      end
    end
  else
    obsnodes[h].aapairnode.branchscale = x[1]
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
  end

  return ll
end

export optimize_aabranchscale_hiddenstate
function optimize_aabranchscale_hiddenstate(h::Int, samples::Array{SequencePairSample,1}, obsnodes::Array{ObservationNode, 1}, regime::Int=0)
  seqindices,hindices = getindices(samples,h)
  localObjectiveFunction = ((param, grad) -> aabranchscalell_hiddenstate(param, h, samples, seqindices ,hindices, obsnodes, regime))
  opt = Opt(:LN_COBYLA, 1)
  lower = zeros(Float64, 1)*1e-5
  lower_bounds!(opt, lower)
  upper = ones(Float64, 1)*1e5
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 150)
  max_objective!(opt, localObjectiveFunction)
  if regime == 0
    (minf,minx,ret) = optimize(opt, [obsnodes[h].aapairnode.branchscale])
    obsnodes[h].aapairnode.branchscale = minx[1]
  elseif regime == 1
    (minf,minx,ret) = optimize(opt, [obsnodes[h].switching.aapairnode_r1.branchscale])
    obsnodes[h].switching.aapairnode_r1.branchscale = minx[1]
  elseif regime == 2
    (minf,minx,ret) = optimize(opt, [obsnodes[h].switching.aapairnode_r2.branchscale])
    obsnodes[h].switching.aapairnode_r2.branchscale = minx[1]
  end
  return minx[1]
end


export aashapescalell
function aashapescalell(x::Array{Float64,1}, obsnodes::Array{ObservationNode, 1})
  obsnodes[1].aapairnode.diagsum = x[1]
  ll = 0.0
  if obsnodes[1].useswitching
    for h=1:length(obsnodes)
      ll += logpdf(Gamma(obsnodes[1].aapairnode.diagsum/branch_gamma_scale + 1.0, branch_gamma_scale), obsnodes[h].switching.aapairnode_r1.branchscale)
      ll += logpdf(Gamma(obsnodes[1].aapairnode.diagsum/branch_gamma_scale + 1.0, branch_gamma_scale), obsnodes[h].switching.aapairnode_r2.branchscale)
    end
  else
    for h=1:length(obsnodes)
      ll += logpdf(Gamma(obsnodes[1].aapairnode.diagsum/branch_gamma_scale + 1.0, branch_gamma_scale), obsnodes[h].aapairnode.branchscale)
    end
  end
  return ll
end

export aashapescale
function aashapescale(obsnodes::Array{ObservationNode, 1})
  localObjectiveFunction = ((param, grad) -> aashapescalell(param, obsnodes))
  opt = Opt(:LN_COBYLA, 1)
  lower = zeros(Float64, 1)
  lower_bounds!(opt, lower)
  upper = ones(Float64, 1)*1e5
  upper_bounds!(opt, upper)
  xtol_rel!(opt,1e-4)
  maxeval!(opt, 150)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, [obsnodes[1].aapairnode.diagsum])
  obsnodes[1].aapairnode.diagsum = minx[1]
  return minx[1]
end
