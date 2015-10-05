include("ModelIO.jl")
include("Sequence.jl")
include("ObservationNode.jl")
#Pkg.add("DataStructures")
#Pkg.add("Formatting")
include("AcceptanceLogger.jl")
include("Utils.jl")

using Formatting
using Distributions


START = 1
MATCH = 2
XINSERT = 3
YINSERT = 4
END = 5
N1 = 6
N2 = 7
N3 = 8
N4 = 9

function get_alignment_transition_probabilities(lambda::Float64, mu::Float64, r::Float64, t::Float64)
  Bt = (1.0 - exp((lambda-mu)*t))/(mu - lambda*exp((lambda-mu)*t))

  expmut = exp(-mu*t)
  aligntransprobs = zeros(Float64, 9, 9)
  aligntransprobs[START,N1] = 1.0

  aligntransprobs[MATCH,MATCH] = r
  aligntransprobs[MATCH,N1] = 1.0-r

  aligntransprobs[XINSERT,XINSERT] = r
  aligntransprobs[XINSERT,N3] = 1.0-r

  aligntransprobs[YINSERT,YINSERT] = r + (1.0-r)*(lambda*Bt)
  aligntransprobs[YINSERT,N2] = (1.0-r)*(1.0-lambda*Bt)

  aligntransprobs[END,END] = 0.0

  aligntransprobs[N1,YINSERT] =  lambda*Bt
  aligntransprobs[N1,N2] = 1.0 - lambda*Bt

  aligntransprobs[N2,END] = 1.0 - (lambda/mu)
  aligntransprobs[N2,N4] = lambda/mu

  aligntransprobs[N3,YINSERT] = (1.0 - mu*Bt - expmut)/(1.0-expmut)
  aligntransprobs[N3,N2] = (mu*Bt)/(1.0-expmut)

  aligntransprobs[N4,MATCH] = expmut
  aligntransprobs[N4,XINSERT] = 1.0 - expmut
  return aligntransprobs
end

type HMMParameters
  aligntransprobs::Array{Float64,2}
  numHiddenStates::Int
  hmminitprobs::Array{Float64, 1}
  hmmtransprobs::Array{Float64,2}
  logaligntransprobs::Array{Float64,2}
  loghmminitprobs::Array{Float64, 1}
  loghmmtransprobs::Array{Float64,2}

  function HMMParameters(aligntransprobs::Array{Float64,2}, hmminitprobs::Array{Float64, 1}, hmmtransprobs::Array{Float64,2})
    new(aligntransprobs, length(hmminitprobs), hmminitprobs, hmmtransprobs, log(aligntransprobs), log(hmminitprobs), log(hmmtransprobs))
  end
end

function tkf92(nsamples::Int, obsnodes::Array{ObservationNode,1}, rng::AbstractRNG, seqpair::SequencePair, pairparams::PairParameters, prior::PriorDistribution, hmminitprobs::Array{Float64, 1}, hmmtransprobs::Array{Float64,2}, cornercut::Int=100, fixAlignment::Bool=false, align1::Array{Int,1}=zeros(Int,1), align2::Array{Int,1}=zeros(Int,1), fixStates::Bool=false, states::Array{Int,1}=zeros(Int,1))
  aligntransprobs = get_alignment_transition_probabilities(pairparams.lambda,pairparams.mu,pairparams.r,pairparams.t)

  n = seqpair.seq1.length
  m = seqpair.seq2.length

  cache = Dict{Int,Float64}()
  numHiddenStates::Int = size(hmmtransprobs,1)
  choice = Array(Float64, numHiddenStates)
  alignmentpath = getalignmentpath(n,m,align1, align2,states)
  hmmparameters = HMMParameters(aligntransprobs, hmminitprobs, hmmtransprobs)
  for h=1:numHiddenStates
    choice[h] = tkf92forward(obsnodes, seqpair, pairparams.t, cache, hmmparameters,n,m,END,h, fixAlignment, cornercut, fixStates, alignmentpath)
  end
  sum = logsumexp(choice)
  choice = exp(choice - sum)



  samples = SequencePairSample[]
  for i=1:nsamples
    pairsample = SequencePairSample(seqpair, pairparams)
    tkf92sample(obsnodes, seqpair, pairparams.t, rng,cache, hmmparameters,n,m, END, sample(rng, choice), pairsample.align1,pairsample.align2, pairsample.states, fixAlignment, cornercut, fixStates, alignmentpath)
    push!(samples, pairsample)
  end

  ll = logprior(prior, pairparams)+sum
  return ll,samples
end


function tkf92sample(obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, rng::AbstractRNG, cache::Dict{Int,Float64}, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, align1::Array{Int,1}, align2::Array{Int,1}, states::Array{Int,1}, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1))
    if alignnode != START
      numAlignStates = size(hmmparameters.aligntransprobs,1)
      numHiddenStates = size(hmmparameters.hmmtransprobs,1)
      choice = Float64[-Inf for i=1:(numAlignStates*numHiddenStates)]

      newi = i
      newj = j
      for prevalignnode=1:numAlignStates
          for prevh=1:numHiddenStates
            transprob = 0.0
            newi = j
            if alignnode == MATCH || alignnode == XINSERT || alignnode == YINSERT
              transprob = hmmparameters.aligntransprobs[prevalignnode, alignnode]*hmmparameters.hmmtransprobs[prevh, h]
            elseif prevh == h
              transprob = hmmparameters.aligntransprobs[prevalignnode, alignnode]
            end
            if transprob > 0.0
              ll =  tkf92forward(obsnodes, seqpair, t, cache, hmmparameters,i,j, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)+log(transprob)
              choice[(prevalignnode-1)*numHiddenStates + prevh] = ll
            end
          end
      end

      sum = logsumexp(choice)
      choice = exp(choice - sum)
      s = sample(rng, choice)

      newalignnode = div(s-1, numHiddenStates) + 1
      newh = ((s-1) % numHiddenStates) + 1

      newi = i
      newj = j
      if newalignnode == MATCH
        newi = i-1
        newj = j-1
        unshift!(align1, i)
        unshift!(align2, j)
        unshift!(states, newh)
      elseif newalignnode == XINSERT
        newi = i-1
        newj = j
        unshift!(align1, i)
        unshift!(align2, 0)
        unshift!(states, newh)
      elseif newalignnode == YINSERT
        newi = i
        newj = j-1
        unshift!(align1, 0)
        unshift!(align2, j)
        unshift!(states, newh)
      end


      tkf92sample(obsnodes, seqpair, t, rng,cache, hmmparameters,newi,newj, newalignnode, newh, align1, align2, states, fixAlignment, cornercut, fixStates, alignmentpath)
  end
end

function uniquekey(seqpair::SequencePair, numHiddenStates::Int, i::Int, j::Int, alignnode::Int, h::Int)
  n = seqpair.seq1.length+1
  m = seqpair.seq2.length+1
  #println((i, j, alignnode, h), "\t", key)
  return (i)*m*9*numHiddenStates + (j)*9*numHiddenStates + (alignnode-1)*numHiddenStates + (h-1)
end

function tkf92forward(obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, cache::Dict{Int,Float64}, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1))

  if fixAlignment && i > 0 && j > 0
    if alignmentpath[i+1,j+1] <= 0
      return -Inf
    elseif fixStates && alignmentpath[i+1,j+1] != h
      return -Inf
    end
  end

  if abs(i-j) > cornercut + abs(seqpair.seq1.length-seqpair.seq2.length)
    return -Inf
  end

  if i < 0 || j < 0
    return -Inf
  elseif i == 0 && j == 0
    if alignnode == START
       return hmmparameters.loghmminitprobs[h]
    end
  end

  numHiddenStates = hmmparameters.numHiddenStates

  key::Int = uniquekey(seqpair, numHiddenStates, i, j, alignnode, h)
  if haskey(cache, key)
    return cache[key]
  end

  prevlik = 0.0
  datalik = 0.0
  if alignnode == MATCH
    if i > 0 && j > 0
      datalik = get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i,j, t)
    end
  elseif alignnode == XINSERT
    if i > 0
      datalik = get_data_lik(obsnodes[h], seqpair.seq1, i)
    end
  elseif alignnode == YINSERT
    if j > 0
      datalik = get_data_lik(obsnodes[h], seqpair.seq2, j)
    end
  end

  sum::Float64 = -Inf
  for prevalignnode=1:9
    if hmmparameters.aligntransprobs[prevalignnode, alignnode] > 0.0
        if alignnode == MATCH || alignnode == XINSERT || alignnode == YINSERT
          if fixAlignment && fixStates && i > 1 && j > 1
              prevh = 0
              if alignnode == MATCH
                prevh = alignmentpath[i,j]
              elseif alignnode == XINSERT
                prevh = alignmentpath[i,j+1]
              elseif alignnode == YINSERT
                prevh = alignmentpath[i+1,j]
              end
              if prevh > 0
                prevlik = -Inf
                if alignnode == MATCH
                  prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)
                elseif alignnode == XINSERT
                  prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)
                elseif alignnode == YINSERT
                  prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)
                end
                sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
              end
          else
            for prevh=1:numHiddenStates
              prevlik = -Inf
              if alignnode == MATCH
                prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)
              elseif alignnode == XINSERT
                prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)
              elseif alignnode == YINSERT
                prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)
              end
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          end
        else
          prevlik =  tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i, j, prevalignnode, h, fixAlignment, cornercut, fixStates, alignmentpath)
          sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode])
        end
    end
  end

  cache[key] = sum
  return sum
end


rho = 0.75
covariance = ones(Float64,2,2)
covariance[1,2] = rho
covariance[2,1] = rho
bivproposal =  MvNormal(covariance)
function mcmc_sequencepair(citer::Int, iter::Int, samplerate::Int, obsnodes::Array{ObservationNode, 1}, rng::AbstractRNG, initialSample::SequencePairSample, prior::PriorDistribution, hmminitprobs::Array{Float64,1}, hmmtransprobs::Array{Float64,2})
  seqpair = initialSample.seqpair
  pairparams = initialSample.params
  current = PairParameters(pairparams)
  proposed = PairParameters(pairparams)
  current_sample = initialSample

  mode = "a"
  if citer == 0
    mode = "w"
  end

  mcmcout = open(string("mcmc",fmt("04d", seqpair.id),".log"), mode)
  alignout = open(string("align",fmt("04d", seqpair.id),".log"), mode)
  acceptanceout = open(string("acceptance",fmt("04d", seqpair.id),".log"), mode)
  if citer == 0
    write(mcmcout, string("iter","\t", "currentll", "\t", "current_lambda","\t","current_mu","\t", "current_ratio", "\t", "current_r", "\t", "current_t","\n"))
  end

  samples = SequencePairSample[]

  logger = AcceptanceLogger()
  moveWeights = Float64[1, 20, 250, 250, 250, 250]
  cornercut= 100
  nsamples = 1
  currentll, current_samples = tkf92(nsamples, obsnodes, rng, seqpair, current, prior, hmminitprobs, hmmtransprobs, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
  current_sample = current_samples[end]


  proposedll = currentll
  logll = Float64[]
  for i=1:iter
      currentiter = citer + i - 1
      move = sample(rng, moveWeights)
      if move == 1
        currentll, current_samples = tkf92(nsamples, obsnodes, rng, seqpair, current, prior, hmminitprobs, hmmtransprobs, cornercut)
        current_sample = current_samples[end]
        currentll, dummy = tkf92(0, obsnodes, rng, seqpair, current, prior, hmminitprobs, hmmtransprobs, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)

        seqpair = current_sample.seqpair
        write(alignout, string(currentiter), "\n")
        write(alignout, string(join(current_sample.states, ""),"\n"))
        write(alignout, getalignment(seqpair.seq1, current_sample.align1),"\n")
        write(alignout, getalignment(seqpair.seq2, current_sample.align2),"\n\n")
        logAccept!(logger, "fullalignment")
      elseif move == 2
        seqpair = current_sample.seqpair
        #println(">>>>>>")
        #println(string(join(current_sample.states, "")))
        #println(getalignment(seqpair.seq1, current_sample.align1))
        #println(getalignment(seqpair.seq1, current_sample.align2))
        currentll, current_samples = tkf92(nsamples, obsnodes, rng, seqpair, current, prior, hmminitprobs, hmmtransprobs, cornercut, true, current_sample.align1, current_sample.align2, false, current_sample.states)
        current_sample = current_samples[end]
        #println(string(join(current_sample.states, "")))
        #println(getalignment(seqpair.seq1, current_sample.align1))
        #println(getalignment(seqpair.seq1, current_sample.align2))
        currentll, dummy = tkf92(0, obsnodes, rng, seqpair, current, prior, hmminitprobs, hmmtransprobs, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
        #println(string(join(dummy[1].states, "")))
        #println(getalignment(seqpair.seq1, dummy[1].align1))
        #println(getalignment(seqpair.seq1, dummy[1].align2))
        logAccept!(logger, "fullstates")
      elseif move >= 3
        movename = ""
        propratio = 0.0

        if move == 3
          proposed.lambda = current.lambda + randn(rng)*0.2
          #s = rand(bivproposal)*0.02
          #proposed.lambda = current.lambda + s[1]
          #proposed.mu = current.mu + s[2]
          movename = "lambda"
        elseif move == 4
          proposed.ratio = current.ratio + randn(rng)*0.05
          movename = "ratio"
        elseif move == 5
          proposed.r = current.r + randn(rng)*0.06
          movename = "r"
        elseif move == 6
          sigma = 0.01
          d1 = Truncated(Normal(current.t, sigma), 0.0, Inf)
          d2 = Truncated(Normal(proposed.t, sigma), 0.0, Inf)
          proposed.t = rand(d1)
          propratio = logpdf(d2, current.t) - logpdf(d1, proposed.t)
          movename = "t"
        end

        proposed.mu = proposed.lambda/proposed.ratio
        if(proposed.lambda > 0.0 && proposed.mu > 0.0 && proposed.lambda < proposed.mu && 0.0 < proposed.r < 1.0 && proposed.t > 0.0)
          proposedll, proposed_samples = tkf92(nsamples, obsnodes, rng, seqpair, proposed, prior, hmminitprobs, hmmtransprobs, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)

          a = rand(rng)
          if(exp(proposedll-currentll+propratio) > a)
            currentll = proposedll
            current = PairParameters(proposed)
            current_sample = proposed_samples[end]
            for c=1:length(proposed_samples)
              current_samples[c] = proposed_samples[c]
            end
            logAccept!(logger, movename)
          else
            proposed = PairParameters(current)
            logReject!(logger, movename)
          end
        else
            logReject!(logger, movename)
        end
      end

      push!(logll, currentll)
      if currentiter % samplerate == 0
        writell = currentll
        if writell == -Inf
          writell = -1e20
        end
        write(mcmcout, string(currentiter,"\t", writell, "\t", current.lambda,"\t",current.mu,"\t",current.ratio, "\t", current.r, "\t", current.t,"\n"))
        flush(mcmcout)
        for s in current_samples
          push!(samples, s)
        end
      end
  end

  close(mcmcout)
  close(alignout)

  write(acceptanceout, string(list(logger),"\n"))
  close(acceptanceout)

  expll = logsumexp(logll+log(1/float64(length(logll))))

  return citer+iter, current, samples, expll
end

#=
Pkg.add("NLopt")
Pkg.add("JuMP")
Pkg.add("Ipopt")
Pkg.update()=#
using NLopt
#using JuMP


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
      ll += get_data_lik(obsnodes[h], seqpair.seq2,j)
    elseif j == 0
      ll += get_data_lik(obsnodes[h], seqpair.seq1,i)
    else
      ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t)
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
  maxeval!(opt, 100)
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
      ll += get_data_lik(obsnodes[h], seqpair.seq2,j)
    elseif j == 0
      ll += get_data_lik(obsnodes[h], seqpair.seq1,i)
    else
      ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t)
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

function hmmopt(samples::Array{SequencePairSample,1}, numHiddenStates::Int)
  hmminitprobs = ones(Float64, numHiddenStates)*1e-4
  hmmtransprobs = ones(Float64, numHiddenStates, numHiddenStates)*1e-2
  for sample in samples
    seqpair = sample.seqpair
    align1 = sample.align1
    align2 = sample.align2
    states = sample.states
    println(states)
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
  maxeval!(opt, 2000)
  max_objective!(opt, localObjectiveFunction)
  (minf,minx,ret) = optimize(opt, ones(Float64, 8))
  return PriorDistribution(minx)
end

function load_sequences(datafile)
  f = open(datafile);
  line = 0
  seq1 = ""
  phi1 = Float64[]
  psi1 = Float64[]
  seq2 = ""
  phi2 = Float64[]
  psi2 = Float64[]
  id = 1
  pairs = SequencePair[]
  for ln in eachline(f)
    if ln[1] == '>'
      line = 0
    end

    if line == 1
      seq1 = strip(ln)
    elseif line == 2
      seq2 = strip(ln)
    elseif line == 3
      phi1 = Float64[float64(s) for s in split(ln, ",")]
    elseif line == 4
      psi1 = Float64[float64(s) for s in split(ln, ",")]
    elseif line == 5
      phi2 = Float64[float64(s) for s in split(ln, ",")]
    elseif line == 6
      psi2 = Float64[float64(s) for s in split(ln, ",")]
    end

    if line == 7
      seqpair = SequencePair(id, Sequence(seq1,phi1,psi1), Sequence(seq2,phi2,psi2))
      id += 1
      push!(pairs, seqpair)
    end
    line += 1
  end
  close(f)

  return pairs
end

function aic(ll::Float64, freeParameters::Int)
	return 2.0*freeParameters - 2.0*ll
end


function mpdalignment(samples::Array{SequencePairSample,1})
  n = samples[1].seqpair.seq1.length
  m = samples[1].seqpair.seq2.length

  counts = zeros(Float64, n+1, m+1)

  nsamples = length(samples)
  for sample in samples
    i = n
    j = m
    for (a,b) in zip(sample.align1, sample.align2)
      counts[a+1,b+1] += 1.0
    end
  end
  counts /= float64(nsamples)
  logprobs = log(counts)

  cache = Dict{Int,Any}()
  align1 = Int[]
  align2 = Int[]

  i = n
  j = m
  while true
    val, index = mpdalignment(logprobs, cache, i, j)

    if index == 1
      unshift!(align1, i)
      unshift!(align2, j)
      i -= 1
      j -= 1
    elseif index == 2
      unshift!(align1, i)
      unshift!(align2, 0)
      i -= 1
    elseif index == 3
      unshift!(align1, 0)
      unshift!(align2, j)
      j -= 1
    end

    if i == 0 && j == 0
      break
    end
  end

  seqpair = samples[1].seqpair
  posterior_probs = [counts[a+1,b+1] for (a,b) in zip(align1, align2)]


  return align1, align2, posterior_probs
end

function mpdalignment(logprobs::Array{Float64,2}, cache::Dict{Int,Any}, i::Int, j::Int)
  if i < 0 || j < 0
    return -Inf
  elseif i == 0 && j == 0
    return 0.0, 0
  end
  m = size(logprobs, 2)+1
  key = (i-1)*m + j-1
  if haskey(cache, key)
    return cache[key]
  end

  sel = zeros(Float64, 3)
  sel[1] = logprobs[i+1, j+1] + mpdalignment(logprobs, cache, i-1, j-1)[1]
  sel[2] = logprobs[i+1, 1] + mpdalignment(logprobs, cache, i-1, j)[1]
  sel[3] = logprobs[1, j+1] + mpdalignment(logprobs, cache, i, j-1)[1]
  index = indmax(sel)
  cache[key] = sel[index], index
  return cache[key]
end

function train()
  pairs = load_sequences("/home/michael/dev/moju/data.txt")[1:10]
  println("N=",length(pairs))

  srand(98418108751401)
  rng = MersenneTwister(242402531025555)

  mcmciter = 100
  samplerate = 20

  loadModel = true
  modelfile = "pairhmm.jls"


  numHiddenStates = 2
  hmminitprobs = ones(Float64,numHiddenStates)/float64(numHiddenStates)
  hmmtransprobs = ones(Float64, numHiddenStates, numHiddenStates)/float64(numHiddenStates)
  prior = PriorDistribution()
  obsnodes = ObservationNode[]
  for h=1:numHiddenStates
    push!(obsnodes, ObservationNode())
    v = rand(Float64,20)
    v /= sum(v)
    set_parameters(obsnodes[h].aapairnode, v, 1.0)
    set_parameters(obsnodes[h].diffusion, 0.1, rand()*2.0*pi - pi, 1.0, 0.1, rand()*2.0*pi - pi, 1.0, 1.0)
  end

  if loadModel
    ser = open(modelfile,"r")
    modelio::ModelIO = deserialize(ser)
    close(ser)
    prior = modelio.prior
    obsnodes = modelio.obsnodes
    hmminitprobs = modelio.hmminitprobs
    hmmtransprobs = modelio.hmmtransprobs
  end

  refs = RemoteRef[]
  for pair in pairs
    println(pair.id)
    ref = @spawn tkf92(1, ObservationNode[ObservationNode(obsnode) for obsnode in obsnodes], rng, pair, PairParameters(), prior, hmminitprobs, hmmtransprobs, 15)
    push!(refs,ref)
  end
  current_samples = SequencePairSample[]
  for ref in refs
    push!(current_samples, fetch(ref)[2][1])
  end
  println("Initialised")

  lastparams = [PairParameters() for i=1:length(pairs)]

  mlwriter = open(string("ml",numHiddenStates,".log"), "w")
  write(mlwriter, "iter\tll\tcount\tavgll\tnumFreeParameters\tAIC\tlambda_shape\tlambda_scale\tmu_shape\tmu_scale\tr_alpha\tr_beta\tt_shape\tt_scale\n")

  freeParameters = 6*numHiddenStates + (numHiddenStates-1) + (numHiddenStates*numHiddenStates - numHiddenStates) + numHiddenStates*19
  currentiter = 0

  for i=1:100
    samples = SequencePairSample[]
    samplell = Float64[]
    obscount = Int[]
    tic()
    refs = RemoteRef[]
    for k=1:length(pairs)
      ref = @spawn mcmc_sequencepair(currentiter, mcmciter, samplerate, ObservationNode[ObservationNode(obsnode) for obsnode in obsnodes], rng, current_samples[k], prior, hmminitprobs, hmmtransprobs)
      push!(refs,ref)
    end
    for k=1:length(pairs)
      f = fetch(refs[k])
      it, lastparams[k], ksamples, expll = f
      current_samples[k] = ksamples[end]
      push!(samplell, expll)
      push!(obscount, pairs[k].seq1.length + pairs[k].seq2.length)
      if k == length(pairs)
        currentiter = it
      end
      for ks in ksamples
        push!(samples, ks)
      end
    end
    estep_elapsed = toc()

    tic()

    aarefs = RemoteRef[]
    drefs = RemoteRef[]
    for h=1:numHiddenStates
      aaref = @spawn aapairopt(h, samples,obsnodes)
      push!(aarefs, aaref)

      dref = @spawn diffusionopt(h, samples, obsnodes)
      push!(drefs, dref)
    end
    for h=1:numHiddenStates
      set_parameters(obsnodes[h].aapairnode, fetch(aarefs[h]), 1.0)
      dopt = fetch(drefs[h])
      set_parameters(obsnodes[h].diffusion, dopt[1], mod2pi(dopt[2]+pi)-pi, dopt[3], dopt[4], mod2pi(dopt[5]+pi)-pi, dopt[6], 1.0)
    end


    hmminitprobs, hmmtransprobs = hmmopt(samples,numHiddenStates)
    prior = prioropt(samples, prior)
    println(prior)
    mstep_elapsed = toc()


    println(hmmopt(samples,numHiddenStates))
    println(length(samples))

    println("E-step time = ", estep_elapsed)
    println("M-step time = ", mstep_elapsed)


    write(mlwriter, string(i-1,"\t",sum(samplell), "\t", sum(obscount), "\t", sum(samplell)/sum(obscount),"\t", freeParameters,"\t", aic(sum(samplell), freeParameters), "\t", join(prior.params,"\t"), "\n"))
    flush(mlwriter)


    modelio = ModelIO(prior, obsnodes, hmminitprobs, hmmtransprobs)
    ser = open(modelfile,"w")
    serialize(ser, modelio)
    close(ser)
  end
end


OBSERVED_DATA = 0
MISSING_DATA = 1
function masksequences(seq1::Sequence, seq2::Sequence, mask::Array{Int,1})
  newseq1 = Sequence(seq1)
  newseq2 = Sequence(seq2)
  for i=1:newseq1.length
    if mask[1] == MISSING_DATA
      newseq1.seq[i] = 0
    end
    if mask[2] == MISSING_DATA
      newseq1.phi[i] = -1000.0
      newseq1.psi[i] = -1000.0
    end
  end
  for i=1:newseq2.length
    if mask[3] == MISSING_DATA
      newseq2.seq[i] = 0
    end
    if mask[4] == MISSING_DATA
      newseq2.phi[i] = -1000.0
      newseq2.psi[i] = -1000.0
    end
  end

  return newseq1, newseq2
end

function sample_missing_values(rng::AbstractRNG, obsnodes::Array{ObservationNode,1}, pairsample::SequencePairSample)
  seqpair = pairsample.seqpair
  newseq1 = Sequence(seqpair.seq1)
  newseq2 = Sequence(seqpair.seq2)
  t = pairsample.params.t

  i = 1
  for (a,b) in zip(pairsample.align1, pairsample.align2)
    h = pairsample.states[i]
    x0 = 0
    xt = 0
    phi0 = -1000.0
    psi0 = -1000.0
    phit = -1000.0
    psit = -1000.0
    if a > 0
       x0 =  seqpair.seq1.seq[a]
       phi0 = seqpair.seq1.phi[a]
       psi0 = seqpair.seq1.psi[a]
    end
    if b > 0
       xt =  seqpair.seq2.seq[b]
       phit = seqpair.seq2.phi[b]
       psit = seqpair.seq2.psi[b]
    end
    x0, xt = sample(obsnodes[h].aapairnode, rng, x0, xt, t)
    phi,psi = sample_phi_psi(obsnodes[h].diffusion, rng, phi0,phit,psi0,psit,t)
    if a > 0
      newseq1.seq[a] = x0
      newseq1.phi[a] = phi[1]
      newseq1.psi[a] = psi[1]
    end
    if b > 0
      newseq2.seq[b] = xt
      newseq2.phi[b] = phi[2]
      newseq2.psi[b] = psi[2]
    end

    i += 1
  end

  return SequencePair(0, newseq1,newseq2)
end

function angular_rmsd(theta1::Array{Float64, 1}, theta2::Array{Float64})
  dist =0.0
  len = 0
  for i=1:length(theta1)
    if theta1[i] > -100.0 && theta2[i] > -100.0
      x0 = cos(theta1[i])
      x1 = sin(theta1[i])
      y0 = cos(theta2[i])
      y1 = sin(theta2[i])
      c = y0-x0
      s = y1-x1
      dist += c*c + s*s
      len += 1
    end
  end

  return sqrt(dist/float(len))
end

function angular_rmsd(theta1::Array{Float64, 1}, theta2::Array{Float64},  align1::Array{Int}, align2::Array{Int})
  dist =0.0
  len = 0
  for (a,b) in zip(align1, align2)
    if a > 0 && b > 0
      if theta1[a] > -100.0 && theta2[b] > -100.0
        x0 = cos(theta1[a])
        x1 = sin(theta1[a])
        y0 = cos(theta2[b])
        y1 = sin(theta2[b])
        c = y0-x0
        s = y1-x1
        dist += c*c + s*s
        len += 1
      end
    end
  end

  return sqrt(dist/float(len))
end





function angular_mean(theta::Array{Float64, 1})
  if length(theta) == 0
    return -1000.0
  end

  c = 0.0
  s = 0.0
  total = float(length(theta))
  for t in theta
    c += cos(t)
    s += sin(t)
  end
  c /= total
  s /= total
  rho = sqrt(c*c + s*s)

  if s > 0
    return acos(c/rho)
  else
    return 2*pi - acos(c / rho)
  end
end

#using Gadfly
#using Compose
#p = plot(x=phi_i, y=psi_i, Geom.histogram2d(xbincount=30, ybincount=30), Scale.x_continuous(minvalue=float(-pi), maxvalue=float(pi)), Scale.y_continuous(minvalue=float(-pi), maxvalue=float(pi)))
#draw(SVG(string("hist", i, ".svg"), 5inch, 5inch), p)

function test()
  pairs = load_sequences("/home/michael/dev/moju/data2.txt")

  srand(98418108751401)
  rng = MersenneTwister(242402531025555)
  loadModel = false
  modelfile = "pairhmm.jls"

  ser = open(modelfile,"r")
  modelio::ModelIO = deserialize(ser)
  close(ser)
  prior = modelio.prior
  obsnodes = modelio.obsnodes
  hmminitprobs = modelio.hmminitprobs
  hmmtransprobs = modelio.hmmtransprobs
  numHiddenStates = length(hmminitprobs)

  mask = Int[OBSERVED_DATA, OBSERVED_DATA, OBSERVED_DATA, MISSING_DATA]
  for input in pairs[2:end]
    seq1, seq2 = masksequences(input.seq1, input.seq2, mask)
    masked = SequencePair(0,seq1, seq2)
    current_sample = tkf92(1, obsnodes, rng, masked, PairParameters(), prior, hmminitprobs, hmmtransprobs, 100)[2][1]
    ret = mcmc_sequencepair(0, 2500, 5, obsnodes, rng, current_sample, prior, hmminitprobs, hmmtransprobs)

    samples = ret[3]
    nsamples = length(ret[3])
    samples = samples[max(1,nsamples/2):end]
    filled_pairs = [sample_missing_values(rng, obsnodes, sample) for sample in samples]

    phi = Float64[]
    psi = Float64[]
    for i=1:filled_pairs[1].seq2.length
      phi_i = Float64[]
      psi_i = Float64[]
      for seqpair in filled_pairs
        if seqpair.seq2.phi[i] > -100.0
          push!(phi_i, seqpair.seq2.phi[i])
        end
        if seqpair.seq2.psi[i] > -100.0
          push!(psi_i, seqpair.seq2.psi[i])
        end
      end

      push!(phi, angular_mean(phi_i))
      push!(psi, angular_mean(psi_i))
    end

    align1, align2, posterior_probs = mpdalignment(samples)
    println(getalignment(input.seq1, align1))
    println(getalignment(input.seq2, align2))

    println("Homologue:\tphi=", angular_rmsd(input.seq2.phi, input.seq1.phi, align2, align1),"\tpsi=", angular_rmsd(input.seq2.psi, input.seq1.psi, align2, align1))
    println("Predicted:\tpsi=", angular_rmsd(input.seq2.phi, phi), "\tpsi=", angular_rmsd(input.seq2.psi, psi))
  end
end

train()
#test()



#@profile main()
#profilewriter = open("profile.log", "w")
#Profile.print(profilewriter)

# TODO ML alignment
# sampling
# hidden state conditioning
