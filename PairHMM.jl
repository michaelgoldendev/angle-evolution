include("ModelParameters.jl")
include("Sequence.jl")
include("ObservationNode.jl")

include("AcceptanceLogger.jl")
include("Utils.jl")
include("ModelOptimization.jl")

using Formatting
using Distributions
using NLopt

#=
Pkg.add("DataStructures")
Pkg.add("Formatting")
Pkg.add("NLopt")
Pkg.add("JuMP")
Pkg.add("Ipopt")
using JuMP
=#



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
    new(aligntransprobs, length(hmminitprobs), hmminitprobs, hmmtransprobs, map(safelog, aligntransprobs), map(safelog, hmminitprobs), map(safelog, hmmtransprobs))
  end
end

type HMMCache
  caches::Array{Dict{Int, Float64},1}
  n::Int
  m::Int
  numHiddenStates::Int
  cornercut::Int
  cornercutbound::Int
  #matrix::Array{Float64,2}

  function HMMCache(n::Int, m::Int, numHiddenStates::Int, cornercut::Int, fixAlignment::Bool, fixStates::Bool)
    caches = Dict{Int,Float64}[]

    hintsize = -1
    if !(fixAlignment || fixStates)
      hintsize = max(n,m)*cornercut*10
    end
    for h=1:numHiddenStates
      d = Dict{Int,Float64}()
      if hintsize > 0
        sizehint!(d, hintsize)
      end
      push!(caches, d)
    end
    new(caches, n,m,numHiddenStates,cornercut, cornercut + abs(n-m))
  end
end

function putvalue(cache::HMMCache, i::Int, j::Int, alignnode::Int, h::Int, v::Float64)
  key::Int = i*(cache.m+1)*9 + j*9 + (alignnode-1) + 1
  cache.caches[h][key] = v

  #cache.matrix[h,key] = v
end

function getvalue(cache::HMMCache, i::Int, j::Int, alignnode::Int, h::Int)
  key::Int = i*(cache.m+1)*9 + j*9 + (alignnode-1) + 1
  if(haskey(cache.caches[h],key))
    return cache.caches[h][key]
  else
    return Inf
  end
  #return cache.matrix[h,key]
end

function uniquekey(seqpair::SequencePair, numHiddenStates::Int, i::Int, j::Int, alignnode::Int, h::Int)
  n = seqpair.seq1.length+1
  m = seqpair.seq2.length+1
  #println((i, j, alignnode, h), "\t", key)
  return (i)*m*9*numHiddenStates + (j)*9*numHiddenStates + (alignnode-1)*numHiddenStates + (h-1)
end

function tkf92(nsamples::Int, obsnodes::Array{ObservationNode,1}, rng::AbstractRNG, seqpair::SequencePair, pairparams::PairParameters, prior::PriorDistribution, hmminitprobs::Array{Float64, 1}, hmmtransprobs::Array{Float64,2}, cornercut::Int=10000000, fixAlignment::Bool=false, align1::Array{Int,1}=zeros(Int,1), align2::Array{Int,1}=zeros(Int,1), fixStates::Bool=false, states::Array{Int,1}=zeros(Int,1))
  #println(pairparams,"\t",cornercut)
  aligntransprobs = get_alignment_transition_probabilities(pairparams.lambda,pairparams.mu,pairparams.r,pairparams.t)

  n = seqpair.seq1.length
  m = seqpair.seq2.length



  numHiddenStates::Int = size(hmmtransprobs,1)
  cache = HMMCache(n,m,numHiddenStates,cornercut, fixAlignment, fixStates)

  choice = Array(Float64, numHiddenStates)
  alignmentpath = getalignmentpath(n,m,align1, align2,states)
  #println(alignmentpath)
  hmmparameters = HMMParameters(aligntransprobs, hmminitprobs, hmmtransprobs)
  if !fixAlignment
    len = min(n,m)
    for i=1:len
        tkf92forward(obsnodes, seqpair, pairparams.t, cache, hmmparameters,i,i,END,1, fixAlignment, cornercut, fixStates, alignmentpath)
    end
    for i=1:n
        tkf92forward(obsnodes, seqpair, pairparams.t, cache, hmmparameters,i,m,END,1, fixAlignment, cornercut, fixStates, alignmentpath)
    end
  end

  for h=1:numHiddenStates
    choice[h] = tkf92forward(obsnodes, seqpair, pairparams.t, cache, hmmparameters,n,m,END,h, fixAlignment, cornercut, fixStates, alignmentpath)
    #println(length(cache))
  end

  sum = logsumexp(choice)
  choice = exp(choice - sum)

  samples = SequencePairSample[]
  for i=1:nsamples
    pairsample = SequencePairSample(seqpair, pairparams)
    tkf92sample(obsnodes, seqpair, pairparams.t, rng,cache, hmmparameters,n,m, END, sample(rng, choice), pairsample.align1,pairsample.align2, pairsample.states, fixAlignment, cornercut, fixStates, alignmentpath)
    push!(samples, pairsample)
  end

  #=
  for h=1:numHiddenStates
    println("size=", length(cache.caches[h]))
  end=#

  ll = logprior(prior, pairparams)+sum
  #=
  #if !fixAlignment
    println(ll, "\t", pairparams)
    align1, align2, posterior_probs = mpdalignment(samples)
    println(getalignment(seqpair.seq1, align1))
    println(getalignment(seqpair.seq2, align2))
    println(posterior_probs)
  #end=#
  return ll,samples
end




function tkf92sample(obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, rng::AbstractRNG, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, align1::Array{Int,1}, align2::Array{Int,1}, states::Array{Int,1}, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1))
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

function tkf92forward(obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1))
  if i < 0 || j < 0
    return -Inf
  end

  v = getvalue(cache,i,j,alignnode,h)
  #println(alignmentpath)
  #println(i,"\t",j,"\t",alignnode, "\t", h,"\t",v,"\t",cache.cornercutbound)
  #exit()
  if v != Inf
    return v
  end

  if abs(i-j) > cache.cornercutbound
    return -Inf
  elseif i == 0 && j == 0
    if alignnode == START
       return hmmparameters.loghmminitprobs[h]
    end
  end


  if fixAlignment && i > 0 && j > 0
    if alignmentpath[i+1,j+1] <= 0
      return -Inf
    elseif fixStates && alignmentpath[i+1,j+1] != h
      return -Inf
    end
  end

  numHiddenStates::Int = hmmparameters.numHiddenStates
  prevlik::Float64 = 0.0
  datalik = 0.0
  sum::Float64 = -Inf
  if alignnode == MATCH || alignnode == XINSERT || alignnode == YINSERT
    if alignnode == MATCH
      if i > 0 && j > 0
        datalik = get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i,j, t)
      end
    elseif alignnode == XINSERT
      if i > 0
        datalik = get_data_lik_x0(obsnodes[h], seqpair.seq1, i, t)
      end
    elseif alignnode == YINSERT
      if j > 0
        datalik = get_data_lik_xt(obsnodes[h], seqpair.seq2, j, t)
      end
    end

    for prevalignnode=1:9
      if hmmparameters.aligntransprobs[prevalignnode, alignnode] > 0.0
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
          if alignnode == MATCH
            for prevh=1:numHiddenStates
              prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          elseif alignnode == XINSERT
            for prevh=1:numHiddenStates
              prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          elseif alignnode == YINSERT
            for prevh=1:numHiddenStates
              prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath)
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          end
        end
      end
    end
  else
    for prevalignnode=1:9
      if hmmparameters.aligntransprobs[prevalignnode, alignnode] > 0.0
        prevlik =  tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i, j, prevalignnode, h, fixAlignment, cornercut, fixStates, alignmentpath)
        sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode])
      end
    end
  end

  putvalue(cache,i,j,alignnode,h,sum)
  return sum
end


rho = 0.75
covariance = ones(Float64,2,2)
covariance[1,2] = rho
covariance[2,1] = rho
bivproposal =  MvNormal(covariance)
function mcmc_sequencepair(citer::Int, iter::Int, samplerate::Int, obsnodes::Array{ObservationNode, 1}, rng::AbstractRNG, initialSample::SequencePairSample, prior::PriorDistribution, hmminitprobs::Array{Float64,1}, hmmtransprobs::Array{Float64,2}, cornercut::Int=100)
  seqpair = initialSample.seqpair
  pairparams = initialSample.params
  current = PairParameters(pairparams)
  proposed = PairParameters(pairparams)
  current_sample = initialSample

  writeoutput = false

  mode = "a"
  if citer == 0
    mode = "w"
  end

  if writeoutput
    mcmcout = open(string("mcmc",fmt("04d", seqpair.id),".log"), mode)
    alignout = open(string("align",fmt("04d", seqpair.id),".log"), mode)
    acceptanceout = open(string("acceptance",fmt("04d", seqpair.id),".log"), mode)
    if citer == 0
      write(mcmcout, string("iter","\t", "currentll", "\t", "current_lambda","\t","current_mu","\t", "current_ratio", "\t", "current_r", "\t", "current_t","\n"))
    end
  end

  samples = SequencePairSample[]

  logger = AcceptanceLogger()
  moveWeights = Float64[0.5, 20, 100, 100, 100, 100]
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
        if writeoutput
          write(alignout, string(currentiter), "\n")
          write(alignout, string(join(current_sample.states, ""),"\n"))
          write(alignout, getalignment(seqpair.seq1, current_sample.align1),"\n")
          write(alignout, getalignment(seqpair.seq2, current_sample.align2),"\n\n")
        end
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
        if writeoutput
          write(mcmcout, string(currentiter,"\t", writell, "\t", current.lambda,"\t",current.mu,"\t",current.ratio, "\t", current.r, "\t", current.t,"\n"))
      end
      end

      if (currentiter+1) % samplerate == 0
        for s in current_samples
          push!(samples, s)
        end
      end
  end

  if writeoutput
    close(mcmcout)
    close(alignout)

    write(acceptanceout, string(list(logger),"\n"))
    close(acceptanceout)
  end

  expll = logsumexp(logll+log(1/float64(length(logll))))

  return citer+iter, current, samples, expll
end

function mlalignmentopt(seqpair::SequencePair, obsnodes::Array{ObservationNode,1}, prior::PriorDistribution, hmminitprobs::Array{Float64,1}, hmmtransprobs::Array{Float64,2}, cornercut::Int)
  fixAlignment=false
  align1 = Int[]
  align2 = Int[]
  for i=1:seqpair.seq1.length
    push!(align1, i)
    push!(align2, 0)
  end
  for i=1:seqpair.seq2.length
    push!(align1, 0)
    push!(align2, i)
  end
  println(getalignment(seqpair.seq1, align1))
  println(getalignment(seqpair.seq2, align2))

  localObjectiveFunction = ((param, grad) -> tkf92(100, obsnodes, MersenneTwister(330101840810391), seqpair, PairParameters(param), prior, hmminitprobs, hmmtransprobs, cornercut, fixAlignment, align1, align2)[1])
  opt = Opt(:LN_COBYLA, 4)
  lower_bounds!(opt, ones(Float64, 4)*1e-10)

  upper = Float64[1e10, 1.0, 0.999, 1e10]
  upper_bounds!(opt, upper)

  xtol_rel!(opt,1e-4)
  maxeval!(opt, 200)
  max_objective!(opt, localObjectiveFunction)
  initial = Float64[0.3, 0.98, 0.5, 0.25]
  (maxf,maxx,ret) = optimize(opt, initial)
  println(maxf,maxx)
  return 0
end

function mlalign()
  pairs = load_sequences("data/holdout_data.txt")

  srand(98418108751401)
  rng = MersenneTwister(242402531025555)
  modelfile = "models/pairhmm16.jls"

  cornercut = 400

  ser = open(modelfile,"r")
  modelio::ModelParameters = deserialize(ser)
  close(ser)
  prior = modelio.prior
  obsnodes = modelio.obsnodes
  hmminitprobs = modelio.hmminitprobs
  hmmtransprobs = modelio.hmmtransprobs
  numHiddenStates = length(hmminitprobs)

  println("use_switching", obsnodes[1].useswitching)
  println("H=", numHiddenStates)

  mask = Int[OBSERVED_DATA, OBSERVED_DATA, OBSERVED_DATA, MISSING_DATA]

  seq1, seq2 = masksequences(pairs[1].seq1, pairs[1].seq2, mask)
  #seq1, seq2 = masksequences(pairs[1].seq1, Sequence("ATG"), mask)
  mlalignmentopt(SequencePair(0,seq1, seq2), obsnodes, prior, hmminitprobs, hmmtransprobs, cornercut)
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

function parallelmcmc(modk::Int, modlen::Int, currentiter::Int, mcmciter::Int, samplerate::Int, obsnodes::Array{ObservationNode, 1}, rng::AbstractRNG, current_samples::Array{SequencePairSample,1}, prior::PriorDistribution, hmminitprobs::Array{Float64,1}, hmmtransprobs::Array{Float64,2}, cornercut::Int)
  samples = SequencePairSample[]
  currentsamples = SequencePairSample[]
  samplell = Float64[]
  k = modk
  while k <= length(current_samples)
    dummy1, dummy2, ksamples, expll = mcmc_sequencepair(currentiter, mcmciter, samplerate, obsnodes, rng, current_samples[k], prior, hmminitprobs, hmmtransprobs, cornercut)
    for ks in ksamples
      push!(samples, ks)
    end
    push!(currentsamples, ksamples[end])
    push!(samplell, expll)
    k += modlen
  end
  return samples, currentsamples, samplell
end


function train()
  maxiters = 10000
  cornercutinit = 25
  cornercut = 75
  useswitching = false
  useparallel = true
  pairs = load_sequences("data/data.txt")[1:10]
  println("N=",length(pairs))

  srand(98418108751401)
  rng = MersenneTwister(242402531025555)

  mcmciter = 30
  samplerate = 5

  numHiddenStates = 4

  loadModel = false
  modelfile = string("models/pairhmm",numHiddenStates,".jls")



  hmminitprobs = ones(Float64,numHiddenStates)/float64(numHiddenStates)
  hmmtransprobs = ones(Float64, numHiddenStates, numHiddenStates)/float64(numHiddenStates)
  prior = PriorDistribution()
  obsnodes = ObservationNode[]
  for h=1:numHiddenStates
    push!(obsnodes, ObservationNode())
    v = rand(Float64,20)
    v /= sum(v)
    obsnodes[h].useswitching = useswitching
    if useswitching
      v = rand(Float64,20)
      v /= sum(v)
      set_parameters(obsnodes[h].switching.aapairnode_r1, v, 1.0)
      v = rand(Float64,20)
      v /= sum(v)
      set_parameters(obsnodes[h].switching.aapairnode_r2, v, 1.0)
      set_parameters(obsnodes[h].switching.diffusion_r1, 0.1, rand()*2.0*pi - pi, 1.0, 0.1, rand()*2.0*pi - pi, 1.0, 1.0)
      set_parameters(obsnodes[h].switching.diffusion_r2, 0.1, rand()*2.0*pi - pi, 1.0, 0.1, rand()*2.0*pi - pi, 1.0, 1.0)
      obsnodes[h].switching.alpha = 5.0 + 20.0*rand(rng)
      obsnodes[h].switching.pi_r1 = rand(rng)
    else
      set_parameters(obsnodes[h].aapairnode, v, 1.0)
      set_parameters(obsnodes[h].diffusion, 0.1, rand()*2.0*pi - pi, 1.0, 0.1, rand()*2.0*pi - pi, 1.0, 1.0)
    end
  end

  if loadModel
    ser = open(modelfile,"r")
    modelio::ModelParameters = deserialize(ser)
    close(ser)
    prior = modelio.prior
    obsnodes = modelio.obsnodes
    hmminitprobs = modelio.hmminitprobs
    hmmtransprobs = modelio.hmmtransprobs
    numHiddenStates = length(hmminitprobs)
  end

  tic()

  current_samples = SequencePairSample[]
  if useparallel
    refs = RemoteRef[]
    for k=1:length(pairs)
      println(pairs[k].id)
      ref = @spawn tkf92(1, ObservationNode[ObservationNode(obsnode) for obsnode in obsnodes], MersenneTwister(abs(rand(Int))), pairs[k], PairParameters(), prior, hmminitprobs, hmmtransprobs, cornercutinit)
      push!(refs,ref)
    end
    for ref in refs
      res = fetch(ref)
      push!(current_samples, res[2][1])
    end
  else
    for pair in pairs
      println(pair.id)
      push!(current_samples, tkf92(1, obsnodes, rng, pair, PairParameters(), prior, hmminitprobs, hmmtransprobs, cornercutinit)[2][1])
    end
  end
  toc()

  println("Initialised")

  mlwriter = open(string("logs/ml",numHiddenStates,".log"), "w")
  write(mlwriter, "iter\tll\tcount\tavgll\tnumFreeParameters\tAIC\tlambda_shape\tlambda_scale\tmu_shape\tmu_scale\tr_alpha\tr_beta\tt_shape\tt_scale\n")

  freeParameters = 6*numHiddenStates + (numHiddenStates-1) + (numHiddenStates*numHiddenStates - numHiddenStates) + numHiddenStates*19
  currentiter = 0

  for i=1:maxiters
    println("ITER=",i)
    samples = SequencePairSample[]
    samplell = Float64[]
    obscount = Int[]
    tic()

    if useparallel
      refs = RemoteRef[]
      for k=1:length(pairs)
        println("K=",k)
        ref = @spawn mcmc_sequencepair(currentiter, mcmciter, samplerate, ObservationNode[ObservationNode(obsnode) for obsnode in obsnodes], MersenneTwister(abs(rand(Int))), SequencePairSample(current_samples[k]), prior, hmminitprobs, hmmtransprobs, cornercut)
        push!(refs,ref)
      end
      for k=1:length(pairs)
        it, newparams, ksamples, expll = fetch(refs[k])
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
      #=
      refs = RemoteRef[]
      numthreads = 8
      for modk=1:numthreads
        ref = @spawn parallelmcmc(modk,numthreads,currentiter, mcmciter, samplerate, ObservationNode[ObservationNode(obsnode) for obsnode in obsnodes], MersenneTwister(abs(rand(Int))), current_samples, prior, hmminitprobs, hmmtransprobs, cornercut)
        push!(refs,ref)
      end
      for ref in refs
        newsamples, csamples, dummy = fetch(ref)
        for s in newsamples
          push!(samples, s)
        end
        for c in csamples
          push!(current_samples, c)
        end
      end
      =#
    else
      for k=1:length(pairs)
        it, newparams, ksamples, expll = mcmc_sequencepair(currentiter, mcmciter, samplerate, ObservationNode[ObservationNode(obsnode) for obsnode in obsnodes],  MersenneTwister(abs(rand(Int))), current_samples[k], prior, hmminitprobs, hmmtransprobs, cornercut)
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
    end
    estep_elapsed = toc()

    tic()

    if useswitching
        refs = RemoteRef[]
        #refs2 = RemoteRef[]
      for h=1:numHiddenStates
        ref = @spawn  switchopt(h, samples,obsnodes)
        push!(refs, ref)
      end
      for h=1:numHiddenStates
        optx = fetch(refs[h])
        set_parameters(obsnodes[h].switching.aapairnode_r1, optx[1:20]/sum(optx[1:20]), 1.0)
        set_parameters(obsnodes[h].switching.aapairnode_r2, optx[21:40]/sum(optx[21:40]), 1.0)
        set_parameters(obsnodes[h].switching.diffusion_r1, optx[41], mod2pi(optx[42]+pi)-pi, optx[43], optx[44], mod2pi(optx[45]+pi)-pi, optx[46], 1.0)
        set_parameters(obsnodes[h].switching.diffusion_r2, optx[47], mod2pi(optx[48]+pi)-pi, optx[49], optx[50], mod2pi(optx[51]+pi)-pi, optx[52], 1.0)
        obsnodes[h].switching.alpha = optx[53]
        obsnodes[h].switching.pi_r1 = optx[54]
      end

      for h=1:numHiddenStates
        println("H=",h,"\t", obsnodes[h].switching.alpha, "\t", obsnodes[h].switching.pi_r1)

      end
    else
      if useparallel
        refs = RemoteRef[]
        for h=1:numHiddenStates
          ref = @spawn mlopt(h, samples,obsnodes)
          push!(refs, ref)
        end
        for h=1:numHiddenStates
          params = fetch(refs[h])
          set_parameters(obsnodes[h].aapairnode, params[1], 1.0)
          dopt = params[2]
          set_parameters(obsnodes[h].diffusion, dopt[1], mod2pi(dopt[2]+pi)-pi, dopt[3], dopt[4], mod2pi(dopt[5]+pi)-pi, dopt[6], 1.0)
        end
      else
        for h=1:numHiddenStates
          params = mlopt(h, samples,obsnodes)
          set_parameters(obsnodes[h].aapairnode, params[1], 1.0)
          dopt = params[2]
          set_parameters(obsnodes[h].diffusion, dopt[1], mod2pi(dopt[2]+pi)-pi, dopt[3], dopt[4], mod2pi(dopt[5]+pi)-pi, dopt[6], 1.0)
        end
      end
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


    modelio = ModelParameters(prior, obsnodes, hmminitprobs, hmmtransprobs)
    ser = open(modelfile,"w")
    serialize(ser, modelio)
    close(ser)


    write_hiddenstates(modelio, "logs/hiddenstates.txt")
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
      newseq1.phi_error[i] = -1000.0
      newseq1.psi_error[i] = -1000.0
    end
  end
  for i=1:newseq2.length
    if mask[3] == MISSING_DATA
      newseq2.seq[i] = 0
    end
    if mask[4] == MISSING_DATA
      newseq2.phi[i] = -1000.0
      newseq2.psi[i] = -1000.0
      newseq2.phi_error[i] = -1000.0
      newseq2.psi_error[i] = -1000.0
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
       #phi0 = seqpair.seq1.phi_error[a]
       #psi0 = seqpair.seq1.psi_error[a]
       phi0 = seqpair.seq1.phi[a]
       psi0 = seqpair.seq1.psi[a]
    end
    if b > 0
       xt =  seqpair.seq2.seq[b]
       #phit = seqpair.seq2.phi_error[b]
       #psit = seqpair.seq2.psi_error[b]
       phit = seqpair.seq2.phi[b]
       psit = seqpair.seq2.psi[b]
    end
    x0, xt, phi, psi  = sample(obsnodes[h], rng, x0, xt, phi0,phit,psi0,psit, t)
    #phi,psi = sample_phi_psi(obsnodes[h].diffusion, rng, phi0,phit,psi0,psit,t)
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

function pimod(angle::Float64)
  theta = mod2pi(angle)
  if theta > pi
    return theta -2.0*pi
  else
    return theta
  end
end



function test()
  pairs = load_sequences("data/holdout_data.txt")

  srand(98418108751401)
  rng = MersenneTwister(242402531025555)
  modelfile = "models/pairhmm16.jls"

  cornercut = 75

  ser = open(modelfile,"r")
  modelio::ModelParameters = deserialize(ser)
  close(ser)
  prior = modelio.prior
  obsnodes = modelio.obsnodes
  hmminitprobs = modelio.hmminitprobs
  hmmtransprobs = modelio.hmmtransprobs
  numHiddenStates = length(hmminitprobs)

  println("use_switching", obsnodes[1].useswitching)
  println("H=", numHiddenStates)

  mask = Int[OBSERVED_DATA, OBSERVED_DATA, OBSERVED_DATA, MISSING_DATA]
  for k=1:length(pairs)
    input = pairs[k]
    #input = SequencePair(k,pairs[k].seq1, pairs[k+1].seq2)
    seq1, seq2 = masksequences(input.seq1, input.seq2, mask)
    masked = SequencePair(0,seq1, seq2)
    current_sample = tkf92(1, obsnodes, rng, masked, PairParameters(), prior, hmminitprobs, hmmtransprobs, cornercut)[2][1]
    ret = mcmc_sequencepair(0, 4000, 5, obsnodes, rng, current_sample, prior, hmminitprobs, hmmtransprobs, cornercut)

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
    #=for (a,b,c,d) in zip(input.seq2.phi, phi, input.seq2.psi, psi)
      println(a,"\t", pimod(b), "\t", c, "\t", pimod(d))
    end
    =#

    for (a,b) in zip(align2, align1)
      if a > 0 && b > 0
        #println(input.seq2.psi[a],"\t", input.seq1.psi[b], "\t", pimod(psi[a]))
      end
    end

    println("Homologue:\tphi=", angular_rmsd(input.seq2.phi, input.seq1.phi, align2, align1),"\tpsi=", angular_rmsd(input.seq2.psi, input.seq1.psi, align2, align1))
    println("Predicted:\tpsi=", angular_rmsd(input.seq2.phi, phi), "\tpsi=", angular_rmsd(input.seq2.psi, psi))
  end
end

#mlalign()
#test()
train()



#@profile train()
#profilewriter = open("profile.log", "w")
#Profile.print(profilewriter)

# TODO ML alignment
# sampling
# hidden state conditioning
