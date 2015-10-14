include("ModelParameters.jl")
include("Sequence.jl")
include("ObservationNode.jl")

include("AcceptanceLogger.jl")
include("Utils.jl")
include("ModelOptimization.jl")
include("AngleUtils.jl")
include("AlignmentUtils.jl")

using Formatting
using Distributions

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
      hintsize = max(n,m)*cornercut*5
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

function tkf92(nsamples::Int, rng::AbstractRNG, seqpair::SequencePair, pairparams::PairParameters, modelparams::ModelParameters, cornercut::Int=10000000, fixAlignment::Bool=false, align1::Array{Int,1}=zeros(Int,1), align2::Array{Int,1}=zeros(Int,1), fixStates::Bool=false, states::Array{Int,1}=zeros(Int,1), partialAlignment::Bool=false, samplealignments::Bool=true)
  #println(pairparams,"\t",cornercut)
  prior = modelparams.prior
  obsnodes = modelparams.obsnodes
  hmminitprobs = modelparams.hmminitprobs
  hmmtransprobs = modelparams.hmmtransprobs
  aligntransprobs = get_alignment_transition_probabilities(pairparams.lambda,pairparams.mu,pairparams.r,pairparams.t)

  n = seqpair.seq1.length
  m = seqpair.seq2.length

  starti = 0
  endi = 0
  startj = 0
  endj  = 0
  if partialAlignment && fixAlignment && !fixStates
    width = 75
    starti = rand(1:n)
    endi = starti+width
    startj = starti + rand(-50:50)
    endj = startj+width
    #=
    starti = rand(1:max(1,n-width))
    endi = starti+75
    startj = starti + rand(-50:50)
    endj = startj+75=#
  end


  numHiddenStates::Int = size(hmmtransprobs,1)
  cache = HMMCache(n,m,numHiddenStates,cornercut, fixAlignment, fixStates)

  choice = Array(Float64, numHiddenStates)
  alignmentpath = getalignmentpath(n,m,align1, align2,states)
  #println(alignmentpath)
  hmmparameters = HMMParameters(aligntransprobs, hmminitprobs, hmmtransprobs)
  if !fixAlignment
    len = min(n,m)
    for i=1:len
      tkf92forward(obsnodes, seqpair, pairparams.t, cache, hmmparameters,i,i,END,1, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
    end
    for i=1:n
      tkf92forward(obsnodes, seqpair, pairparams.t, cache, hmmparameters,i,m,END,1, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
    end
  end

  for h=1:numHiddenStates
    choice[h] = tkf92forward(obsnodes, seqpair, pairparams.t, cache, hmmparameters,n,m,END,h, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
    #println(length(cache))
  end
  sum = logsumexp(choice)
  choice = exp(choice - sum)

  if samplealignments
    samples = SequencePairSample[]
    for i=1:nsamples
      pairsample = SequencePairSample(seqpair, pairparams)
      tkf92sample(obsnodes, seqpair, pairparams.t, rng,cache, hmmparameters,n,m, END, sample(rng, choice), pairsample.align1,pairsample.align2, pairsample.states, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
      push!(samples, pairsample)
    end

    ll = logprior(prior, pairparams)+sum
    #println(ll)
    return ll,samples
  else
    mlsample = SequencePairSample(seqpair, pairparams)
    tkf92viterbi(obsnodes, seqpair, pairparams.t, rng,cache, hmmparameters,n,m, END, indmax(choice), mlsample.align1,mlsample.align2, mlsample.states, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
    ll = logprior(prior, pairparams)+sum
    #println(ll,"\t", length(cache.caches[1]))
    return ll,mlsample
  end
end




function tkf92sample(obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, rng::AbstractRNG, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, align1::Array{Int,1}, align2::Array{Int,1}, states::Array{Int,1}, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1), starti::Int=0, endi::Int=0, startj::Int=0, endj::Int=0)
  newalignnode::Int = alignnode
  newh::Int = h
  newi::Int = i
  newj::Int = j

  numAlignStates::Int = size(hmmparameters.aligntransprobs,1)
  numHiddenStates::Int = size(hmmparameters.hmmtransprobs,1)

  while !(newalignnode == START && newi == 0 && newj == 0)

    choice = Float64[-Inf for i=1:(numAlignStates*numHiddenStates)]

    for prevalignnode=1:numAlignStates
      for prevh=1:numHiddenStates
        transprob = 0.0
        if newalignnode == MATCH || newalignnode == XINSERT || newalignnode == YINSERT
          transprob = hmmparameters.aligntransprobs[prevalignnode, newalignnode]*hmmparameters.hmmtransprobs[prevh, newh]
        elseif prevh == newh
          transprob = hmmparameters.aligntransprobs[prevalignnode, newalignnode]
        end
        if transprob > 0.0
          ll =  tkf92forward(obsnodes, seqpair, t, cache, hmmparameters,newi,newj, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)+log(transprob)
          choice[(prevalignnode-1)*numHiddenStates + prevh] = ll
        end
      end
    end

    s = GumbelSample(rng, choice)
    #sum = logsumexp(choice)
    #choice = exp(choice - sum)
    #s = sample(rng, choice)

    newalignnode = div(s-1, numHiddenStates) + 1
    newh = ((s-1) % numHiddenStates) + 1

    if newalignnode == MATCH
      unshift!(align1, newi)
      unshift!(align2, newj)
      unshift!(states, newh)
      newi = newi-1
      newj = newj-1
    elseif newalignnode == XINSERT
      unshift!(align1, newi)
      unshift!(align2, 0)
      unshift!(states, newh)
      newi = newi-1
      newj = newj
    elseif newalignnode == YINSERT
      unshift!(align1, 0)
      unshift!(align2, newj)
      unshift!(states, newh)
      newi = newi
      newj = newj-1
    end
  end
end



function tkf92sample2(obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, rng::AbstractRNG, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, align1::Array{Int,1}, align2::Array{Int,1}, states::Array{Int,1}, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1), starti::Int=0, endi::Int=0, startj::Int=0, endj::Int=0)
  if alignnode != START
    numAlignStates = size(hmmparameters.aligntransprobs,1)
    numHiddenStates = size(hmmparameters.hmmtransprobs,1)
    choice = Float64[-Inf for i=1:(numAlignStates*numHiddenStates)]

    for prevalignnode=1:numAlignStates
      for prevh=1:numHiddenStates
        transprob = 0.0
        if alignnode == MATCH || alignnode == XINSERT || alignnode == YINSERT
          transprob = hmmparameters.aligntransprobs[prevalignnode, alignnode]*hmmparameters.hmmtransprobs[prevh, h]
        elseif prevh == h
          transprob = hmmparameters.aligntransprobs[prevalignnode, alignnode]
        end
        if transprob > 0.0
          ll =  tkf92forward(obsnodes, seqpair, t, cache, hmmparameters,i,j, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)+log(transprob)
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


    tkf92sample(obsnodes, seqpair, t, rng,cache, hmmparameters,newi,newj, newalignnode, newh, align1, align2, states, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
  end
end

function tkf92viterbi(obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, rng::AbstractRNG, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, align1::Array{Int,1}, align2::Array{Int,1}, states::Array{Int,1}, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1), starti::Int=0, endi::Int=0, startj::Int=0, endj::Int=0)
  newalignnode::Int = alignnode
  newh::Int = h
  newi::Int = i
  newj::Int = j

  numAlignStates::Int = size(hmmparameters.aligntransprobs,1)
  numHiddenStates::Int = size(hmmparameters.hmmtransprobs,1)

  while !(newalignnode == START && newi == 0 && newj == 0)

    choice = Float64[-Inf for i=1:(numAlignStates*numHiddenStates)]

    for prevalignnode=1:numAlignStates
      for prevh=1:numHiddenStates
        transprob = 0.0
        if newalignnode == MATCH || newalignnode == XINSERT || newalignnode == YINSERT
          transprob = hmmparameters.aligntransprobs[prevalignnode, newalignnode]*hmmparameters.hmmtransprobs[prevh, newh]
        elseif prevh == newh
          transprob = hmmparameters.aligntransprobs[prevalignnode, newalignnode]
        end
        if transprob > 0.0
          ll =  tkf92forward(obsnodes, seqpair, t, cache, hmmparameters,newi,newj, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)+log(transprob)
          choice[(prevalignnode-1)*numHiddenStates + prevh] = ll
        end
      end
    end

    s = indmax(choice)
    newalignnode = div(s-1, numHiddenStates) + 1
    newh = ((s-1) % numHiddenStates) + 1

    if newalignnode == MATCH
      unshift!(align1, newi)
      unshift!(align2, newj)
      unshift!(states, newh)
      newi = newi-1
      newj = newj-1
    elseif newalignnode == XINSERT
      unshift!(align1, newi)
      unshift!(align2, 0)
      unshift!(states, newh)
      newi = newi-1
      newj = newj
    elseif newalignnode == YINSERT
      unshift!(align1, 0)
      unshift!(align2, newj)
      unshift!(states, newh)
      newi = newi
      newj = newj-1
    end
  end
end

function uniquekey(seqpair::SequencePair, numHiddenStates::Int, i::Int, j::Int, alignnode::Int, h::Int)
  n = seqpair.seq1.length+1
  m = seqpair.seq2.length+1
  #println((i, j, alignnode, h), "\t", key)
  return (i)*m*9*numHiddenStates + (j)*9*numHiddenStates + (alignnode-1)*numHiddenStates + (h-1)
end

function tkf92forward(obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1), starti::Int=0, endi::Int=0, startj::Int=0, endj::Int=0)
  if i < 0 || j < 0
    return -Inf
  end

  v = getvalue(cache,i,j,alignnode,h)
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


  if fixAlignment && i > 0 && j > 0 && !(starti <= i <= endi && startj <= j <= endj)
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
        if fixStates && fixAlignment && i > 1 && j > 1
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
              prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
            elseif alignnode == XINSERT
              prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
            elseif alignnode == YINSERT
              prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
            end
            sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
          end
        else
          if alignnode == MATCH
            for prevh=1:numHiddenStates
              prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          elseif alignnode == XINSERT
            for prevh=1:numHiddenStates
              prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i-1, j, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          elseif alignnode == YINSERT
            for prevh=1:numHiddenStates
              prevlik = tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          end
        end
      end
    end
  else
    for prevalignnode=1:9
      if hmmparameters.aligntransprobs[prevalignnode, alignnode] > 0.0
        prevlik =  tkf92forward(obsnodes, seqpair, t, cache, hmmparameters, i, j, prevalignnode, h, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
        sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode])
      end
    end
  end

  putvalue(cache,i,j,alignnode,h,sum)
  return sum
end

function mcmc_sequencepair(citer::Int, niter::Int, samplerate::Int, rng::AbstractRNG, initialSample::SequencePairSample, modelparams::ModelParameters, cornercut::Int=100, fixAlignment::Bool=false, writeoutput::Bool=false, outputdir::AbstractString="")
  burnin = niter / 2
  seqpair = initialSample.seqpair
  pairparams = initialSample.params
  current = PairParameters(pairparams)
  proposed = PairParameters(pairparams)
  current_sample = initialSample

  #simplemodelparams = readmodel("models/pairhmm4.jls")

  mode = "a"
  if citer == 0
    mode = "w"
  end

  if writeoutput
    mcmcout = open(string(outputdir, "mcmc",fmt("04d", seqpair.id),".log"), mode)
    alignout = open(string(outputdir, "align",fmt("04d", seqpair.id),".log"), mode)
    acceptanceout = open(string(outputdir, "acceptance",fmt("04d", seqpair.id),".log"), mode)
    if citer == 0
      write(mcmcout, string("iter","\t", "currentll", "\t", "current_lambda","\t","current_mu","\t", "current_ratio", "\t", "current_r", "\t", "current_t","\n"))
    end
  end

  samples = SequencePairSample[]

  logger = AcceptanceLogger()
  moveWeights = Float64[0.0, 0.0, 40.0, 400.0, 400.0, 400.0, 400.0]
  if fixAlignment
    moveWeights = Float64[0.0, 40.0, 0.0, 400.0, 400.0, 200.0, 400.0]
  end
  nsamples = 1
  currentll, current_samples = tkf92(nsamples, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
  current_sample = current_samples[end]


  proposedll = currentll
  logll = Float64[]
  for i=1:niter
    currentiter = citer + i - 1
    move = sample(rng, moveWeights)
    if move == 1
      currentll, current_samples = tkf92(nsamples, rng, seqpair, current, modelparams, cornercut)
      current_sample = current_samples[end]
      currentll, dummy = tkf92(0, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)

      seqpair = current_sample.seqpair
      if writeoutput
        write(alignout, string(currentiter), "\n")
        write(alignout, string(join(current_sample.states, ""),"\n"))
        write(alignout, getalignment(seqpair.seq1, current_sample.align1),"\n")
        write(alignout, getalignment(seqpair.seq2, current_sample.align2),"\n\n")
        flush(alignout)
      end
      logAccept!(logger, "fullalignment")
    elseif move == 2
      seqpair = current_sample.seqpair
      currentll, current_samples = tkf92(nsamples, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2, false, current_sample.states)
      current_sample = current_samples[end]
      currentll, dummy = tkf92(0, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
      logAccept!(logger, "fullstates")
    elseif move == 3
      seqpair = current_sample.seqpair
      if writeoutput
        println("I=", i)
        println(">>>>>>")
        println(getalignment(seqpair.seq1, current_sample.align1))
        println(getalignment(seqpair.seq2, current_sample.align2))
      end
      currentll, current_samples = tkf92(nsamples, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2, false, current_sample.states, true)
      current_sample = current_samples[end]
      if writeoutput
        println(getalignment(seqpair.seq1, current_sample.align1))
        println(getalignment(seqpair.seq2, current_sample.align2))
      end
      currentll, dummy = tkf92(0, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
      logAccept!(logger, "partialalignment")

      #=
          dummyll, simple_samples = tkf92(1, rng, seqpair, current, simplemodelparams, cornercut)
          simple_sample = simple_samples[1]

          beforell, dummy = tkf92(0, rng, seqpair, current, simplemodelparams, cornercut, true, current_sample.align1, current_sample.align2)
          afterll, dummy = tkf92(0, rng, seqpair, current, simplemodelparams, cornercut, true, simple_sample.align1, simple_sample.align2)
          complexll, dummy = tkf92(0, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2)
          proposedll, dummy = tkf92(1, rng, seqpair, current, modelparams, cornercut, true, simple_sample.align1, simple_sample.align2)
          newsample = dummy[1]
          ll = (proposedll-complexll)+(beforell-afterll)

          a = rand(rng)
          println("A", beforell,"\t", afterll, "\t", complexll, "\t", proposedll, "\t", ll, "\t", exp(ll))
          seqpair = current_sample.seqpair
          #println(getalignment(seqpair.seq1, simple_sample.align1))
          #println(getalignment(seqpair.seq2, simple_sample.align2))
          #println(getalignment(seqpair.seq1, current_sample.align1))
          #println(getalignment(seqpair.seq2, current_sample.align2))

          movename = "gibbs_alignment"
          if exp(ll) > a
            current_sample = newsample
            currentll, dummy = tkf92(0, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
            logAccept!(logger, movename)
          else
            logReject!(logger, movename)
          end
          =#

    elseif move >= 4
      movename = ""
      propratio = 0.0
      if move == 4
        proposed.lambda = current.lambda + randn(rng)*0.2
        movename = "lambda"
      elseif move == 5
        proposed.ratio = current.ratio + randn(rng)*0.05
        movename = "ratio"
      elseif move == 6
        proposed.r = current.r + randn(rng)*0.06
        movename = "r"
      elseif move == 7
        sigma = 0.01
        movename = "t0.01"
        if rand(rng) < 0.30
          sigma = 0.1
          movename = "t0.1"
        end
        d1 = Truncated(Normal(current.t, sigma), 0.0, Inf)
        d2 = Truncated(Normal(proposed.t, sigma), 0.0, Inf)
        proposed.t = rand(d1)
        propratio = logpdf(d2, current.t) - logpdf(d1, proposed.t)

      end

      proposed.mu = proposed.lambda/proposed.ratio
      if(proposed.lambda > 0.0 && proposed.mu > 0.0 && proposed.lambda < proposed.mu && 0.0 < proposed.r < 1.0 && proposed.t > 0.0)
        proposedll, proposed_samples = tkf92(nsamples, rng, seqpair, proposed, modelparams, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)

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

    if i >= burnin && (currentiter+1) % samplerate == 0
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

  expll = logsumexp(logll+log(1/Float64(length(logll))))

  return citer+niter, current, samples, expll
end

function mlalignmentopt(seqpair::SequencePair, modelparams::ModelParameters, cornercut::Int, initialParams::PairParameters, maxiter::Int=50)
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
  localObjectiveFunction = ((param, grad) -> tkf92(0, MersenneTwister(330101840810391), seqpair, PairParameters(param), modelparams, cornercut, fixAlignment, align1, align2)[1])
  opt = Opt(:LN_COBYLA, 4)
  lower_bounds!(opt, ones(Float64, 4)*1e-10)

  upper = Float64[1e10, 1.0, 0.999, 1e10]
  upper_bounds!(opt, upper)

  xtol_rel!(opt,1e-4)
  maxeval!(opt, maxiter)
  max_objective!(opt, localObjectiveFunction)

  initial = Float64[initialParams.lambda, initialParams.ratio, initialParams.r, initialParams.t]
  (maxf,maxx,ret) = optimize(opt, initial)
  println(maxf,maxx)
  return PairParameters(maxx)
end

function mlalign()
  pairs = load_sequences("data/holdout_data.txt")

  srand(98418108751401)
  rng = MersenneTwister(242402531025555)
  modelfile = "models/pairhmm16_noswitching_n128.jls"

  cornercut = 75

  ser = open(modelfile,"r")
  modelparams::ModelParameters = deserialize(ser)
  close(ser)
  prior = modelparams.prior
  obsnodes = modelparams.obsnodes
  hmminitprobs = modelparams.hmminitprobs
  hmmtransprobs = modelparams.hmmtransprobs
  numHiddenStates = length(hmminitprobs)

  println("use_switching", obsnodes[1].useswitching)
  println("H=", numHiddenStates)

  mask = Int[OBSERVED_DATA, OBSERVED_DATA, OBSERVED_DATA, MISSING_DATA]

  seq1, seq2 = masksequences(pairs[1].seq1, pairs[1].seq2, mask)
  #seq1, seq2 = masksequences(pairs[1].seq1, Sequence("ATG"), mask)
  seqpair = SequencePair(0,seq1, seq2)
  #=
  mlparams = mlalignmentopt(seqpair, modelparams, cornercut)
  #tkf92(nsamples::Int, rng::AbstractRNG, seqpair::SequencePair, pairparams::PairParameters, modelparams::ModelParameters, cornercut::Int=10000000, fixAlignment::Bool=false, align1::Array{Int,1}=zeros(Int,1), align2::Array{Int,1}=zeros(Int,1), fixStates::Bool=false, states::Array{Int,1}=zeros(Int,1), partialAlignment::Bool=false, samplealignments::Bool=true)
  ll, mlsample = tkf92(0, rng, seqpair, mlparams, modelparams, cornercut, false, zeros(Int,1), zeros(Int,1),false,zeros(Int,1),false,false)
  =#
  ll, mlsample = mlalignment(seqpair, modelparams, cornercut)
  println(getalignment(seqpair.seq1, mlsample.align1))
  println(getalignment(seqpair.seq2, mlsample.align2))
  println(mlsample.states)
end

function mlalignment(seqpair::SequencePair, modelparams::ModelParameters, cornercut::Int, initialParams::PairParameters)
  mlparams = mlalignmentopt(seqpair, modelparams, cornercut, initialParams, 3)
  ll, mlsample = tkf92(0, MersenneTwister(242402531025555), seqpair, mlparams, modelparams, cornercut, false, zeros(Int,1), zeros(Int,1),false,zeros(Int,1),false,false)
  mlsample.params = mlparams
  println(getalignment(seqpair.seq1, mlsample.align1))
  println(getalignment(seqpair.seq2, mlsample.align2))
  println(mlsample.states)
  return ll, mlsample
end

type MCMCResult
  ll::Float64
  obscount::Int
  samples::Array{SequencePairSample,1}
  current_sample::SequencePairSample

  function MCMCResult()
    samplell = 0.0
    obscount = 0
    samples = SequencePairSample[]
    current_samples = SequencePairSample()
    new(samplell,obscount,samples,current_samples)
  end
end

function parallelmcmc(startk::Int, endk::Int, currentiter::Int, mcmciter::Int, samplerate::Int, obsnodes::Array{ObservationNode, 1}, rng::AbstractRNG, current_samples::Array{SequencePairSample,1}, prior::PriorDistribution, hmminitprobs::Array{Float64,1}, hmmtransprobs::Array{Float64,2}, cornercut::Int)
  result = MCMCResult()
  for k=startk:endk
    it, newparams, ksamples, expll = mcmc_sequencepair(currentiter, mcmciter, samplerate, MersenneTwister(abs(rand(Int))), SequencePairSample(current_samples[k]), deepcopy(modelparams), cornercut)
    push!(result.current_samples, ksamples[end])
    push!(result.samplell, expll)
    push!(result.obscount, pairs[k].seq1.length + pairs[k].seq2.length)
    for ks in ksamples
      push!(result.samples, ks)
    end
  end
  return result
end

function train()

  srand(98418108751401)
  rng = MersenneTwister(242402531025555)

  maxiters = 100000
  cornercutinit = 10
  cornercut = 75
  useswitching = true
  useparallel = true
  fixInputAlignments = false
  #pairs = shuffle!(rng, load_sequences("data/data.txt"))
  #pairs = shuffle!(rng, load_sequences("data/data_diverse.txt"))
  inputsamples = shuffle!(rng, load_sequences_and_alignments("data/data.txt"))
  #inputsamples = shuffle!(rng, load_sequences_and_alignments("data/data_diverse.txt"))[1:50]
  pairs = SequencePair[sample.seqpair for sample in inputsamples]

  println("N=",length(pairs))
  println("N=",length(pairs))

  mcmciter = 100
  samplerate = 20

  numHiddenStates = 8

  freeParameters = 6*numHiddenStates + (numHiddenStates-1) + (numHiddenStates*numHiddenStates - numHiddenStates) + numHiddenStates*19
  if useswitching
    freeParameters = 12*numHiddenStates + (numHiddenStates-1) + (numHiddenStates*numHiddenStates - numHiddenStates) + numHiddenStates*38 + numHiddenStates*2
  end

  loadModel = false
  if useswitching
    modelfile = string("models/pairhmm",numHiddenStates,"_switching_n",length(pairs),".jls")
  else
    modelfile = string("models/pairhmm",numHiddenStates,"_noswitching_n",length(pairs),".jls")
  end



  hmminitprobs = ones(Float64,numHiddenStates)/Float64(numHiddenStates)
  hmmtransprobs = ones(Float64, numHiddenStates, numHiddenStates)/Float64(numHiddenStates)
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
      set_parameters(obsnodes[h].switching.diffusion_r1, 0.1, rand()*2.0*pi - pi, 1.0, 0.1, rand()*2.0*pi - pi, 1.0, 1.0, 1.0)
      set_parameters(obsnodes[h].switching.diffusion_r2, 0.1, rand()*2.0*pi - pi, 1.0, 0.1, rand()*2.0*pi - pi, 1.0, 1.0, 1.0)
      obsnodes[h].switching.alpha = 5.0 + 20.0*rand(rng)
      obsnodes[h].switching.pi_r1 = rand(rng)
    else
      set_parameters(obsnodes[h].aapairnode, v, 1.0)
      set_parameters(obsnodes[h].diffusion, 0.1, rand()*2.0*pi - pi, 1.0, 0.1, rand()*2.0*pi - pi, 1.0, 1.0, 1.0)
    end
  end

  modelparams::ModelParameters = ModelParameters(prior, obsnodes, hmminitprobs, hmmtransprobs)
  if loadModel
    modelparams = readmodel(modelfile)
    prior = modelparams.prior
    obsnodes = modelparams.obsnodes
    hmminitprobs = modelparams.hmminitprobs
    hmmtransprobs = modelparams.hmmtransprobs
    numHiddenStates = length(hmminitprobs)
  end

  tic()

  current_samples = SequencePairSample[]
  if useparallel
    refs = RemoteRef[]
    for k=1:length(pairs)
      if !inputsamples[k].aligned
        ref = @spawn tkf92(1, MersenneTwister(abs(rand(Int))), pairs[k], PairParameters(), deepcopy(modelparams), cornercutinit)
        push!(refs,ref)
      end
    end
    index = 1
    for k=1:length(pairs)
      if !inputsamples[k].aligned
        ref = refs[index]
        index += 1
        res = fetch(ref)
        push!(current_samples, res[2][1])
      else
        push!(current_samples, inputsamples[k])
        inputsamples[k].states = zeros(Int, length(inputsamples[k].align1))
        for j=1:length(inputsamples[k].states)
          inputsamples[k].states[j] = rand(1:numHiddenStates)
        end
      end
    end
  else
    for k=1:length(pairs)
      println(pairs[k].id)
      if !inputsamples[k].aligned
        push!(current_samples, tkf92(1, rng, pairs[k], PairParameters(), modelparams, cornercutinit)[2][1])
      else
        push!(current_samples, inputsamples[k])
        inputsamples[k].states = zeros(Int, length(inputsamples[k].align1))
        for j=1:length(inputsamples[k].states)
          inputsamples[k].states[j] = rand(1:numHiddenStates)
        end
      end
    end
  end
  toc()

  println("Initialised")

  mlwriter = open(string("logs/ml",numHiddenStates,".log"), "w")
  write(mlwriter, "iter\tll\tcount\tavgll\tnumFreeParameters\tAIC\tlambda_shape\tlambda_scale\tmu_shape\tmu_scale\tr_alpha\tr_beta\tt_shape\tt_scale\n")


  currentiter = 0

  for i=1:maxiters
    println("ITER=",i)
    samples = SequencePairSample[]
    samplell = Float64[]
    obscount = Int[]
    tic()

    if useparallel

      refs = RemoteRef[]
      #=for k=1:length(pairs)
        ref = @spawn parallelmcmc(currentiter, mcmciter, samplerate, MersenneTwister(abs(rand(Int))), SequencePairSample(current_samples[k]), deepcopy(modelparams), cornercut)
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
      end=#

      for k=1:length(pairs)
        ref = @spawn mcmc_sequencepair(currentiter, mcmciter, samplerate, MersenneTwister(abs(rand(Int))), SequencePairSample(current_samples[k]), deepcopy(modelparams), cornercut, fixInputAlignments && inputsamples[k].aligned)
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
    else
      for k=1:length(pairs)
        it, newparams, ksamples, expll = mcmc_sequencepair(currentiter, mcmciter, samplerate,  MersenneTwister(abs(rand(Int))), current_samples[k], modelparams, cornercut, fixInputAlignments && inputsamples[k].aligned)
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
      for h=1:numHiddenStates
        samplescopy = SequencePairSample[deepcopy(s) for s in samples]
        ref = @spawn switchopt(h, samplescopy, deepcopy(obsnodes))
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
          samplescopy = SequencePairSample[deepcopy(s) for s in samples]
          ref = @spawn mlopt(h, samplescopy, deepcopy(obsnodes))
          push!(refs, ref)
        end
        for h=1:numHiddenStates
          params = fetch(refs[h])
          set_parameters(obsnodes[h].aapairnode, params[1], 1.0)
          dopt = params[2]
          set_parameters(obsnodes[h].diffusion, dopt[1], mod2pi(dopt[2]+pi)-pi, dopt[3], dopt[4], mod2pi(dopt[5]+pi)-pi, dopt[6], 1.0, dopt[7])
        end
      else
        for h=1:numHiddenStates
          params = mlopt(h, samples,deepcopy(obsnodes))
          set_parameters(obsnodes[h].aapairnode, params[1], 1.0)
          dopt = params[2]
          set_parameters(obsnodes[h].diffusion, dopt[1], mod2pi(dopt[2]+pi)-pi, dopt[3], dopt[4], mod2pi(dopt[5]+pi)-pi, dopt[6], 1.0, dopt[7])
        end
      end
    end


    hmminitprobs, hmmtransprobs = hmmopt(samples,numHiddenStates)
    prior = prioropt(samples, prior)
    mstep_elapsed = toc()


    println(hmminitprobs,"\n",hmmtransprobs)
    println(length(samples))

    println("E-step time = ", estep_elapsed)
    println("M-step time = ", mstep_elapsed)


    write(mlwriter, string(i-1,"\t",sum(samplell), "\t", sum(obscount), "\t", sum(samplell)/sum(obscount),"\t", freeParameters,"\t", aic(sum(samplell), freeParameters), "\t", join(prior.params,"\t"), "\n"))
    flush(mlwriter)


    modelparams = ModelParameters(prior, obsnodes, hmminitprobs, hmmtransprobs)
    ser = open(modelfile,"w")
    serialize(ser, modelparams)
    close(ser)

    for h=1:numHiddenStates
      println(modelparams.obsnodes[h].diffusion)
    end


    write_hiddenstates(modelparams, "logs/hiddenstates.txt")
  end
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
   # println(h, x0, xt, phi0,phit,psi0,psit, t)
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

function bin2d(phi::Array{Float64, 1}, psi::Array{Float64, 1})
  nbins = 20
  for (a,b) in zip(phi,psi)
    c = int((a / float(pi))*nbins)
    d = int((b / float(pi))*nbins)
    println(c,"\t",d)
  end
end

using Gadfly
using Compose
using Cairo
function test()
  #pairs = load_sequences_and_alignments("data/data.txt")
  pairs = load_sequences_and_alignments("data/holdout_data.txt")

  srand(98418108751401)
  rng = MersenneTwister(242402531025555)
  #modelfile = "models/pairhmm4_switching.jls"
  #modelfile = "models/pairhmm8_noswitching.jls"
  #modelfile = "models/pairhmm12_noswitching.jls"
  #modelfile = "models/pairhmm8_noswitching.jls"

  modelfile = "models/pairhmm8_switching_n128.jls"
  outputdir = "logs/pairhmm8_switching_n128/"

  fixAlignment = false
  cornercut = 75

  ser = open(modelfile,"r")
  modelparams::ModelParameters = deserialize(ser)
  close(ser)
  prior = modelparams.prior
  obsnodes = modelparams.obsnodes
  hmminitprobs = modelparams.hmminitprobs
  hmmtransprobs = modelparams.hmmtransprobs
  numHiddenStates = length(hmminitprobs)

  write_hiddenstates(modelparams, "myhiddenstates.txt")

  println("use_switching", obsnodes[1].useswitching)
  println("H=", numHiddenStates)

  mkpath(outputdir)
  outfile = open(string(outputdir, "benchmarks",numHiddenStates,".txt"), "w")
  write(outfile, "mask\tphi_homologue\tpsi_homologue\tphi_predicted\tpsi_predicted\n")

  mask = Int[OBSERVED_DATA, OBSERVED_DATA, OBSERVED_DATA, MISSING_DATA]
  for k=1:length(pairs)
    inputalign1 = pairs[k].align1
    inputalign2 = pairs[k].align2
    input = pairs[k].seqpair
    seq1, seq2 = masksequences(input.seq1, input.seq2, mask)
    masked = SequencePair(0,seq1, seq2)
    current_sample = tkf92(1, rng, masked, PairParameters(), modelparams, cornercut, true, inputalign1, inputalign2)[2][1]
    current_sample.seqpair.id = k
    #mcmc_sequencepair(citer::Int, niter::Int, samplerate::Int, rng::AbstractRNG, initialSample::SequencePairSample, modelparams::ModelParameters, cornercut::Int=100, fixAlignment::Bool=false, writeoutput::Bool=false, outputdir::AbstractString="")
    ret = mcmc_sequencepair(0, 2000, 1, rng, current_sample, modelparams, cornercut, fixAlignment, true, outputdir)

    samples = ret[3]
    nsamples = length(ret[3])
    burnin = int(max(1,nsamples/2))
    samples = samples[burnin:end]
    filled_pairs = [sample_missing_values(rng, obsnodes, sample) for sample in samples]

    mpdalign1, mpdalign2, posterior_probs = mpdalignment(samples)

    #ll, mlalign = mlalignment(masked, modelparams, cornercut, samples[end].params)
    #filled_pairs = [sample_missing_values(rng, obsnodes, deepcopy(mlalign)) for sample in samples]

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

      conf1, conf2 = getconfigurations(mpdalign1, mpdalign2)

      push!(phi, angular_mean(phi_i))
      push!(psi, angular_mean(psi_i))
      if input.seq2.phi[i] > -100.0 && input.seq2.psi[i] > -100.0
        xvals = Float64[]
        yvals = Float64[]
        labels = AbstractString[]

        push!(xvals, pimod(angular_mean(phi_i)))
        push!(yvals, pimod(angular_mean(psi_i)))
        push!(labels, "P")
        if conf2[i] > 0
          push!(xvals, input.seq1.phi[conf2[i]])
          push!(yvals, input.seq1.psi[conf2[i]])
          l1 = string("A (", aminoacids[input.seq1.seq[conf2[i]]],")")
          push!(labels, l1)
        end
        push!(xvals, input.seq2.phi[i])
        push!(yvals, input.seq2.psi[i])
        l2 = string("B (", aminoacids[input.seq2.seq[i]],")")
        push!(labels, l2)

        #println(phi_i)
        #println(psi_i)
        #println(phi)
        #println(psi)

        outputdir2  = string(outputdir,"structure_",k,"/")
        mkpath(outputdir2)
        p = plot(layer(x=xvals, y=yvals, label=labels, Geom.label), layer(x=xvals, y=yvals, Geom.point), layer(x=phi_i, y=psi_i, Geom.histogram2d(xbincount=30, ybincount=30)), Coord.Cartesian(xmin=Float64(-pi), xmax=Float64(pi), ymin=Float64(-pi), ymax=Float64(pi)))
        draw(SVG(string(outputdir2,"hist",i,".svg"), 5inch, 5inch), p)
        #draw(PNG(string(outputdir2,"hist",i,".png"), 5inch, 5inch), p)
      end
    end


    align1 = inputalign1
    align2 = inputalign2

    #=
      sumphi = Float64[]
      sumpsi = Float64[]
      for sample in filled_pairs
        push!(sumphi, angular_rmsd(sample.seq2.phi, phi))
        push!(sumpsi, angular_rmsd(sample.seq2.psi, psi))
      end
      println("TTT", sqrt(sum(sumphi .^ 2.0)/length(sumphi)),"\t", sqrt(sum(sumpsi .^ 2.0)/length(sumpsi)))
      =#



    for (a,b) in zip(align2, align1)
      if a > 0 && b > 0
        #println(input.seq2.psi[a],"\t", input.seq1.psi[b], "\t", pimod(psi[a]))
      end
    end

    println("Homologue:\tphi=", angular_rmsd(input.seq2.phi, input.seq1.phi, align2, align1),"\tpsi=", angular_rmsd(input.seq2.psi, input.seq1.psi, align2, align1))
    println("Predicted:\tpsi=", angular_rmsd(input.seq2.phi, phi), "\tpsi=", angular_rmsd(input.seq2.psi, psi))
    write(outfile, join(mask, ""), "\t", string(angular_rmsd(input.seq2.phi, input.seq1.phi, align2, align1)), "\t", string(angular_rmsd(input.seq2.psi, input.seq1.psi, align2, align1)), "\t")
    write(outfile, string(angular_rmsd(input.seq2.phi, phi)), "\t", string(angular_rmsd(input.seq2.psi, psi),"\n"))
    flush(outfile)
  end
  close(outfile)
end


  #@profile train()
  #profilewriter = open("profile.log", "w")
  #Profile.print(profilewriter)

  # TODO ML alignment
  # sampling
  # hidden state conditioning
#end
