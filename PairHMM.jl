using Formatting
using Distributions
using DataStructures
using UtilsModule
using NodesModule

#include("Cornercut.jl")
#include("StatisticalAlignmentHMM.jl")
using Cornercut


using AlignmentHMM

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

function get_alignment_transition_probabilities(lambda::Float64, mu::Float64, r::Float64, t::Float64)
  Bt = (1.0 - exp((lambda-mu)*t))/(mu - lambda*exp((lambda-mu)*t))

  expmut = exp(-mu*t)
  aligntransprobs = zeros(Float64, 5, 5)
  aligntransprobs[START,YINSERT] = lambda*Bt
  aligntransprobs[START,END] = (1.0 - lambda*Bt)*(1.0 - (lambda/mu))
  aligntransprobs[START,MATCH] = (1.0 - lambda*Bt)*((lambda/mu)*expmut)
  aligntransprobs[START,XINSERT] = (1.0 - lambda*Bt)*((lambda/mu)*(1.0 - expmut))

  aligntransprobs[MATCH,MATCH] = r + (1.0-r)*(1.0 - lambda*Bt)*((lambda/mu)*expmut)
  aligntransprobs[MATCH,YINSERT] = (1.0-r)*lambda*Bt
  aligntransprobs[MATCH,END] = (1.0-r)*(1.0 - lambda*Bt)*(1.0 - (lambda/mu))
  aligntransprobs[MATCH,XINSERT] = (1.0-r)*(1.0 - lambda*Bt)*((lambda/mu)*(1.0 - expmut))

  aligntransprobs[XINSERT,XINSERT] = r + (1.0-r)*((mu*Bt)/(1.0-expmut))*((lambda/mu)*(1.0 - expmut))
  aligntransprobs[XINSERT,YINSERT] = (1.0-r)*((1.0 - mu*Bt - expmut)/(1.0-expmut))
  aligntransprobs[XINSERT,END] = (1.0-r)*((mu*Bt)/(1.0-expmut))*(1.0 - (lambda/mu))
  aligntransprobs[XINSERT,MATCH] = (1.0-r)*((mu*Bt)/(1.0-expmut))*((lambda/mu)*expmut)

  aligntransprobs[YINSERT,YINSERT] = r + (1.0-r)*(lambda*Bt)
  aligntransprobs[YINSERT,END] = (1.0-r)*(1.0-lambda*Bt)*(1.0 - (lambda/mu))
  aligntransprobs[YINSERT,MATCH] = (1.0-r)*(1.0-lambda*Bt)*((lambda/mu)*expmut)
  aligntransprobs[YINSERT,XINSERT] = (1.0-r)*(1.0-lambda*Bt)*((lambda/mu)*(1.0 - expmut))

  aligntransprobs[END,END] = 0.0

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
end

function getvalue(cache::HMMCache, i::Int, j::Int, alignnode::Int, h::Int)
  key::Int = i*(cache.m+1)*9 + j*9 + (alignnode-1) + 1
  if(haskey(cache.caches[h],key))
    return cache.caches[h][key]
  else
    return Inf
  end
end
#=
type DatalikCache
  caches::Array{Dict{Int, Float64},1}
  n::Int
  m::Int
  numHiddenStates::Int
  t::Float64
  seq1cache::Array{Float64,2}
  seq2cache::Array{Float64,2}

  function DatalikCache(n::Int, m::Int, numHiddenStates::Int)
    caches = Dict{Int,Float64}[]
    for h=1:numHiddenStates
      d = Dict{Int,Float64}()
      push!(caches, d)
    end
    new(caches, n,m,numHiddenStates, -1.0, ones(Float64, n, numHiddenStates)*Inf, ones(Float64, m, numHiddenStates)*Inf)
  end
end

function get_data_lik_cache(cache::DatalikCache, obsnodes::Array{ObservationNode,1}, h::Int, seq1::Sequence, seq2::Sequence, k::Int, l::Int, t::Float64)
  i = k
  j = l
  s1 = seq1
  s2 = seq2
  if i > j
    i,j = j,i
    s1,s2 = s2,s1
  end
  key::Int = i*(cache.m+1)*9 + j*9
  if cache.t != t || !haskey(cache.caches[h],key)
    if cache.t != t
      for k=1:cache.numHiddenStates
        cache.caches[k] = Dict{Int,Float64}()
      end
    end
    cache.t = t
    datalik = get_data_lik(obsnodes[h], s1, s2, i,j, t)
    cache.caches[h][key] = datalik
    return datalik
  else
    return cache.caches[h][key]
  end
end

function get_data_lik_cache_seq1(cache::DatalikCache, obsnodes::Array{ObservationNode,1}, h::Int, seq1::Sequence, i::Int)
  if cache.seq1cache[i,h] == Inf
    cache.seq1cache[i,h] = get_data_lik_x0(obsnodes[h], seq1, i, 1.0)
    return cache.seq1cache[i,h]
  else
    return cache.seq1cache[i,h]
  end
end

function get_data_lik_cache_seq2(cache::DatalikCache, obsnodes::Array{ObservationNode,1}, h::Int, seq1::Sequence, i::Int)
  if cache.seq2cache[i,h] == Inf
    cache.seq2cache[i,h] = get_data_lik_xt(obsnodes[h], seq1, i, 1.0)
    return cache.seq2cache[i,h]
  else
    return cache.seq2cache[i,h]
  end
end

function tkf92(datalikcache::DatalikCache, nsamples::Int, rng::AbstractRNG, seqpair::SequencePair, pairparams::PairParameters, modelparams::ModelParameters, cornercutin::Int=10000000, fixAlignment::Bool=false, align1::Array{Int,1}=zeros(Int,1), align2::Array{Int,1}=zeros(Int,1), fixStates::Bool=false, states::Array{Int,1}=zeros(Int,1), partialAlignment::Bool=false, samplealignments::Bool=true)
  cornercut = cornercutin
  aligntransprobs = get_alignment_transition_probabilities(pairparams.lambda,pairparams.mu,pairparams.r,pairparams.t)
  obsnodes = modelparams.obsnodes
  numHiddenStates = modelparams.numHiddenStates

  n = seqpair.seq1.length
  m = seqpair.seq2.length

  starti = 0
  endi = 0
  startj = 0
  endj  = 0
  if partialAlignment && fixAlignment && !fixStates
    width = 100
    #width= rand(50:150)
    starti = rand(1:max(1,n-width))
    endi = starti+width

    #startj = starti + rand(-75:75)
    #endj = startj+width
    startj = rand(1:max(1,m-width))
    endj = startj+width
  end

  cache = HMMCache(n,m,modelparams.numHiddenStates,cornercut, fixAlignment, fixStates)

  choice = Array(Float64, modelparams.numHiddenStates)
  alignmentpath = getalignmentpath(n,m,align1, align2,states)
  states1 = zeros(Int,1)
  states2 = zeros(Int,1)
  if fixStates
    states1, states2 = getstates(states,align1,align2)
  end

  temph = Array(Float64, modelparams.numHiddenStates)

  hmmparameters = HMMParameters(aligntransprobs, modelparams.hmminitprobs, modelparams.hmmtransprobs)
  if !fixAlignment
    len = min(n,m)
    for i=1:len
      tkf92forward(temph, datalikcache, obsnodes, seqpair, pairparams.t, cache, hmmparameters,i,i,END,1, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj,states1, states2)
    end
    for i=1:n
      tkf92forward(temph, datalikcache, obsnodes, seqpair, pairparams.t, cache, hmmparameters,i,m,END,1, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj,states1, states2)
    end
  end

  for h=1:numHiddenStates
    choice[h] = tkf92forward(temph, datalikcache, obsnodes, seqpair, pairparams.t, cache, hmmparameters,n,m,END,h, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj,states1, states2)
  end
  sum = logsumexp(choice)

  if samplealignments
    samples = SequencePairSample[]
    if sum > -1e8
      if nsamples > 0
        choice = exp(choice - sum)
        for i=1:nsamples
          pairsample = SequencePairSample(seqpair, pairparams)
          tkf92sample(datalikcache, obsnodes, seqpair, pairparams.t, rng,cache, hmmparameters,n,m, END, UtilsModule.sample(rng, choice), pairsample.align1,pairsample.align2, pairsample.states, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj,states1, states2)
          push!(samples, pairsample)
        end
      end
    else
      println(seqpair.id, "WOAAAH","\t", sum)
      return -Inf, samples
    end

    ll = logprior(modelparams.prior, pairparams)+sum

    return ll,samples
  else
    mlsample = SequencePairSample(seqpair, pairparams)
    tkf92viterbi(obsnodes, seqpair, pairparams.t, rng,cache, hmmparameters,n,m, END, indmax(choice), mlsample.align1,mlsample.align2, mlsample.states, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)
    ll = logprior(modelparams.prior, pairparams)+sum
    return ll,mlsample
  end
end




function tkf92sample(datalikcache::DatalikCache, obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, rng::AbstractRNG, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, align1::Array{Int,1}, align2::Array{Int,1}, states::Array{Int,1}, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1), starti::Int=0, endi::Int=0, startj::Int=0, endj::Int=0, states1::Array{Int,1}=zeros(Int,1), states2::Array{Int,1}=zeros(Int,1))
  newalignnode::Int = alignnode
  newh::Int = h
  newi::Int = i
  newj::Int = j

  numAlignStates::Int = size(hmmparameters.aligntransprobs,1)
  numHiddenStates::Int = size(hmmparameters.hmmtransprobs,1)
  temph = Array(Float64, numHiddenStates)

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
          ll =  tkf92forward(temph, datalikcache, obsnodes, seqpair, t, cache, hmmparameters,newi,newj, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj,states1, states2)+log(transprob)
          choice[(prevalignnode-1)*numHiddenStates + prevh] = ll
        end
      end
    end

    s = GumbelSample(rng, choice)
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
end=#

function tkf92viterbi(obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, rng::AbstractRNG, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, align1::Array{Int,1}, align2::Array{Int,1}, states::Array{Int,1}, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1), starti::Int=0, endi::Int=0, startj::Int=0, endj::Int=0)
  newalignnode::Int = alignnode
  newh::Int = h
  newi::Int = i
  newj::Int = j

  numAlignStates::Int = size(hmmparameters.aligntransprobs,1)
  numHiddenStates::Int = size(hmmparameters.hmmtransprobs,1)
  temph = Array(Float64, numHiddenStates)

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
          ll =  tkf92forward(temph, datalikcache, obsnodes, seqpair, t, cache, hmmparameters,newi,newj, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj)+log(transprob)
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
#=
function tkf92forward(temph::Array{Float64,1}, datalikcache::DatalikCache, obsnodes::Array{ObservationNode,1}, seqpair::SequencePair, t::Float64, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, h::Int, fixAlignment::Bool=false, cornercut::Int=10000000, fixStates::Bool=false, alignmentpath::SparseMatrixCSC=spzeros(Int, 1, 1), starti::Int=0, endi::Int=0, startj::Int=0, endj::Int=0, states1::Array{Int,1}=zeros(Int,1), states2::Array{Int,1}=zeros(Int,1))
  if i < 0 || j < 0
    return -Inf
  end

  v = getvalue(cache,i,j,alignnode,h)
  if v != Inf
    return v
  end

  if abs(i-j) > cache.cornercut
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
        datalik = get_data_lik_cache(datalikcache, obsnodes, h, seqpair.seq1, seqpair.seq2, i, j, t)
      end
    elseif alignnode == XINSERT
      if i > 0
        datalik = get_data_lik_cache_seq1(datalikcache, obsnodes, h, seqpair.seq1, i)
      end
    elseif alignnode == YINSERT
      if j > 0
        datalik = get_data_lik_cache_seq2(datalikcache, obsnodes, h, seqpair.seq2, j)
      end
    end

    for prevalignnode=1:5
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
              prevlik = tkf92forward(temph, datalikcache, obsnodes, seqpair, t, cache, hmmparameters, i-1, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj, states1, states2)
            elseif alignnode == XINSERT
              prevlik = tkf92forward(temph, datalikcache, obsnodes, seqpair, t, cache, hmmparameters, i-1, j, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj, states1, states2)
            elseif alignnode == YINSERT
              prevlik = tkf92forward(temph, datalikcache, obsnodes, seqpair, t, cache, hmmparameters, i, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj, states1, states2)
            end
            sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
          end
        else
          if alignnode == MATCH
            for prevh=1:numHiddenStates
              prevlik = tkf92forward(temph, datalikcache, obsnodes, seqpair, t, cache, hmmparameters, i-1, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj, states1, states2)
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          elseif alignnode == XINSERT
            for prevh=1:numHiddenStates
              prevlik = tkf92forward(temph, datalikcache, obsnodes, seqpair, t, cache, hmmparameters, i-1, j, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj, states1, states2)
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          elseif alignnode == YINSERT
            for prevh=1:numHiddenStates
              prevlik = tkf92forward(temph, datalikcache, obsnodes, seqpair, t, cache, hmmparameters, i, j-1, prevalignnode, prevh, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj, states1, states2)
              sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+hmmparameters.loghmmtransprobs[prevh, h]+datalik)
            end
          end
        end
      end
    end
  else
    for prevalignnode=1:5
      if hmmparameters.aligntransprobs[prevalignnode, alignnode] > 0.0
        prevlik =  tkf92forward(temph, datalikcache, obsnodes, seqpair, t, cache, hmmparameters, i, j, prevalignnode, h, fixAlignment, cornercut, fixStates, alignmentpath, starti, endi, startj, endj, states1, states2)
        sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode])
      end
    end
  end

  putvalue(cache,i,j,alignnode,h,sum)
  return sum
end=#

function hmmsample(datalikcache::AlignmentHMM.DatalikCache, rng::AbstractRNG, current_sample::SequencePairSample, modelparams::ModelParameters)
  current = deepcopy(current_sample)
  obsnodes = modelparams.obsnodes
  numHiddenStates = modelparams.numHiddenStates
  hmminitprobs = modelparams.hmminitprobs
  hmmtransprobs = modelparams.hmmtransprobs
  loghmminitprobs = log(modelparams.hmminitprobs)
  loghmmtransprobs = log(modelparams.hmmtransprobs)
  seqpair = current.seqpair
  t = current.params.t
  len = length(current.align1)
  align1 = current.align1
  align2 = current.align2

  hmm = zeros(Float64, len, numHiddenStates)
  dataloglik = zeros(Float64, numHiddenStates)
  for a=1:len
    i = align1[a]
    j = align2[a]

    maxll = -Inf
    for h=1:numHiddenStates
      if i > 0 && j > 0
        dataloglik[h] = AlignmentHMM.get_data_lik_cache(datalikcache, obsnodes, h, seqpair.seq1, seqpair.seq2, i, j, t)
      elseif i > 0
        dataloglik[h] = AlignmentHMM.get_data_lik_cache_seq1(datalikcache, obsnodes, h, seqpair.seq1, i)
      elseif j > 0
        dataloglik[h] = AlignmentHMM.get_data_lik_cache_seq2(datalikcache, obsnodes, h, seqpair.seq2, j)
      end
      maxll = max(maxll, dataloglik[h])
    end

    sumlik = 0.0
    for h=1:numHiddenStates
      hmm[a,h] = 0.0
      v = dataloglik[h]-maxll
      if v > -20.0
        datalik = exp(v)
        if a == 1
          hmm[a,h] = hmminitprobs[h]*datalik
        else
          for prevh=1:numHiddenStates
            hmm[a,h] += hmm[a-1,prevh] * hmmtransprobs[prevh,h] * datalik
          end
        end
        sumlik += hmm[a,h]
      end
    end

    for h=1:numHiddenStates
      hmm[a,h] /= sumlik
    end
  end

  nexth = UtilsModule.sample(rng, hmm[len,:])
  current.states[len] = nexth
  choice = zeros(Float64, numHiddenStates)
  for b=1:(len-1)
    a = len - b
    for h=1:numHiddenStates
      choice[h] = hmm[a,h] * hmmtransprobs[h,nexth]
    end
    nexth = UtilsModule.sample(rng, choice)
    current.states[a] = nexth
  end

  return current
end



function sampleobs(rng::AbstractRNG, current_sample::SequencePairSample, modelparams::ModelParameters)
  vmerror1 = current_sample.seqpair.seq1.error_distribution
  vmerror2 = current_sample.seqpair.seq2.error_distribution
  width = 0.04

  for a=1:length(current_sample.align1)
    currentobsll = computeobsll(current_sample, modelparams.obsnodes, a)

    for l=1:15
      i = current_sample.align1[a]
      j = current_sample.align2[a]
      h = current_sample.states[a]
      if i !=0
        oldphi1 = current_sample.seqpair.seq1.phi[i]
        oldpsi1 = current_sample.seqpair.seq1.psi[i]
        if current_sample.seqpair.seq1.phi_obs[i] > -100.0
          current_sample.seqpair.seq1.phi[i] += randn(rng)*width
          current_sample.seqpair.seq1.phi[i] = mod2pi(current_sample.seqpair.seq1.phi[i]+pi)-pi
        end
        if current_sample.seqpair.seq1.psi_obs[i] > -100.0
          current_sample.seqpair.seq1.psi[i] += randn(rng)*width
          current_sample.seqpair.seq1.psi[i] = mod2pi(current_sample.seqpair.seq1.psi[i]+pi)-pi
        end
      end

      if j !=0
        oldphi2 = current_sample.seqpair.seq2.phi[j]
        oldpsi2 = current_sample.seqpair.seq2.psi[j]
        if current_sample.seqpair.seq2.phi_obs[j] > -100.0
          current_sample.seqpair.seq2.phi[j] += randn(rng)*width
          current_sample.seqpair.seq2.phi[j] = mod2pi(current_sample.seqpair.seq2.phi[j]+pi)-pi
        end
        if current_sample.seqpair.seq2.psi_obs[j] > -100.0
          current_sample.seqpair.seq2.psi[j] += randn(rng)*width
          current_sample.seqpair.seq2.psi[j] = mod2pi(current_sample.seqpair.seq2.psi[j]+pi)-pi
        end
      end

      proposedobsll = computeobsll(current_sample, modelparams.obsnodes, a)
      if exp(proposedobsll-currentobsll) > rand(rng)
        currentobsll = proposedobsll
      else
        if i !=0
          current_sample.seqpair.seq1.phi[i] = oldphi1
          current_sample.seqpair.seq1.psi[i] = oldpsi1
        end

        if j !=0
          current_sample.seqpair.seq2.phi[j] = oldphi2
          current_sample.seqpair.seq2.psi[j] = oldpsi2
        end
      end
    end
  end

  return current_sample
end


export MCMCLogger
type MCMCLogger
    dict::Dict{AbstractString, Array{Float64,1}}

    function MCMCLogger()
        return new(Dict())
    end
end

export logvalue
function logvalue(logger::MCMCLogger, key::AbstractString, v::Float64)
  if !haskey(logger.dict, key)
    logger.dict[key] = Float64[]
  end
  push!(logger.dict[key],v)
end

function mcmc_sequencepair(tunemcmc::TuneMCMC, citer::Int, niter::Int, samplerate::Int, rng::AbstractRNG, initialSample::SequencePairSample, modelparams::ModelParameters, B::Float64, cornercut::Int=100, fixAlignmentIn::Bool=false, writeoutput::Bool=false, outputdir::AbstractString="", userrordistribution::Bool=false,msamples::Int=1)
  tic()
  burnin = niter / 2
  seqpair = initialSample.seqpair
  pairparams = initialSample.params
  current = PairParameters(pairparams)
  proposed = PairParameters(pairparams)
  current_sample = initialSample
  count1 = zeros(Float64, seqpair.seq1.length, modelparams.numHiddenStates)
  count2 = zeros(Float64, seqpair.seq2.length, modelparams.numHiddenStates)
  regimes1 = zeros(Float64, seqpair.seq1.length, modelparams.numHiddenStates*2)
  regimes2 = zeros(Float64, seqpair.seq2.length, modelparams.numHiddenStates*2)
  datalikcache = AlignmentHMM.DatalikCache(seqpair.seq1.length, seqpair.seq2.length, modelparams.numHiddenStates)

  fixAlignment = fixAlignmentIn || initialSample.seqpair.single
  cache = AlignmentHMM.HMMCache(seqpair.seq1.length+1,seqpair.seq2.length+1,modelparams.numHiddenStates,cornercut)

  mode = "a"
  if citer == 0
    mode = "w"
  end

  printoutput = false

  mcmclogger = MCMCLogger()
  if writeoutput
    mcmcout = open(string(outputdir, "mcmc",fmt("04d", seqpair.id),".log"), mode)
    alignout = open(string(outputdir, "align",fmt("04d", seqpair.id),".log"), mode)
    acceptanceout = open(string(outputdir, "acceptance",fmt("04d", seqpair.id),".log"), mode)
    angleout =  open(string(outputdir, "angleout",fmt("04d", seqpair.id),".log"), mode)

    if citer == 0
      write(mcmcout, string("iter","\t", "currentll", "\t", "current_lambda","\t","current_mu","\t", "current_ratio", "\t", "current_r", "\t", "current_t","\n"))
      write(angleout, string("iter","\t", join([i for i=1:current_sample.seqpair.seq1.length],"\t"),"\n"))
    end
  end

  samples = SequencePairSample[]

  randweights = Float64[0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25]
  logger = AcceptanceLogger()
  moveWeights = Float64[0.0, 0.0, 7.5, 100.0]
  if fixAlignment
    moveWeights = Float64[0.0, 0.0, 0.0, 100.0]
  end
  nsamples = 1
  currentll, current_samples, dummy = AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2)
  #currentll, current_samples = tkf92(datalikcache, nsamples, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
  current_sample = current_samples[end]

  proposedll = currentll
  logll = Float64[]
  for i=1:niter
     currentiter = citer + i - 1

    if writeoutput && currentiter % 25 == 0
      write(angleout, string(currentiter,"\t",join(current_sample.seqpair.seq1.phi,"\t"), "\n"))
    end
    if modelparams.useerrordistribution && rand(rng) < 0.15
      current_sample = sampleobs(rng, current_sample, modelparams)
    end

    move = UtilsModule.sample(rng, moveWeights)
    if move == 1
      currentll, current_samples = tkf92(datalikcache, nsamples, rng, seqpair, current, modelparams, B, cornercut)
      current_sample = current_samples[end]
      currentll, dummy = tkf92(datalikcache, 0, rng, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)

      seqpair = current_sample.seqpair
      if writeoutput
        write(alignout, string(currentiter), "\n")
        write(alignout, string(join(current_sample.states, ""),"\n"))
        write(alignout, getaminoacidalignment(seqpair.seq1, current_sample.align1),"\n")
        write(alignout, getaminoacidalignment(seqpair.seq2, current_sample.align2),"\n\n")
        flush(alignout)
      end
      logAccept!(logger, "fullalignment")
    elseif move == 2
      logAccept!(logger, "fullstates")
    elseif move == 3
      seqpair = current_sample.seqpair
      if writeoutput
        tic()
      end
      #=if rand(rng) < 0.1
        println("full")
        ll, current_samples, dummy = AlignmentHMM.tkf92(msamples, rng, datalikcache, seqpair, current, modelparams, cornercut)
      else=#
        width = rand(50:100)
        partialAlignment = rand(1:2)
        if partialAlignment == 1
          startpos = rand(1:max(1, seqpair.seq1.length-width))
          endpos = min(seqpair.seq1.length, startpos + width)
          ll, current_samples, dummy = AlignmentHMM.tkf92(msamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, false, current_sample.align1, current_sample.align2, false, current_sample.states, 1, startpos, endpos)
        else
          startpos = rand(1:max(1, seqpair.seq2.length-width))
          endpos = min(seqpair.seq2.length, startpos + width)
          ll, current_samples, dummy = AlignmentHMM.tkf92(msamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, false, current_sample.align1, current_sample.align2, false, current_sample.states, 2, startpos, endpos)
        end
      #end

      current_sample = current_samples[end]

      if writeoutput
        alignment_time = toc()
        println("alignment=", alignment_time,"\t", width)
        for s in current_samples
          write(alignout,string("iter=", i, "\n"))
          write(alignout,string(getaminoacidalignment(seqpair.seq1, s.align1), "\n"))
          write(alignout,string(getaminoacidalignment(seqpair.seq2, s.align2), "\n"))
          write(alignout,string(getssalignment(seqpair.seq1, s.align1), "\n"))
          write(alignout,string(getssalignment(seqpair.seq2, s.align2), "\n"))
          write(alignout,string(getstatestring(s.states),"\n"))
        end
      end
      logAccept!(logger, "partialalignment")
    elseif move >= 4
      fixStates = true
      numMCMCiter = 30
      AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, false, current_sample.states)
      fixStates = true
      #=
      if rand(rng) < 0.5
        fixStates = false
        numMCMCiter = 5
      end
      =#
      currentll, dummy1, dummy2 = AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, fixStates, current_sample.states)
      current_sample = dummy1[end]
      #tic()
      for q=1:numMCMCiter
        move = UtilsModule.sample(rng, randweights)
        if !current_sample.seqpair.single
          movename = ""
          propratio = 0.0
          if move == 4
            sigma = 0.2*getfactor(tunemcmc, 1)
            d1 = Truncated(Normal(current.lambda, sigma), 0.0, Inf)
            proposed.lambda = rand(d1)
            d2 = Truncated(Normal(proposed.lambda, sigma), 0.0, Inf)
            propratio = logpdf(d2, current.lambda) - logpdf(d1, proposed.lambda)
            movename = "lambda"
          elseif move == 5
            sigma = 0.15*getfactor(tunemcmc, 2)
            d1 = Truncated(Normal(current.ratio, sigma), 0.0, 1.0)
            proposed.ratio = rand(d1)
            d2 = Truncated(Normal(proposed.ratio, sigma), 0.0, 1.0)
            propratio = logpdf(d2, current.ratio) - logpdf(d1, proposed.ratio)
            movename = "ratio"
          elseif move == 6
            sigma = 0.25*getfactor(tunemcmc, 3)
            d1 = Truncated(Normal(current.r, sigma), 0.0, 1.0)
            proposed.r = rand(d1)
            d2 = Truncated(Normal(proposed.r, sigma), 0.0, 1.0)
            propratio = logpdf(d2, current.r) - logpdf(d1, proposed.r)
            movename = "r"
          elseif move == 7
            sigma = 0.1*getfactor(tunemcmc, 4)
            movename = "t"
            d1 = Truncated(Normal(current.t, sigma), 0.0, Inf)
            proposed.t = rand(d1)
            d2 = Truncated(Normal(proposed.t, sigma), 0.0, Inf)
            propratio = logpdf(d2, current.t) - logpdf(d1, proposed.t)
          end

          proposed.mu = proposed.lambda/proposed.ratio
          if(proposed.lambda > 0.0 && proposed.mu > 0.0 && proposed.lambda < proposed.mu && 0.001 < proposed.ratio < 0.999 && 0.001 < proposed.r < 0.999 && proposed.t > 0.0)
            proposedll, proposed_samples, dummy2 = AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, proposed, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, fixStates, current_sample.states)

            if(exp(proposedll-currentll+propratio) > rand(rng))
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
      end
      #time = toc()

      currentll, dummy1, dummy2 = AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
      current_sample = dummy1[end]
      #println("A", time)
    end

    push!(logll, currentll)
    if currentiter % samplerate == 0
      writell = currentll
      if writell == -Inf
        writell = -1e20
      end
      if writeoutput
        logvalue(mcmclogger, "ll", writell)
        logvalue(mcmclogger, "lambda", current.lambda)
        logvalue(mcmclogger, "mu", current.mu)
        logvalue(mcmclogger, "ratio", current.ratio)
        logvalue(mcmclogger, "r", current.r)
        logvalue(mcmclogger, "t", current.t)
        write(mcmcout, string(currentiter,"\t", writell, "\t", current.lambda,"\t",current.mu,"\t",current.ratio, "\t", current.r, "\t", current.t,"\n"))
      end
    end

    if i >= burnin && (currentiter+1) % samplerate == 0
      for s in current_samples
        if modelparams.obsnodes[1].useswitching
          align1 = s.align1
          align2 = s.align2
          states = s.states
          len = length(align1)
          t = s.params.t
          s.regimes = zeros(Int, len)
          for a=1:len
            i = align1[a]
            j = align2[a]
            h = states[a]
            s.regimes[a] = sample_regimes(modelparams.obsnodes[states[a]].switching, rng, s.seqpair.seq1, s.seqpair.seq2, i, j, t)
            if s.regimes[a] == 1
              if i > 0
                regimes1[i,(h-1)*2+1] += 1
              end
              if j > 0
                regimes2[j,(h-1)*2+1] += 1
              end
            elseif s.regimes[a] == 2
              if i > 0
                regimes1[i,(h-1)*2+1] += 1
              end
              if j > 0
                regimes2[j,(h-1)*2+2] += 1
              end
            elseif s.regimes[a] == 3
              if i > 0
                regimes1[i,(h-1)*2+2] += 1
              end
              if j > 0
                regimes2[j,(h-1)*2+1] += 1
              end
            elseif s.regimes[a] == 4
              if i > 0
                regimes1[i,(h-1)*2+2] += 1
              end
              if j > 0
                regimes2[j,(h-1)*2+2] += 1
              end
            end
          end
        end

        h1, h2 = getsequencestates(s.align1, s.align2, s.states)
        for f=1:length(h1)
          count1[f,h1[f]] += 1
        end
        for f=1:length(h2)
          count2[f,h2[f]] += 1
        end
        push!(samples, s)
      end
    end
  end

  if writeoutput
    close(mcmcout)
    close(alignout)

    write(acceptanceout, string(UtilsModule.list(logger),"\n"))
    close(acceptanceout)
    close(angleout)
  end


  if !current_sample.seqpair.single
    logacceptance(tunemcmc, 1, getacceptanceratio(logger, "lambda"))
    logacceptance(tunemcmc, 2, getacceptanceratio(logger, "ratio"))
    logacceptance(tunemcmc, 3, getacceptanceratio(logger, "r"))
    logacceptance(tunemcmc, 4, getacceptanceratio(logger, "t"))
    #println("F", getfactor(tunemcmc, 1),"\t", gethistory(tunemcmc, 1),"\t", getacceptanceratio(logger, "lambda"))
    #println("G", getfactor(tunemcmc, 2),"\t", gethistory(tunemcmc, 2),"\t", getacceptanceratio(logger, "ratio"))
    #println("H", getfactor(tunemcmc, 3),"\t", gethistory(tunemcmc, 3),"\t", getacceptanceratio(logger, "r"))
    #println("I", getfactor(tunemcmc, 4),"\t", gethistory(tunemcmc, 4),"\t", getacceptanceratio(logger, "t"))
  end
  expll = logsumexp(logll+log(1/Float64(length(logll))))

  toc()

  return citer+niter, current, samples, expll, tunemcmc, count1, count2, regimes1, regimes2, mcmclogger
end


function mcmc_sequencepair2(tunemcmc::TuneMCMC, citer::Int, niter::Int, samplerate::Int, rng::AbstractRNG, initialSample::SequencePairSample, modelparams::ModelParameters, B::Float64, cornercut::Int=100, fixAlignmentIn::Bool=false, writeoutput::Bool=false, outputdir::AbstractString="", userrordistribution::Bool=false,msamples::Int=1)
  tic()
  burnin = niter / 3
  seqpair = initialSample.seqpair
  pairparams = initialSample.params
  current = PairParameters(pairparams)
  proposed = PairParameters(pairparams)
  current_sample = initialSample
  count1 = zeros(Float64, seqpair.seq1.length, modelparams.numHiddenStates)
  count2 = zeros(Float64, seqpair.seq2.length, modelparams.numHiddenStates)
  regimes1 = zeros(Float64, seqpair.seq1.length, modelparams.numHiddenStates*2)
  regimes2 = zeros(Float64, seqpair.seq2.length, modelparams.numHiddenStates*2)
  datalikcache = AlignmentHMM.DatalikCache(seqpair.seq1.length, seqpair.seq2.length, modelparams.numHiddenStates)

  fixAlignment = fixAlignmentIn || initialSample.seqpair.single
  #cache = ones(seqpair.seq1.length+1, seqpair.seq2.length+1, 3, modelparams.numHiddenStates)*-Inf
  cache = AlignmentHMM.HMMCache(1,1,modelparams.numHiddenStates,1)
  if !fixAlignment
    cache = AlignmentHMM.HMMCache(seqpair.seq1.length+1,seqpair.seq2.length+1,modelparams.numHiddenStates,cornercut)
  end

  mode = "a"
  if citer == 0
    mode = "w"
  end

  printoutput = false

  mcmclogger = MCMCLogger()
  if writeoutput
    mcmcout = open(string(outputdir, "mcmc",Formatting.fmt("04d", seqpair.id),"_B",B,".log"), mode)
    alignout = open(string(outputdir, "align",Formatting.fmt("04d", seqpair.id),".log"), mode)
    acceptanceout = open(string(outputdir, "acceptance",Formatting.fmt("04d", seqpair.id),".log"), mode)
    angleout =  open(string(outputdir, "angleout",Formatting.fmt("04d", seqpair.id),".log"), mode)

    if citer == 0
      write(mcmcout, string("iter","\t", "currentll", "\t", "current_lambda","\t","current_mu","\t", "current_ratio", "\t", "current_r", "\t", "current_t","\n"))
      write(angleout, string("iter","\t", join([i for i=1:current_sample.seqpair.seq1.length],"\t"),"\n"))
    end
  end

  samples = SequencePairSample[]

  randweights = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]
  logger = AcceptanceLogger()
  moveWeights = Float64[0.0, 0.0, 7.5, 100.0]
  if fixAlignment
    moveWeights = Float64[0.0, 0.0, 0.0, 100.0]
  end
  nsamples = 1
  currentll, current_samples, dummy = AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2)
  #currentll, current_samples = tkf92(datalikcache, nsamples, rng, seqpair, current, modelparams, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
  current_sample = current_samples[end]

  proposedll = currentll
  logll = Float64[]
  for i=1:niter
     currentiter = citer + i - 1

    if writeoutput && currentiter % 25 == 0
      write(angleout, string(currentiter,"\t",join(current_sample.seqpair.seq1.phi,"\t"), "\n"))
    end
    if modelparams.useerrordistribution && rand(rng) < 0.15
      current_sample = sampleobs(rng, current_sample, modelparams)
    end

    #println(current_samples[end].seqpair.seq1.phi)
    #println(current_samples[end].seqpair.seq1.phi_obs)




    move = UtilsModule.sample(rng, moveWeights)
    if move == 1
      currentll, current_samples = tkf92(datalikcache, nsamples, rng, seqpair, current, modelparams, B, cornercut)
      current_sample = current_samples[end]
      currentll, dummy = tkf92(datalikcache, 0, rng, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)

      seqpair = current_sample.seqpair
      if writeoutput
        write(alignout, string(currentiter), "\n")
        write(alignout, string(join(current_sample.states, ""),"\n"))
        write(alignout, getaminoacidalignment(seqpair.seq1, current_sample.align1),"\n")
        write(alignout, getaminoacidalignment(seqpair.seq2, current_sample.align2),"\n\n")
        flush(alignout)
      end
      logAccept!(logger, "fullalignment")
    elseif move == 2
      logAccept!(logger, "fullstates")
    elseif move == 3
      seqpair = current_sample.seqpair
      if writeoutput
        tic()
      end
      #=if rand(rng) < 0.1
        println("full")
        ll, current_samples, dummy = AlignmentHMM.tkf92(msamples, rng, datalikcache, seqpair, current, modelparams, cornercut)
      else=#
        width = rand(50:100)
        partialAlignment = rand(1:2)
        if partialAlignment == 1
          startpos = rand(1:max(1, seqpair.seq1.length-width))
          endpos = min(seqpair.seq1.length, startpos + width)
          ll, current_samples, dummy = AlignmentHMM.tkf92(msamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, false, current_sample.align1, current_sample.align2, false, current_sample.states, 1, startpos, endpos)
        else
          startpos = rand(1:max(1, seqpair.seq2.length-width))
          endpos = min(seqpair.seq2.length, startpos + width)
          ll, current_samples, dummy = AlignmentHMM.tkf92(msamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, false, current_sample.align1, current_sample.align2, false, current_sample.states, 2, startpos, endpos)
        end
      #end

      current_sample = current_samples[end]

      if writeoutput
        alignment_time = toc()
        println("alignment=", alignment_time,"\t", width)
        for s in current_samples
          write(alignout,string("iter=", i, "\n"))
          write(alignout,string(getaminoacidalignment(seqpair.seq1, s.align1), "\n"))
          write(alignout,string(getaminoacidalignment(seqpair.seq2, s.align2), "\n"))
          write(alignout,string(getssalignment(seqpair.seq1, s.align1), "\n"))
          write(alignout,string(getssalignment(seqpair.seq2, s.align2), "\n"))
          write(alignout,string(getstatestring(s.states),"\n"))
        end
      end
      logAccept!(logger, "partialalignment")
    elseif move >= 4
      fixStates = true
      numMCMCiter = 10
      AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, false, current_sample.states)
      fixStates = true

      currentll, dummy1, dummy2 = AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, fixStates, current_sample.states)
      current_sample = dummy1[end]

      for q=1:numMCMCiter
        move = UtilsModule.sample(rng, randweights)
        if !current_sample.seqpair.single
          movename = ""
          propratio = 0.0
          if move == 4
            sigma = 0.2*getfactor(tunemcmc, 1)
            d1 = Truncated(Normal(current.lambda, sigma), 0.0, Inf)
            proposed.lambda = rand(d1)
            d2 = Truncated(Normal(proposed.lambda, sigma), 0.0, Inf)
            propratio = logpdf(d2, current.lambda) - logpdf(d1, proposed.lambda)
            movename = "lambda"
          elseif move == 5
            sigma = 0.15*getfactor(tunemcmc, 2)
            d1 = Truncated(Normal(current.ratio, sigma), 0.0, 1.0)
            proposed.ratio = rand(d1)
            d2 = Truncated(Normal(proposed.ratio, sigma), 0.0, 1.0)
            propratio = logpdf(d2, current.ratio) - logpdf(d1, proposed.ratio)
            movename = "ratio"
          elseif move == 6
            sigma = 0.25*getfactor(tunemcmc, 3)
            d1 = Truncated(Normal(current.r, sigma), 0.0, 1.0)
            proposed.r = rand(d1)
            d2 = Truncated(Normal(proposed.r, sigma), 0.0, 1.0)
            propratio = logpdf(d2, current.r) - logpdf(d1, proposed.r)
            movename = "r"
          elseif move == 7
            #sigma = 0.25
            sigma = 0.1*getfactor(tunemcmc, 4)
            movename = "t"
            #=
            d1 = Truncated(Normal(current.t, sigma), 0.0, Inf)
            proposed.t = rand(d1)
            d2 = Truncated(Normal(proposed.t, sigma), 0.0, Inf)
            propratio = logpdf(d2, current.t) - logpdf(d1, proposed.t)=#
            proposed.t += rand(Normal(0.0, sigma))
          end

          proposed.mu = proposed.lambda/proposed.ratio
          if(proposed.lambda > 0.0 && proposed.mu > 0.0 && proposed.lambda < proposed.mu && 0.001 < proposed.ratio < 0.999 && 0.001 < proposed.r < 0.999 && proposed.t > 0.0)
            proposedll, proposed_samples, dummy2 = AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, proposed, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, fixStates, current_sample.states)

            if(exp(proposedll-currentll+propratio) > rand(rng))
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
            proposed = PairParameters(current)
            logReject!(logger, movename)
          end
        end
      end

      currentll, dummy1, dummy2 = AlignmentHMM.tkf92(nsamples, rng, cache, datalikcache, seqpair, current, modelparams, B, cornercut, true, current_sample.align1, current_sample.align2, true, current_sample.states)
      current_sample = dummy1[end]
    end

    push!(logll, currentll)
    if currentiter % samplerate == 0
      writell = currentll
      if writell == -Inf
        writell = -1e20
      end
      if writeoutput
        logvalue(mcmclogger, "ll", writell)
        logvalue(mcmclogger, "lambda", current.lambda)
        logvalue(mcmclogger, "mu", current.mu)
        logvalue(mcmclogger, "ratio", current.ratio)
        logvalue(mcmclogger, "r", current.r)
        logvalue(mcmclogger, "t", current.t)
        write(mcmcout, string(currentiter,"\t", writell, "\t", current.lambda,"\t",current.mu,"\t",current.ratio, "\t", current.r, "\t", current.t,"\n"))
      end
    end

    if i >= burnin && (currentiter+1) % samplerate == 0
      for s in current_samples
        h1, h2 = getsequencestates(s.align1, s.align2, s.states)
        for f=1:length(h1)
          count1[f,h1[f]] += 1
        end
        for f=1:length(h2)
          count2[f,h2[f]] += 1
        end
        push!(samples, s)
      end
    end
  end

  if writeoutput
    close(mcmcout)
    close(alignout)

    write(acceptanceout, string(list(logger),"\n"))
    close(acceptanceout)
    close(angleout)
  end


  if !current_sample.seqpair.single
    logacceptance(tunemcmc, 1, getacceptanceratio(logger, "lambda"))
    logacceptance(tunemcmc, 2, getacceptanceratio(logger, "ratio"))
    logacceptance(tunemcmc, 3, getacceptanceratio(logger, "r"))
    logacceptance(tunemcmc, 4, getacceptanceratio(logger, "t"))
  end
  expll = logsumexp(logll+log(1/Float64(length(logll))))

  toc()

  return citer+niter, current, samples, expll, tunemcmc, count1, count2, regimes1, regimes2, mcmclogger
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
  localObjectiveFunction = ((param, grad) -> tkf92(datalikcache, 0, MersenneTwister(330101840810391), seqpair, PairParameters(param), modelparams, cornercut, fixAlignment, align1, align2)[1])
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
  modelfile = "models/pairhmm16_nossoptratesching_n128.jls"

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

  mask = Int[OBSERVED_DATA, OBSERVED_DATA,MISSING_DATA, OBSERVED_DATA, MISSING_DATA,MISSING_DATA]

  seq1, seq2 = masksequences(pairs[1].seq1, pairs[1].seq2, mask)
  seqpair = SequencePair(0,seq1, seq2)
  ll, mlsample = mlalignment(seqpair, modelparams, cornercut)
  println(getaminoacidalignment(seqpair.seq1, mlsample.align1))
  println(getaminoacidalignment(seqpair.seq2, mlsample.align2))
  println(mlsample.states)
end

function mlalignment(seqpair::SequencePair, modelparams::ModelParameters, cornercut::Int, initialParams::PairParameters)
  mlparams = mlalignmentopt(seqpair, modelparams, cornercut, initialParams, 3)
  ll, mlsample = tkf92(datalikcache, 0, MersenneTwister(242402531025555), seqpair, mlparams, modelparams, cornercut, false, zeros(Int,1), zeros(Int,1),false,zeros(Int,1),false,false)
  mlsample.params = mlparams
  println(getaminoacidalignment(seqpair.seq1, mlsample.align1))
  println(getaminoacidalignment(seqpair.seq2, mlsample.align2))
  println(mlsample.states)
  return ll, mlsample
end

function count_kmers(samples::Array{SequencePairSample,1}, k::Int, countfilename)
  count = counter(AbstractString)
  for sample in samples
    len = length(sample.states)-k+1
    for i=1:len
      kmer = string(join(sample.states[i:(i+k-1)], "-"))
      push!(count, kmer)
    end
  end

  out = open(countfilename, "w")
  for key in keys(count)
        sp = split(key,"-")
        central = sp[round(Int64, (k+1)/2)]
        write(out,string(key,"\t",central,"\t",count[key],"\n"))
  end
  close(out)
  #println(count)
end

function analysejumps()
    modelfile = "models/pairhmm12_switching_n25_fixalignment=false.jls"
    modelparams = readmodel(modelfile)
    prior = modelparams.prior
    obsnodes = modelparams.obsnodes
    hmminitprobs = modelparams.hmminitprobs
    hmmtransprobs = modelparams.hmmtransprobs
    numHiddenStates = length(hmminitprobs)
    inputsamples = modelparams.samples
    pairs = SequencePair[sample.seqpair for sample in inputsamples]

    sscounts = zeros(Float64, 3, 3)
    sstotals = zeros(Float64, 3, 3)

    for current_sample in inputsamples
      align1 = current_sample.align1
      align2 = current_sample.align2
      states = current_sample.states
      len = length(align1)
      t = current_sample.params.t
      #current_sample.regimes = zeros(Int, len)
      for a=1:len
        h = states[a]
        i = align1[a]
        j = align2[a]
        if i > 0 && j > 0
          ss1 = current_sample.seqpair.seq1.ss[i]
          ss2 = current_sample.seqpair.seq2.ss[j]
          if  ss1 > 0 && ss2 > 0
            if current_sample.regimes[a] == 2 || current_sample.regimes[a] == 3
              sscounts[ss1,ss2] += 1.0
              sscounts[ss2,ss1] += 1.0
            end
            sstotals[ss1,ss2] += 1.0
            sstotals[ss2,ss1] += 1.0
          end
        end
      end
    end

  println(sscounts)
  println(sstotals)
  println(sscounts./sstotals)
end

function train()
  starttime = now()

  srand(98418108751401)
  rng = MersenneTwister(242402531025555)

  maxiters = 100000
  cornercutinit = 10
  cornercut = 125
  usesecondarystructure = true
  useswitching = true
  useparallel = true
  fixInputAlignments = false
  optimizeratematrix = true
  useerrordistribution = false
  B = 1.0
  smartcornercutting = false

  writeoutput = false
  #inputsamples = shuffle!(rng, load_sequences_and_alignments("data/glob.txt"))
  inputsamples = shuffle!(rng, load_sequences_and_alignments("data/data_diverse_new2.txt"))
  println("N=",length(inputsamples))
  #inputsamples = inputsamples[1:10]
  pairs = SequencePair[sample.seqpair for sample in inputsamples]


  mcmcoutputdir = string("mcmcoutput/")
  mkpath(mcmcoutputdir)

  println("N=",length(pairs))

  tunemcmc = TuneMCMC[]
  for i=1:length(pairs)
    push!(tunemcmc, TuneMCMC(4))
  end

  mcmciter = 40
  samplerate = 2

  numHiddenStates = 64

  freeParameters = 6*numHiddenStates + (numHiddenStates-1) + (numHiddenStates*numHiddenStates - numHiddenStates) + numHiddenStates*19
  if useswitching
    freeParameters = 12*numHiddenStates + (numHiddenStates-1) + (numHiddenStates*numHiddenStates - numHiddenStates) + numHiddenStates*38 + numHiddenStates*2
  end

  mkpath("models/")
  loadModel = false
  logoutputdir = ""
  if useswitching
    logoutputdir = string("logs/training_pairhmm",numHiddenStates,"_switching_n",length(pairs),"_fixalignment=",fixInputAlignments,"/")
    modelfile = string("models/pairhmm",numHiddenStates,"_switching_n",length(pairs),"_fixalignment=",fixInputAlignments,".jls")
  else
    logoutputdir = string("logs/training_pairhmm",numHiddenStates,"_noswitching_n",length(pairs),"_fixalignment=",fixInputAlignments,"/")
    modelfile = string("models/pairhmm",numHiddenStates,"_noswitching_n",length(pairs),"_fixalignment=",fixInputAlignments,".jls")
  end
  mkpath(logoutputdir)



  hmminitprobs = ones(Float64,numHiddenStates)/Float64(numHiddenStates)
  hmmtransprobs = ones(Float64, numHiddenStates, numHiddenStates)/Float64(numHiddenStates)
  prior = PriorDistribution()
  obsnodes = ObservationNode[]
  for h=1:numHiddenStates
    push!(obsnodes, ObservationNode())
    obsnodes[h].usesecondarystructure = usesecondarystructure
    obsnodes[h].useswitching = useswitching
    obsnodes[h].switching.ss_r1.ctmc.enabled = usesecondarystructure
    obsnodes[h].switching.ss_r2.ctmc.enabled = usesecondarystructure
    if useswitching
      set_parameters(obsnodes[h].switching.aapairnode_r1, rand(Dirichlet(ones(Float64,20)*1.5)), 1.0)
      set_parameters(obsnodes[h].switching.aapairnode_r2, rand(Dirichlet(ones(Float64,20)*1.5)), 1.0)
      set_parameters(obsnodes[h].switching.diffusion_r1, 1.0, rand()*2.0*pi - pi, 0.5, 1.0, rand()*2.0*pi - pi, 0.5, 0.1, 1.0,1.0)
      set_parameters(obsnodes[h].switching.diffusion_r2, 1.0, rand()*2.0*pi - pi, 0.5, 1.0, rand()*2.0*pi - pi, 0.5, 0.1, 1.0,1.0)
      obsnodes[h].switching.alpha = 25.0
      obsnodes[h].switching.pi_r1 = 0.5
      set_parameters(obsnodes[h].switching.ss_r1, rand(Dirichlet(ones(Float64,3)*1.5)), 1.0)
      set_parameters(obsnodes[h].switching.ss_r2, rand(Dirichlet(ones(Float64,3)*1.5)), 1.0)
    else
      set_parameters(obsnodes[h].aapairnode, rand(Dirichlet(ones(Float64,20)*1.5)), 1.0)
      set_parameters(obsnodes[h].diffusion, 1.0, rand()*2.0*pi - pi, 0.5, 1.0, rand()*2.0*pi - pi, 0.5, 0.1, 1.0, 1.0)
      set_parameters(obsnodes[h].ss, rand(Dirichlet(ones(Float64,3)*1.5)), 1.0)
    end
  end

  modelparams::ModelParameters = ModelParameters(prior, obsnodes, hmminitprobs, hmmtransprobs, useerrordistribution, inputsamples)
  if loadModel
    modelparams = readmodel(modelfile)
    prior = modelparams.prior
    obsnodes = modelparams.obsnodes
    hmminitprobs = modelparams.hmminitprobs
    hmmtransprobs = modelparams.hmmtransprobs
    numHiddenStates = length(hmminitprobs)
    inputsamples = modelparams.samples
    pairs = SequencePair[sample.seqpair for sample in inputsamples]
  end
  modelparams.useerrordistribution = false
  aashapescale(obsnodes)

  tic()

  current_samples = SequencePairSample[]
  if useparallel
    refs = RemoteRef[]
    for k=1:length(pairs)
      if !inputsamples[k].aligned
        cache = AlignmentHMM.HMMCache(inputsamples[k].seqpair.seq1.length+1,inputsamples[k].seqpair.seq2.length+1,modelparams.numHiddenStates,cornercut)
        ref = @spawn  tkf92(1, MersenneTwister(abs(rand(Int))), cache, AlignmentHMM.DatalikCache(inputsamples[k].seqpair.seq1.length, inputsamples[k].seqpair.seq2.length, modelparams.numHiddenStates),pairs[k], inputsamples[k].params, deepcopy(modelparams),  B, cornercut)
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
        push!(current_samples, tkf92(datalikcache, 1, rng, pairs[k], PairParameters(), modelparams, cornercut)[2][1])
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


  refs = RemoteRef[]
  if smartcornercutting
    for sample in current_samples
      ref = @spawn findcornertcutbound(sample.seqpair, sample.align1, sample.align2)
      push!(refs,ref)
    end
    for k=1:length(current_samples)
      sample = current_samples[k]
      cornercut = fetch(refs[k])
      println(k,"\t",sample.seqpair.seq1.length,"\t",sample.seqpair.seq2.length,"\t",cornercut)
      sample.seqpair.cornercut = cornercut[2]
    end
  else
    for sample in current_samples
      cornecut = 100
      sample.seqpair.cornercut = cornercut + abs(sample.seqpair.seq1.length-sample.seqpair.seq2.length)
    end
  end



  println("Initialised")

  mkpath("logs/")
  mlfilename = string("logs/ml",numHiddenStates,".log")
  if useswitching
    mlfilename = string("logs/ml",numHiddenStates,"_switching.log")
  end
  mlwriter = open(mlfilename, "w")
  write(mlwriter, "iter\tll\tcount\tavgll\tnumFreeParameters\tAIC\tlambda_shape\tlambda_scale\tmu_shape\tmu_scale\tr_alpha\tr_beta\tt_shape\tt_scale\ttime_per_iter\n")


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
        println(k,"\t",current_samples[k].seqpair.cornercut)
        ref = @spawn mcmc_sequencepair(deepcopy(tunemcmc[k]), currentiter, mcmciter, samplerate, MersenneTwister(abs(rand(Int))), deepcopy(current_samples[k]), deepcopy(modelparams), 1.0, current_samples[k].seqpair.cornercut, fixInputAlignments && inputsamples[k].aligned, writeoutput, mcmcoutputdir)
        push!(refs,ref)
      end

      for k=1:length(pairs)
        it, newparams, ksamples, expll, tunemcmctemp = fetch(refs[k])
        tunemcmc[k] = tunemcmctemp
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

      countfilename = string("logs/count",numHiddenStates,".log")
      if useswitching
        countfilename = string("logs/count",numHiddenStates,"_switching.log")
      end
      count_kmers(samples,3,countfilename)
    else
      for k=1:length(pairs)
        it, newparams, ksamples, expll, tunemcmctemp = mcmc_sequencepair(deepcopy(tunemcmc[k]), currentiter, mcmciter, samplerate,  MersenneTwister(abs(rand(Int))), current_samples[k], modelparams, 1.0, current_samples[k].seqpair.cornercut, fixInputAlignments && inputsamples[k].aligned, writeoutput)
        tunemcmc[k] = tunemcmctemp
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
    println("estep=",estep_elapsed)

    tic()
    if useparallel
      refs = RemoteRef[]

      for h=1:numHiddenStates
        samplescopy = SequencePairSample[deepcopy(s) for s in samples]
        ref = @spawn mlopt(h, samplescopy, deepcopy(obsnodes))
        push!(refs, ref)
      end
      for h=1:numHiddenStates

        params = fetch(refs[h])
        if obsnodes[h].useswitching
          set_parameters(obsnodes[h].switching.aapairnode_r1, params[1][1], 1.0)
          obsnodes[h].switching.aapairnode_r1.branchscale = params[1][2]
          set_parameters(obsnodes[h].switching.aapairnode_r2, params[2][1], 1.0)
          obsnodes[h].switching.aapairnode_r2.branchscale = params[2][2]
          set_parameters(obsnodes[h].switching.diffusion_r1, params[3])
          set_parameters(obsnodes[h].switching.diffusion_r2, params[4])
          if usesecondarystructure
            set_parameters(obsnodes[h].switching.ss_r1, params[5], 1.0)
            set_parameters(obsnodes[h].switching.ss_r2, params[6], 1.0)
          end
          obsnodes[h].switching.pi_r1 = params[7][1]
          obsnodes[h].switching.alpha = params[7][2]

          println(h,"\t",1,"\t", obsnodes[h].switching.aapairnode_r1.branchscale)
          println(h,"\t",2,"\t", obsnodes[h].switching.aapairnode_r2.branchscale)
        else
          set_parameters(obsnodes[h].aapairnode, params[1][1], 1.0)
          obsnodes[h].aapairnode.branchscale = params[1][2]
          set_parameters(obsnodes[h].diffusion, params[2])
          if usesecondarystructure
            set_parameters(obsnodes[h].ss, params[3], 1.0)
          end
        end
      end

      if i % 5 == 0
        refs = RemoteRef[]
        if obsnodes[1].useswitching
          for h=1:numHiddenStates
            samplescopy = SequencePairSample[deepcopy(s) for s in samples]
            ref = @spawn fullswitchingopt(h, samplescopy, deepcopy(obsnodes))
            push!(refs, ref)
          end
          for h=1:numHiddenStates
            set_parameters(obsnodes[h].switching, fetch(refs[h]))
          end
        end
      end
    else
      for h=1:numHiddenStates
        mlopt(h, samples, obsnodes)
      end
    end
    toc()

    tic()
    if usesecondarystructure
      minsss = ssoptrates(samples, obsnodes)
      for obsnode in obsnodes
        set_parameters(obsnode.ss, minsss[1], minsss[2], minsss[3], 1.0)
        set_parameters(obsnode.switching.ss_r1, minsss[1], minsss[2], minsss[3], 1.0)
        set_parameters(obsnode.switching.ss_r2, minsss[1], minsss[2], minsss[3], 1.0)
      end
      println(obsnodes[1].switching.ss_r1.ctmc.S)
    end

    if useerrordistribution
      angle_error_kappa = optimize_diffusion_error(samples,modelparams)[1]
      for obsnode in modelparams.obsnodes
        obsnode.angle_error_kappa = angle_error_kappa
      end
      println("angle_error_kappa",modelparams.obsnodes[1].angle_error_kappa)
    end

     aashapescale(obsnodes)
   # tic()
    #optimize_aabranchscale(samples,obsnodes)
    #aabranchscale = toc()
    #println("scale=",obsnodes[1].aapairnode.branchscale)
    #println("aabranchscale=", aabranchscale)

    #tic()
    if optimizeratematrix
      if i > 1 && i % 10 == 0
        #optimize_aaratematrix_parallel(samples,obsnodes)
        optimize_aaratematrix(samples,obsnodes)
        swriter = open("S.txt","w")
        write(swriter, string(obsnodes[1].aapairnode.S, "\n"))
        close(swriter)
      end
    end
    #aaopt_elapsed = toc()
    #println("aaopt=",aaopt_elapsed)

    hmminitprobs, hmmtransprobs = hmmopt(samples,numHiddenStates)
    prior = prioropt(samples, prior)
    mstep_elapsed = toc()

    tempiter = i
    likelihood = 0.0
    N = 0
    for current_sample in current_samples
      align1 = current_sample.align1
      align2 = current_sample.align2
      states = current_sample.states
      len = length(align1)
      t = current_sample.params.t
      for a=1:len
        h = states[a]
        i = align1[a]
        j = align2[a]
        if i > 0 && j > 0
          likelihood += get_data_lik(obsnodes[h], current_sample.seqpair.seq1, current_sample.seqpair.seq2, i, j, t)
          N += 2
        elseif i > 0
          likelihood += get_data_lik_x0(obsnodes[h], current_sample.seqpair.seq1, i, t)
          N += 1
        elseif j > 0
          likelihood += get_data_lik_xt(obsnodes[h], current_sample.seqpair.seq2, j, t)
          N += 1
        end
      end
    end
    i = tempiter

    currenttime = now()
    write(mlwriter, string(i-1,"\t",likelihood, "\t", N, "\t", likelihood/Float64(N),"\t", freeParameters,"\t", aic(likelihood, freeParameters), "\t", join(prior.params,"\t"), "\t", Float64(currenttime-starttime)/Float64(i*1000), "\n"))
    flush(mlwriter)

    modelparams = ModelParameters(prior, obsnodes, hmminitprobs, hmmtransprobs, useerrordistribution, current_samples)
    ser = open(modelfile,"w")
    serialize(ser, modelparams)
    close(ser)


    if useswitching
      export_json(modelparams, string(logoutputdir,"pairhmm",numHiddenStates,"_switching_n",length(pairs),"_fixalignment=",fixInputAlignments,".json"))
    else
      export_json(modelparams, string(logoutputdir,"pairhmm",numHiddenStates,"_noswitching_n",length(pairs),"_fixalignment=",fixInputAlignments,".json"))
    end

    if useswitching
      hiddenstatesfile = string("logs/hiddenstates",numHiddenStates,"_switching_n",length(pairs),".txt")
      switchingfile = string("logs/switchingrates",numHiddenStates,"_switching_n",length(pairs),".txt")
    else
      hiddenstatesfile = string("logs/hiddenstates",numHiddenStates,"_noswitching_n",length(pairs),".txt")
      switchingfile = string("logs/switchingrates",numHiddenStates,"_noswitching_n",length(pairs),".txt")
    end
    write_hiddenstates(modelparams, hiddenstatesfile)

    if modelparams.obsnodes[1].useswitching
      if i % 3 == 1
        jumpwriter = open("empiricaljumps.txt", "w")
        write(jumpwriter, string(computeEmpiricalJumpPosteriorProbabilities(modelparams, current_samples)))
        close(jumpwriter)
      end
    end

    switchingout = open(switchingfile,"w")
    for k=1:length(obsnodes)
      obsnode = obsnodes[k]
      write(switchingout, string(k, "\t", obsnode.switching.alpha, "\t", obsnode.switching.pi_r1, "\n"))
    end
    close(switchingout)
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
    ss0 = 0
    sst = 0
    if a > 0
      x0 =  seqpair.seq1.seq[a]
      phi0 = seqpair.seq1.phi[a]
      psi0 = seqpair.seq1.psi[a]
      ss0 = seqpair.seq1.ss[a]
    end
    if b > 0
      xt =  seqpair.seq2.seq[b]
      phit = seqpair.seq2.phi[b]
      psit = seqpair.seq2.psi[b]
      sst = seqpair.seq2.ss[b]
    end

    x0, xt, phi, psi, ss0, sst  = NodesModule.sample(obsnodes[h], rng, x0, xt, phi0,phit,psi0,psit, ss0, sst, t)

    if a > 0
      newseq1.seq[a] = x0
      newseq1.phi[a] = phi[1]
      newseq1.psi[a] = psi[1]
      newseq1.ss[a] = ss0
    end
    if b > 0
      newseq2.seq[b] = xt
      newseq2.phi[b] = phi[2]
      newseq2.psi[b] = psi[2]
      newseq2.ss[b] = sst
    end

    i += 1
  end

  return SequencePair(0, newseq1,newseq2)
end

function computeEmpiricalJumpPosteriorProbabilities(modelparams::ModelParameters, samples::Array{SequencePairSample,1})
  jumpprobs = zeros(Float64, modelparams.numHiddenStates, 4)
  for pairsample in samples
    seqpair = pairsample.seqpair
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
      ss0 = 0
      sst = 0
      if a > 0
        x0 =  seqpair.seq1.seq[a]
        phi0 = seqpair.seq1.phi[a]
        psi0 = seqpair.seq1.psi[a]
        ss0 = seqpair.seq1.ss[a]
      end
      if b > 0
        xt =  seqpair.seq2.seq[b]
        phit = seqpair.seq2.phi[b]
        psit = seqpair.seq2.psi[b]
        sst = seqpair.seq2.ss[b]
      end
      v = get_regime_logprobs(modelparams.obsnodes[h].switching, x0, xt, phi0, psi0, phit, psit, ss0, sst, t)
      vsum = logsumexp(v)
      v = exp(v-vsum)
      for k=1:4
        jumpprobs[h,k] += v[k]
      end

      i += 1
    end
  end

  for h=1:modelparams.numHiddenStates
    jumpprobs[h,:] /= sum(jumpprobs[h,:])
  end
  return jumpprobs
end

function computeJumpPosteriorProbabilities(modelparams::ModelParameters, samples::Array{SequencePairSample,1})
  js1 = zeros(Float64,samples[1].seqpair.seq1.length)
  jt1 = zeros(Float64,samples[1].seqpair.seq1.length)
  js2 = zeros(Float64,samples[1].seqpair.seq2.length)
  jt2 = zeros(Float64,samples[1].seqpair.seq2.length)

  for pairsample in samples
    seqpair = pairsample.seqpair
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
      ss0 = 0
      sst = 0
      if a > 0
        x0 =  seqpair.seq1.seq[a]
        phi0 = seqpair.seq1.phi[a]
        psi0 = seqpair.seq1.psi[a]
        ss0 = seqpair.seq1.ss[a]
      end
      if b > 0
        xt =  seqpair.seq2.seq[b]
        phit = seqpair.seq2.phi[b]
        psit = seqpair.seq2.psi[b]
        sst = seqpair.seq2.ss[b]
      end
      v = get_regime_logprobs(modelparams.obsnodes[h].switching, x0, xt, phi0, psi0, phit, psit, ss0, sst, t)
      vsum = logsumexp(v)
      v = exp(v-vsum)
      if a > 0
        js1[a] += v[2]+v[3]
        jt1[a] += 1.0
      end
      if b > 0
        js2[b] += v[2]+v[3]
        jt2[b] += 1.0
      end

      #println(v)
      i += 1
    end
  end
  return (js1./jt1,js2./jt2)
end

using Gadfly
using Compose
using Cairo
using JSON
function test()
  srand(98418108751401)
  rng = MersenneTwister(242402531025555)

  #benchmarksfilepath = "data/glob.txt"
  #benchmarksfilepath = "data/holdout_data.txt"
  benchmarksfilepath = "data/2jp1_ltn40.pdb.txt"
  benchmarksname = basename(benchmarksfilepath)

  #pairs = shuffle!(rng, load_sequences_and_alignments(benchmarksfilepath))
  pairs = load_sequences_and_alignments(benchmarksfilepath)
  #pairs = load_sequences_and_alignments("data/glob.txt")
  radius = 0.0

  modelfile = "models/pairhmm64_switching_n674_fixalignment=false.jls"
  mask = Int[OBSERVED_DATA, OBSERVED_DATA,OBSERVED_DATA, MISSING_DATA, MISSING_DATA,MISSING_DATA]
  #modelfile = "models/pairhmm12_switching_n240_fixalignment=false.jls"

  modelname = basename(modelfile)
  fixAlignment = true
  outputdir = string("logs/",modelname,"_",benchmarksname, "_", join(mask),"_fixalignment=",fixAlignment,"/")
  println(outputdir)
  B = 1.0
  mkpath(outputdir)


  cornercut = 125

  ser = open(modelfile,"r")
  modelparams::ModelParameters = deserialize(ser)
  close(ser)
  #modelparams.prior.tprior = Gamma(1.05,0.25)
  prior = modelparams.prior
  obsnodes = modelparams.obsnodes
  hmminitprobs = modelparams.hmminitprobs
  hmmtransprobs = modelparams.hmmtransprobs
  numHiddenStates = length(hmminitprobs)
  export_json(modelparams, string(modelfile, ".json"))
  #exit()

  write_hiddenstates(modelparams, string(outputdir, "hiddenstates.txt"))
  print_hiddenstates(modelparams, "test")

  println("use_switching", obsnodes[1].useswitching)
  println("H=", numHiddenStates)
  outfile = open(string(outputdir, "benchmarks",numHiddenStates,".txt"), "w")
  write(outfile, "mask\tphi_homologue\tpsi_homologue\tphi_predicted\tpsi_predicted\n")

  tunemcmc = TuneMCMC[]
  for i=1:length(pairs)
    push!(tunemcmc, TuneMCMC(4))
  end




  for k=1:length(pairs)
    jsondict = Dict()
    distancefile = open(string(outputdir, "distances",k,".txt"), "w")

    avg_phi_ranks = Float64[]
    avg_psi_ranks = Float64[]
    avg_phi_psi_ranks = Float64[]
    avg_phi_psi_ranks2 = Float64[]
    inputalign1 = pairs[k].align1
    inputalign2 = pairs[k].align2
    input = pairs[k].seqpair
    seq1, seq2 = masksequences(input.seq1, input.seq2, mask)
    println(seq1.name)
    println(seq2.name)
    masked = SequencePair(0,seq1, seq2)

    current_sample = tkf92(1, rng, AlignmentHMM.HMMCache(seq1.length+1, seq2.length+1, modelparams.numHiddenStates, cornercut), AlignmentHMM.DatalikCache(seq1.length, seq2.length, modelparams.numHiddenStates), masked, PairParameters(), modelparams, B, cornercut, true, inputalign1, inputalign2)[2][1]
    current_sample.seqpair.id = k


   #mcmc_sequencepair(citer::Int, niter::Int, samplerate::Int, rng::AbstractRNG, initialSample::SequencePairSample, modelparams::ModelParameters, cornercut::Int=100, fixAlignment::Bool=false, writeoutput::Bool=false, outputdir::AbstractString="")

    seqpaircornercut = findcornertcutbound(current_sample.seqpair, current_sample.align1, current_sample.align2)
    println(seqpaircornercut)
    for q=1:4
      ret = mcmc_sequencepair(tunemcmc[k], 0, 50, 1, rng, current_sample, modelparams, 1.0, seqpaircornercut[2], fixAlignment, true, outputdir,false,4)
      tunemcmc[k] = ret[5]
    end
    ret = mcmc_sequencepair(tunemcmc[k], 0, 2500, 1, rng, current_sample, modelparams, 1.0, seqpaircornercut[2], fixAlignment, true, outputdir,false,4)

    samples = ret[3]
    nsamples = length(ret[3])
    println(nsamples)
    burnin = round(Int, max(1,nsamples/2))
    hiddencount1 = ret[6]
    hiddencount2 = ret[7]
    regimes1 = ret[8]
    regimes2 = ret[9]
    mcmclogger = ret[10]

    println(mcmclogger.dict)

    if obsnodes[1].useswitching
      computeJumpPosteriorProbabilities(modelparams, samples)
    end

    samples = samples[burnin:2:end]
    filled_pairs = [sample_missing_values(rng, obsnodes, sample) for sample in samples]

    mpdalign1, mpdalign2, posterior_probs = mpdalignment(samples)
    jsondict["mpdalign1"] = string(getaminoacidalignment(seq1, mpdalign1))
    jsondict["mpdalign2"] = string(getaminoacidalignment(seq2, mpdalign2))
    jsondict["alignment_posterior_probs"] = posterior_probs
    jsondict[string("protein1:sequence")] = getaasequence(input.seq1)
    jsondict[string("protein2:sequence")] = getaasequence(input.seq2)
    jsondict[string("protein1:name")] = input.seq1.name
    jsondict[string("protein2:name")] = input.seq2.name
    jsondict[string("protein1:phi")] = input.seq1.phi
    jsondict[string("protein1:psi")] = input.seq1.psi
    jsondict[string("protein2:phi")] = input.seq2.phi
    jsondict[string("protein2:psi")] = input.seq2.psi
    jsondict["protein1:length"] = seq1.length
    jsondict["protein2:length"] = seq2.length
    jsondict["protein1:hiddencount"] = hiddencount1
    jsondict["protein2:hiddencount"] = hiddencount2
    jsondict["protein1:regimes"] = regimes1
    jsondict["protein2:regimes"] = regimes2
    jsondict["protein1:mask:sequence"] = mask[1]
    jsondict["protein1:mask:angles"] = mask[2]
    jsondict["protein1:mask:ss"] = MISSING_DATA
    jsondict["protein2:mask:sequence"] = mask[3]
    jsondict["protein2:mask:angles"] = mask[4]
    jsondict["protein2:mask:ss"] = MISSING_DATA

    mpdwriter = open(string(outputdir, "mpdalignment",k,".txt"), "w")
    write(mpdwriter,string(getaminoacidalignment(samples[end].seqpair.seq1, mpdalign1), "\n"))
    write(mpdwriter,string(getaminoacidalignment(samples[end].seqpair.seq2, mpdalign2), "\n"))
    write(mpdwriter,string(getssalignment(samples[end].seqpair.seq1, mpdalign1), "\n"))
    write(mpdwriter,string(getssalignment(samples[end].seqpair.seq2, mpdalign2), "\n"))
    for i=1:length(posterior_probs)
      write(mpdwriter, string(i,"\t",posterior_probs[i],"\n"))
    end
    close(mpdwriter)


    phi = Float64[]
    psi = Float64[]
    distx = Float64[]
    disty = Float64[]

    for i=1:filled_pairs[1].seq2.length
      phi_i = Float64[]
      psi_i = Float64[]
      distyi = Float64[]
      disttrue = Float64[]
      distcount = 0.0
      disttotal = 0.0
      rmsd_phi_i = Float64[]
      rmsd_psi_i = Float64[]
      rmsd_phi_psi_i = Float64[]
      rmsd_d = Float64[]
      for sm=1:length(filled_pairs)
        seqpair = filled_pairs[sm]
        seqpairsample = samples[sm]
        if seqpair.seq2.phi[i] > -100.0
          push!(phi_i, seqpair.seq2.phi[i])
          if input.seq2.phi[i] > -100.0
            d = angular_rmsd(input.seq2.phi[i], seqpair.seq2.phi[i])
            if d >= 0.0 && !isnan(d)
              push!(rmsd_phi_i, d)
            end
          end
        end

        if seqpair.seq2.psi[i] > -100.0
          push!(psi_i, seqpair.seq2.psi[i])
          if input.seq2.psi[i] > -100.0
            d = angular_rmsd(input.seq2.psi[i], seqpair.seq2.psi[i])
            if d >= 0.0 && !isnan(d)
              push!(rmsd_psi_i, d)
            end

            d1d2 = angular_rmsd(input.seq2.phi[i], seqpair.seq2.phi[i], input.seq2.psi[i], seqpair.seq2.psi[i])
            if d1d2 >= 0.0 && !isnan(d1d2)
              push!(rmsd_phi_psi_i, d1d2)
            end
          end
        end

        f2 = angular_rmsd(input.seq2.phi[i], seqpair.seq2.phi[i], input.seq2.psi[i], seqpair.seq2.psi[i])
        if f2 >= radius && !isnan(f2)
          push!(distyi, f2)
        end
        sampleconf1, sampleconf2 = getconfigurations(seqpairsample.align1, seqpairsample.align2)
        if sampleconf2[i] > 0
          f1 = angular_rmsd(input.seq2.phi[i], input.seq1.phi[sampleconf2[i]], input.seq2.psi[i], input.seq1.psi[sampleconf2[i]])
          if f1 >= radius && !isnan(f1)
            push!(disttrue, f1)
          end
          if f1 >= radius && !isnan(f1) && f2 >= radius && !isnan(f2)
            if f2 < f1
              distcount += 1.0
            end
            disttotal += 1.0
          end
        end
      end

      jsondict[string("protein2:pos",i,":dist_true")] = disttrue
      jsondict[string("protein2:pos",i,":dist_sampled")] = distyi


      conf1, conf2 = getconfigurations(mpdalign1, mpdalign2)
      jsondict["mpdconf1"] = conf1
      jsondict["mpdconf2"] = conf2
      conf1, conf2 = getconfigurations(inputalign1, inputalign2)



      if conf2[i] > 0
        d = angular_rmsd(input.seq2.phi[i], input.seq1.phi[conf2[i]])
        if d >= radius && !isnan(d)
          pr = percentilerank(rmsd_phi_i, d)
          if pr >= 0.0 && !isnan(pr)
            push!(avg_phi_ranks, pr)
          end
        end
        d = angular_rmsd(input.seq2.psi[i], input.seq1.psi[conf2[i]])
        if d >= radius && !isnan(d)
          pr = percentilerank(rmsd_psi_i, d)
          if pr >= 0.0 && !isnan(pr)
            push!(avg_psi_ranks, pr)
          end
        end

        d1d2 = angular_rmsd(input.seq2.phi[i], input.seq1.phi[conf2[i]], input.seq2.psi[i], input.seq1.psi[conf2[i]])
        if d1d2 >= radius && !isnan(d1d2)
          for z=1:length(rmsd_phi_psi_i)

            if z % 50 == 0
              push!(distx, d1d2)
              push!(disty, rmsd_phi_psi_i[z])
            end
          end

          pr =  percentilerank(rmsd_phi_psi_i, d1d2)
          if pr >= 0.0 && !isnan(pr)

            push!(avg_phi_psi_ranks, pr)
          end
        end
      end

      jsondict[string("protein2:pos",i,":dist_rank_input")] = -1.0
      if conf2[i] > 0
        d1d2 = angular_rmsd(input.seq2.phi[i], input.seq1.phi[conf2[i]], input.seq2.psi[i], input.seq1.psi[conf2[i]])
        jsondict[string("protein2:pos",i,":dist_rank_input")] = percentilerank(distyi, d1d2)
      end

      jsondict[string("protein2:pos",i,":sampled_phi")] = phi_i
      jsondict[string("protein2:pos",i,":sampled_psi")] = psi_i
      jsondict[string("protein2:pos",i,":dist_rank_sample")] =  distcount/disttotal
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
          d1 = angular_rmsd(input.seq2.phi[i], input.seq1.phi[conf2[i]])
          d2 = angular_rmsd(input.seq2.psi[i], input.seq1.psi[conf2[i]])
          d3 = angular_rmsd(input.seq2.phi[i], input.seq1.phi[conf2[i]],input.seq2.psi[i], input.seq1.psi[conf2[i]])

          push!(xvals, input.seq1.phi[conf2[i]])
          push!(yvals, input.seq1.psi[conf2[i]])
          l1 = string("A (", aminoacids[input.seq1.seq[conf2[i]]],")")
          write(distancefile, string(i,"\t",d1,"\t",d2,"\t",d3,"\n"))
          push!(labels, l1)
        end
        push!(xvals, input.seq2.phi[i])
        push!(yvals, input.seq2.psi[i])
        l2 = string("B (", aminoacids[input.seq2.seq[i]],")")
        push!(labels, l2)

        outputdir2  = string(outputdir,"structure_",k,"/")
        mkpath(outputdir2)

        rank = angular_rmsd_rank(phi_i, psi_i, input.seq2.phi[i], input.seq2.psi[i], 0.5)
        if rank >= 0.0
          push!(avg_phi_psi_ranks2, rank)
        end
      end
    end


    align1 = inputalign1
    align2 = inputalign2

    j1,j2 = computeJumpPosteriorProbabilities(modelparams, samples)
    jsondict[string("protein1:jump_posterior_prob")] = j1
    jsondict[string("protein2:jump_posterior_prob")] = j2
    jumpfile = open(string("jump",k,"_s1.txt"),"w")
    for j=1:length(j1)
      write(jumpfile, string(j,"\t",j1[j],"\n"))
    end
    close(jumpfile)
    jumpfile = open(string("jump",k,"_s2.txt"),"w")
    for j=1:length(j2)
      write(jumpfile, string(j,"\t",j2[j],"\n"))
    end
    close(jumpfile)


    for key in keys(mcmclogger.dict)
      jsondict[string("mcmc.",key)] = mcmclogger.dict[key]
    end


    jsondict["dist_a_b"] = distx
    jsondict["dist_a_sample"] = disty
    jsonfilename = string(outputdir,"json",k,"_",join(mask,""),".txt")
    if fixAlignment
      jsonfilename = string(outputdir,"json",k,"_",join(mask,""),"_fix.txt")
    end
    jsonout = open(jsonfilename,"w")
    JSON.print(jsonout, jsondict)
    close(jsonout)

    println("W",length(distx))
    println("X", length(disty))
    diagx = [0.0, 4.5]
    diagy = [0.0, 4.5]
    if length(distx) > 0 && length(disty) > 0
      #=
      p2 = plot(layer(x=diagx, y=diagy, Geom.line), layer(x=distx, y=disty, Geom.point), Coord.Cartesian(xmin=0.0, xmax=Float64(4.5), ymin=0.0, ymax=Float64(4.5)))
      draw(SVG(string(outputdir,"dist",k,".svg"), 5inch, 5inch), p2)
      p3 = plot(layer(x=diagx, y=diagy, Geom.line), layer(x=distx, y=disty, Geom.histogram2d(xbincount=25, ybincount=25)), Coord.Cartesian(xmin=0.0, xmax=Float64(4.5), ymin=0.0, ymax=Float64(4.5)))
      draw(SVG(string(outputdir,"hist",k,".svg"), 5inch, 5inch), p3)
      =#
    end

    println("Homologue:\tphi=", angular_rmsd(input.seq2.phi, input.seq1.phi, align2, align1),"\tpsi=", angular_rmsd(input.seq2.psi, input.seq1.psi, align2, align1))
    println("Predicted:\tphi=", angular_rmsd(input.seq2.phi, phi), "\tpsi=", angular_rmsd(input.seq2.psi, psi))
    #println("Rank:\tphi=", angular_rmsd(input.seq2.phi, phi), "\tpsi=", angular_rmsd(input.seq2.psi, psi))

    println("A",sum(avg_phi_ranks)/length(avg_phi_ranks))
    println("B",sum(avg_psi_ranks)/length(avg_psi_ranks))
    rmsdphipsi = angular_rmsd(input.seq2.phi, input.seq1.phi, input.seq2.psi, input.seq1.psi, align2, align1)
    write(outfile, join(mask, ""), "\t", string(angular_rmsd(input.seq2.phi, input.seq1.phi, align2, align1)), "\t", string(angular_rmsd(input.seq2.psi, input.seq1.psi, align2, align1),"\t",rmsdphipsi, "\t"))
    rmsdphipsi = angular_rmsd(input.seq2.phi, phi,input.seq2.psi, psi)
    write(outfile, string(angular_rmsd(input.seq2.phi, phi)), "\t", string(angular_rmsd(input.seq2.psi, psi),"\t",rmsdphipsi,"\t"))
    write(outfile, string(sum(avg_phi_ranks)/length(avg_phi_ranks), "\t", sum(avg_psi_ranks)/length(avg_psi_ranks), "\t", sum(avg_phi_psi_ranks)/length(avg_phi_psi_ranks),"\t",sum(avg_phi_psi_ranks2)/length(avg_phi_psi_ranks2),"\n"))
    flush(outfile)
    close(distancefile)


  end
  close(outfile)
end

using Cubature
using Grid
using NLopt
export computemarginallikelihoods
function computemarginallikelihoods()
  srand(984181083751401)
  rng = MersenneTwister(2412402531025555)

  #benchmarksfilepath = "data/glob.txt"
  #benchmarksfilepath = "data/holdout_data.txt"
  benchmarksfilepath = "data/holdout_data_diverse_new.txt"
  #benchmarksfilepath = "data/2jp1_ltn40.pdb.txt"
  benchmarksname = basename(benchmarksfilepath)

  #pairs = shuffle!(rng, load_sequences_and_alignments(benchmarksfilepath))
  pairs = load_sequences_and_alignments(benchmarksfilepath)
  #pairs = load_sequences_and_alignments("data/glob.txt")
  radius = 0.0

  #modelfile = "models/pairhmm12_switching_n25_fixalignment=false.jls"
  #modelfile = "models/pairhmm4_switching_n10_fixalignment=false.jls"
  modelfile = "models/pairhmm8_noswitching_n10_fixalignment=false.jls"
  #modelfile = "models/pairhmm12_switching_n25_fixalignment=false.jls"
  #modelfile = "models/pairhmm72_switching_n674_fixalignment=false.jls"
  mask = Int[OBSERVED_DATA, OBSERVED_DATA,OBSERVED_DATA, OBSERVED_DATA, OBSERVED_DATA,OBSERVED_DATA]
  #modelfile = "models/pairhmm12_switching_n240_fixalignment=false.jls"

  modelname = basename(modelfile)
  fixAlignment = true
  outputdir = string("logs2/",modelname,"_",benchmarksname, "_", join(mask),"_fixalignment=",fixAlignment,"/")
  println(outputdir)

  Ntemps = 30
  Bs = linspace(0.0,1.0,Ntemps)
  #=
  Bs = Float64[]
  alpha = 3.0
  for i=1:Ntemps
    x = (i / Float64(Ntemps+1)) - 0.5
    push!(Bs, exp(alpha*x)/(exp(alpha*x)+exp(-alpha*x)))
  end
  Bs -= Bs[1]
  Bs /= Bs[end]=#
  println(Bs)

  B = 1.0
  mkpath(outputdir)

  ratios = Float64[]

  cornercut = 125

  ser = open(modelfile,"r")
  modelparams::ModelParameters = deserialize(ser)
  close(ser)

  prior = modelparams.prior
  obsnodes = modelparams.obsnodes
  hmminitprobs = modelparams.hmminitprobs
  hmmtransprobs = modelparams.hmmtransprobs
  numHiddenStates = length(hmminitprobs)


  out2 = open("marginallikelihoods.txt","w")

  for k=1:10
    inputalign1 = pairs[k].align1
    inputalign2 = pairs[k].align2
    input = pairs[k].seqpair
    seq1, seq2 = masksequences(input.seq1, input.seq2, mask)
    masked = SequencePair(0,seq1, seq2)
    datalikcache = AlignmentHMM.DatalikCache(seq1.length, seq2.length, modelparams.numHiddenStates)
    alignmentcache = AlignmentHMM.HMMCache(1, 1, modelparams.numHiddenStates, 1)
    if !fixAlignment
      alignmentcache = AlignmentHMM.HMMCache(seq1.length+1, seq2.length+1, modelparams.numHiddenStates, cornercut)
    end
    current_sample = AlignmentHMM.tkf92(1, rng, alignmentcache, datalikcache, masked, PairParameters(), modelparams, B, cornercut, true, inputalign1, inputalign2)[2][1]
    current_sample.seqpair.id = k

    maxt = 200.0
    localObjectiveFunction = ((param, grad) -> computelikelihood(current_sample, modelparams, true, cornercut, param[1]))
    opt = Opt(:LN_COBYLA, 1)
    lower_bounds!(opt, Float64[1e-10])
    upper_bounds!(opt, Float64[maxt])
    max_objective!(opt, localObjectiveFunction)
    (minf,minx,ret) = optimize(opt, Float64[0.1])
    maxll = minf

    integral,error = hquadrature(x -> exp(computelikelihood(current_sample, modelparams, true, cornercut, x)-maxll), 0.0,maxt)

    marginalloglikelihood = maxll + log(integral)
    println("maxll=",maxll,"\tintegral=",integral, "\tp(D|M)=", marginalloglikelihood, "\terror=", error)
  end
end

function computelikelihood(current_sample::SequencePairSample, modelparams::ModelParameters, fixAlignment::Bool, cornercut::Int, t::Float64)
  current_sample.params.t = t
  seq1 = current_sample.seqpair.seq1
  seq2 = current_sample.seqpair.seq2
  datalikcache = AlignmentHMM.DatalikCache(seq1.length, seq2.length, modelparams.numHiddenStates)
  alignmentcache = AlignmentHMM.HMMCache(seq1.length+1, seq2.length+1, modelparams.numHiddenStates, cornercut)
  ll =  logpdf(modelparams.prior.tprior, t) + AlignmentHMM.tkf92(1, MersenneTwister(49018401841081), alignmentcache, datalikcache, current_sample.seqpair, current_sample.params, modelparams, -1.0, cornercut, true, current_sample.align1, current_sample.align2, false, current_sample.states, 0, 1, 1, false)[1]
  return ll
end

function computemarginal(seed::Int, current_sample::SequencePairSample, modelparams::ModelParameters, Bcurr::Float64, Bprev::Float64, fixAlignment::Bool, cornercut::Int, outputdir::AbstractString)
  seq1 = current_sample.seqpair.seq1
  seq2 = current_sample.seqpair.seq2

  tunemcmc = TuneMCMC(4)
  rng = MersenneTwister(seed)
  for q=1:4
    ret = mcmc_sequencepair2(tunemcmc, 0, 40, 1, rng, current_sample, modelparams, Bprev, cornercut, fixAlignment, true, outputdir,false,4)
    tunemcmc = ret[5]
  end
  ret = mcmc_sequencepair2(tunemcmc, 0, 160, 1, rng, current_sample, modelparams, Bprev, cornercut, fixAlignment, true, outputdir,false,4)

  newsamples = ret[3]
  v = Float64[]
  ts = Float64[]
  alignmentcache = AlignmentHMM.HMMCache(1, 1, modelparams.numHiddenStates, 1)
  if !fixAlignment
    alignmentcache = AlignmentHMM.HMMCache(seq1.length+1, seq2.length+1, modelparams.numHiddenStates, cornercut)
  end
  datalikcache = AlignmentHMM.DatalikCache(seq1.length, seq2.length, modelparams.numHiddenStates)
  tic()

  avgll = -Inf
  for current_sample in newsamples
    currentll2, dummy1, dummy2 = AlignmentHMM.tkf92(1, rng, alignmentcache, datalikcache, current_sample.seqpair, current_sample.params, modelparams, -1.0, cornercut, true, current_sample.align1, current_sample.align2, false, current_sample.states, 0, 1, 1, false)
    push!(v,currentll2)
    push!(ts, current_sample.params.t)
    avgll = logsumexp(avgll, currentll2*(Bcurr-Bprev))
  end
  toc()
  avgll -= log(length(newsamples))
  avgll2 = logsumexpstable(v, 1, length(v)) - log(length(newsamples))
  avgll3 = mean(v)
  return avgll, avgll2, avgll3, Bprev, v, ts
end

function simulatestationary()
   srand(98418108751401)
  rng = MersenneTwister(242402531025555)


  pairs = shuffle!(rng, load_sequences_and_alignments("data/data_diverse_new.txt"))
  phi = Float64[]
  psi = Float64[]
  for p in pairs
    pair = p.seqpair
    for i=1:pair.seq1.length
      push!(phi, pair.seq1.phi[i])
      push!(psi, pair.seq1.psi[i])
    end
    for i=1:pair.seq2.length
      push!(phi, pair.seq2.phi[i])
      push!(psi, pair.seq2.psi[i])
    end
  end

  jsondict = Dict()
  jsondict["phi"] = phi
  jsondict["psi"] = psi

  for aa=1:20
    phi = Float64[]
    psi = Float64[]
    for p in pairs
      pair = p.seqpair
      for i=1:pair.seq1.length
        if pair.seq1.seq[i] == aa
          push!(phi, pair.seq1.phi[i])
          push!(psi, pair.seq1.psi[i])
        end
      end
      for i=1:pair.seq2.length
        if pair.seq2.seq[i] == aa
          push!(phi, pair.seq2.phi[i])
          push!(psi, pair.seq2.psi[i])
        end
      end
    end
    jsondict[string("phi",aa)] = phi
    jsondict[string("psi",aa)] = psi
  end
  jsonout = open("datasamples.json", "w")
  JSON.print(jsonout, jsondict)
  close(jsonout)

  modelfile = "models/pairhmm64_switching_n674_fixalignment=false.jls"
  mask = Int[MISSING_DATA, MISSING_DATA,MISSING_DATA, MISSING_DATA, MISSING_DATA,MISSING_DATA]

  modelname = basename(modelfile)
  fixAlignment = true
  outputdir = ""
  #println(outputdir)
  #mkpath(outputdir)
  cornercut = 125
  B = 1.0

  ser = open(modelfile,"r")
  modelparams::ModelParameters = deserialize(ser)
  close(ser)
  prior = modelparams.prior
  obsnodes = modelparams.obsnodes
  hmminitprobs = modelparams.hmminitprobs
  hmmtransprobs = modelparams.hmmtransprobs
  numHiddenStates = length(hmminitprobs)
  len = 250
  inputalign1 = Int[i for i=1:len]
  inputalign2 = Int[i for i=1:len]
  seq1, seq2 = Sequence(len), Sequence(len)

  seq1, seq2 = masksequences(seq1, seq2, mask)
  masked = SequencePair(0,seq1, seq2)
  tunemcmc = TuneMCMC(4)
  current_sample = tkf92(1, rng, AlignmentHMM.HMMCache(seq1.length+1, seq2.length+1, modelparams.numHiddenStates, cornercut), AlignmentHMM.DatalikCache(seq1.length, seq2.length, modelparams.numHiddenStates), masked, PairParameters(), modelparams, B, cornercut, true, inputalign1, inputalign2)[2][1]
  ret = mcmc_sequencepair(tunemcmc, 0, 2000, 1, rng, current_sample, modelparams, 100, fixAlignment, true, outputdir,false,10)
  samples = ret[3]
  nsamples = length(ret[3])
  println(nsamples)
  burnin = round(Int, max(1,nsamples/2))
  hiddencount1 = ret[6]
  hiddencount2 = ret[7]
  regimes1 = ret[8]
  regimes2 = ret[9]
  mcmclogger = ret[10]
  samples = samples[burnin:end]
  filled_pairs = [sample_missing_values(rng, obsnodes, sample) for sample in samples]

  phi = Float64[]
  psi = Float64[]
  for pair in filled_pairs
    for i=1:pair.seq1.length
      push!(phi, pair.seq1.phi[i])
      push!(psi, pair.seq1.psi[i])
    end
  end

  jsondict = Dict()
  jsondict["phi"] = phi
  jsondict["psi"] = psi

  for aa=1:20
    phi = Float64[]
    psi = Float64[]
    for pair in filled_pairs
      for i=1:pair.seq1.length
        if pair.seq1.seq[i] == aa
          push!(phi, pair.seq1.phi[i])
          push!(psi, pair.seq1.psi[i])
        end
        #=
         if pair.seq2.seq[i] == aa
          push!(phi, pair.seq2.phi[i])
          push!(psi, pair.seq2.psi[i])
        end=#
      end
    end
    jsondict[string("phi",aa)] = phi
    jsondict[string("psi",aa)] = psi
  end

  jsonout = open("modelsamples.json", "w")
  JSON.print(jsonout, jsondict)
  close(jsonout)

end

function simulatestationary2()
   srand(98418108751401)
  rng = MersenneTwister(242402531025555)


  contingency = zeros(Float64,2,2)
  B = 1.0

  pairs = shuffle!(rng, load_sequences_and_alignments("data/data_diverse.txt"))
  phi = Float64[]
  psi = Float64[]
  for p in pairs
    pair = p.seqpair
    for i=1:pair.seq1.length
      push!(phi, pair.seq1.phi[i])
      push!(psi, pair.seq1.psi[i])
    end
    for i=1:pair.seq2.length
      push!(phi, pair.seq2.phi[i])
      push!(psi, pair.seq2.psi[i])
    end
  end

  jsondict = Dict()
  jsondict["phi"] = phi
  jsondict["psi"] = psi

  for aa=1:20
    phi = Float64[]
    psi = Float64[]
    for p in pairs
      pair = p.seqpair
      align1 = p.align1
      align2 = p.align2
      for a=1:length(align1)
        if align1[a] > 0
          if align2[a] > 0
            #println(pair.seq1.ss[align1[a]])
            if pair.seq1.ss[align1[a]] == 1
              contingency[1,1] += 1.0
            elseif pair.seq1.ss[align1[a]] == 2
              contingency[1,2] += 1.0
            end
          else
            if pair.seq1.ss[align1[a]] == 1
              contingency[2,1] += 1.0
            elseif pair.seq1.ss[align1[a]] == 2
              contingency[2,2] += 1.0
            end
          end
        end
      end

      for i=1:pair.seq1.length
        if pair.seq1.seq[i] == aa
          push!(phi, pair.seq1.phi[i])
          push!(psi, pair.seq1.psi[i])
        end
      end
      for i=1:pair.seq2.length
        if pair.seq2.seq[i] == aa
          push!(phi, pair.seq2.phi[i])
          push!(psi, pair.seq2.psi[i])
        end
      end
    end
    jsondict[string("phi",aa)] = phi
    jsondict[string("psi",aa)] = psi
  end
  jsonout = open("datasamples.json", "w")
  JSON.print(jsonout, jsondict)
  close(jsonout)
   println(contingency)

  modelfile = "models/pairhmm64_switching_n674_fixalignment=false.jls"
  mask = Int[MISSING_DATA, MISSING_DATA,MISSING_DATA, MISSING_DATA, MISSING_DATA,MISSING_DATA]

  modelname = basename(modelfile)
  fixAlignment = true
  outputdir = ""
  #println(outputdir)
  #mkpath(outputdir)
  cornercut = 125

  ser = open(modelfile,"r")
  modelparams::ModelParameters = deserialize(ser)
  close(ser)
  prior = modelparams.prior
  obsnodes = modelparams.obsnodes
  hmminitprobs = modelparams.hmminitprobs
  hmmtransprobs = modelparams.hmmtransprobs
  numHiddenStates = length(hmminitprobs)
  len = 250
  inputalign1 = Int[i for i=1:len]
  inputalign2 = Int[i for i=1:len]
  seq1, seq2 = Sequence(len), Sequence(len)

  seq1, seq2 = masksequences(seq1, seq2, mask)
  masked = SequencePair(0,seq1, seq2)
  tunemcmc = TuneMCMC(4)
  current_sample = tkf92(1, rng, AlignmentHMM.HMMCache(seq1.length+1, seq2.length+1, modelparams.numHiddenStates, cornercut), AlignmentHMM.DatalikCache(seq1.length, seq2.length, modelparams.numHiddenStates), masked, PairParameters(), modelparams, B, cornercut, true, inputalign1, inputalign2)[2][1]
  ret = mcmc_sequencepair(tunemcmc, 0, 2000, 1, rng, current_sample, modelparams, 100, fixAlignment, true, outputdir,false,10)
  samples = ret[3]
  nsamples = length(ret[3])
  println(nsamples)
  burnin = round(Int, max(1,nsamples/2))
  hiddencount1 = ret[6]
  hiddencount2 = ret[7]
  regimes1 = ret[8]
  regimes2 = ret[9]
  mcmclogger = ret[10]
  samples = samples[burnin:end]
  filled_pairs = [sample_missing_values(rng, obsnodes, sample) for sample in samples]

  phi = Float64[]
  psi = Float64[]
  for pair in filled_pairs
    for i=1:pair.seq1.length
      push!(phi, pair.seq1.phi[i])
      push!(psi, pair.seq1.psi[i])
    end
  end

  jsondict = Dict()
  jsondict["phi"] = phi
  jsondict["psi"] = psi

  for aa1=1:20
    for aa2=1:20
      phi1 = Float64[]
      psi1 = Float64[]
      phi2 = Float64[]
      psi2 = Float64[]
      dist = Float64[]
      for j=1:length(filled_pairs)
        pair = filled_pairs[j]
        #=
        align1 = samples[j].align1
        align2 = samples[j].align2

        for a=1:length(align1)
          if align1[a] > 0
            if align2[a] > 0
              #println(pair.seq1.ss[align1[a]])
              if pair.seq1.ss[align1[a]] == 1
                contingency[1,1] += 1.0
              elseif pair.seq1.ss[align1[a]] == 2
                contingency[1,2] += 1.0
              end
            else
              if pair.seq1.ss[align1[a]] == 1
                contingency[2,1] += 1.0
              elseif pair.seq1.ss[align1[a]] == 2
                contingency[2,2] += 1.0
              end
            end
          end
        end=#

        for i=1:pair.seq1.length

          if pair.seq1.seq[i] == aa1 && pair.seq2.seq[i] == aa2
            push!(phi1, pair.seq1.phi[i])
            push!(psi1, pair.seq1.psi[i])
            push!(phi2, pair.seq2.phi[i])
            push!(psi2, pair.seq2.psi[i])
            push!(dist, angular_rmsd(pair.seq1.phi[i], pair.seq2.phi[i], pair.seq1.psi[i], pair.seq2.psi[i]))
          end
        end
      end
      jsondict[string("phi1",aa1,"_",aa2)] = phi1
      jsondict[string("psi1",aa1,"_",aa2)] = psi1
      jsondict[string("phi2",aa1,"_",aa2)] = phi2
      jsondict[string("psi2",aa1,"_",aa2)] = psi2
      jsondict[string("dist",aa1,"_",aa2)] = dist
    end
  end



  jsonout = open("modelsamples.json", "w")
  JSON.print(jsonout, jsondict)
  close(jsonout)

end

#=
function simulation(modelparams::ModelParameters, seq::Sequence)
  t = 1.0
  params = PairParameters()
  seqpair = SequencePair(0,seq, Sequence(seq.length))

  align1 = Int[i for i=1:seq.length]
  align2 = Int[i for i=1:seq.length]
  states = Int[1 for i=1:seq.length]

  rng = MersenneTwister(210104494032)
  tkf92(datalikcache, 1, rng, seqpair::SequencePair, params, modelparams, 100, true, align1, align2, false, states)


  steps = 100
  dt = t / Float64(steps)
  params.t = dt


  samples = Sequence[]
  push!(samples,seq)
  for i=1:steps
    next = Sequence(seq.length)
    seqpair = SequencePair(0,samples[i], next)
    sample = SequencePairSample(seqpair, params)
    sample.align1 = align1
    sample.align2 = align2
    sample.states = states
    newseqpair = sample_missing_values(rng, modelparams.obsnodes, sample)
    push!(samples,newseqpair.seq2)

  end



  out = open("animation2.txt", "w")
  for j=1:length(samples)
    sample = samples[j]
    write(out, string(getaminoacidalignment(sample, align1)),"\n")
    write(out, string(sample.phi),"\n")
    write(out, string(sample.psi),"\n")
  end
  close(out)


  for i=1:length(align1)
    phi_i = Float64[]
    psi_i = Float64[]
    aax = Float64[]
    aay = Float64[]
    labels = AbstractString[]
    layers = Gadfly.Layer[]
    for j=1:length(samples)
      seq = samples[j]
      l = string(aminoacids[seq.seq[i]])
      if j > 1 && (abs(seq.phi[i]-phi_i[end]) > pi || abs(seq.psi[i]-psi_i[end]) > pi)
        append!(layers, layer(x=aax, y=aay, label=labels, Geom.label))
        append!(layers, layer(x=phi_i, y=psi_i, Geom.line))
        phi_i = Float64[]
        psi_i = Float64[]
        aax = Float64[]
        aay = Float64[]
        labels = AbstractString[]
      end

      push!(phi_i, seq.phi[i])
      push!(psi_i, seq.psi[i])
      push!(aax, seq.phi[i])
      push!(aay, seq.psi[i])
      push!(labels, l)
    end
    append!(layers, layer(x=aax, y=aay, label=labels, Geom.label))
    append!(layers, layer(x=phi_i, y=psi_i, Geom.line))
    p = plot(layers, Coord.Cartesian(xmin=Float64(-pi), xmax=Float64(pi), ymin=Float64(-pi), ymax=Float64(pi)))
    draw(SVG(string("trajectory",i,".svg"), 5inch, 5inch), p)
  end
end=#
