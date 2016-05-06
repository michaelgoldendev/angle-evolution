include("UtilsModule.jl")
#include("NodesModule.jl")

#using UtilsModule
#using NodesModule
#include("AcceptanceLogger.jl")
module Cornercut
  using Formatting
  using Distributions
  using DataStructures
  using UtilsModule
  using NodesModule

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
    n::Int
    m::Int
    numHiddenStates::Int
    cornercut::Int
    cornercutbound::Int
    store::Array{Float64,3}

    function HMMCache(n::Int, m::Int, numHiddenStates::Int, cornercut::Int, fixAlignment::Bool, fixStates::Bool)
      new(n,m,numHiddenStates,cornercut, cornercut + abs(n-m), ones(Float64,n+1,m+1,5)*Inf)
    end
  end

  function putvalue(cache::HMMCache, i::Int, j::Int, alignnode::Int, h::Int, v::Float64)
    cache.store[i+1,j+1,alignnode] = v
  end

  function getvalue(cache::HMMCache, i::Int, j::Int, alignnode::Int, h::Int)
    return cache.store[i+1,j+1,alignnode]

  end

  function tkf92forward(aapairnode::AAPairNode, seqpair::SequencePair, t::Float64, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, cornercut::Int=10000000)
    if i < 0 || j < 0
      return -Inf
    end

    h = 1
    v = getvalue(cache,i,j,alignnode,h)
    if v != Inf
      return v
    end

    if abs(i-j) > cache.cornercut
      return -Inf
    elseif i == 0 && j == 0
      if alignnode == START
        return 0.0
      end
    end

    prevlik::Float64 = 0.0
    datalik = 0.0
    sum::Float64 = -Inf
    if alignnode == MATCH || alignnode == XINSERT || alignnode == YINSERT
      if alignnode == MATCH
        if i > 0 && j > 0
          datalik = get_data_lik(aapairnode, seqpair.seq1.seq[i], seqpair.seq2.seq[j], t)
        end
      elseif alignnode == XINSERT
        if i > 0
          datalik = get_data_lik(aapairnode, seqpair.seq1.seq[i])
        end
      elseif alignnode == YINSERT
        if j > 0
          datalik = get_data_lik(aapairnode, seqpair.seq2.seq[j])
        end
      end

      for prevalignnode=1:5
        if hmmparameters.aligntransprobs[prevalignnode, alignnode] > 0.0
          if alignnode == MATCH
            prevlik = tkf92forward(aapairnode, seqpair, t, cache, hmmparameters, i-1, j-1, prevalignnode, cornercut)
            sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+datalik)
          elseif alignnode == XINSERT
            prevlik = tkf92forward(aapairnode, seqpair, t, cache, hmmparameters, i-1, j, prevalignnode, cornercut)
            sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+datalik)
          elseif alignnode == YINSERT
            prevlik = tkf92forward(aapairnode, seqpair, t, cache, hmmparameters, i, j-1, prevalignnode, cornercut)
            sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode]+datalik)
          end
        end
      end
    else
      for prevalignnode=1:5
        if hmmparameters.aligntransprobs[prevalignnode, alignnode] > 0.0
          prevlik = tkf92forward(aapairnode, seqpair, t, cache, hmmparameters, i, j, prevalignnode, cornercut)
          sum = logsumexp(sum, prevlik+hmmparameters.logaligntransprobs[prevalignnode, alignnode])
        end
      end
    end

    putvalue(cache,i,j,alignnode,h,sum)
    return sum
  end

  function tkf92sample(aapairnode::AAPairNode, seqpair::SequencePair, t::Float64, rng::AbstractRNG, cache::HMMCache, hmmparameters::HMMParameters, i::Int, j::Int, alignnode::Int, align1::Array{Int,1}, align2::Array{Int,1}, cornercut::Int=10000000)
    newalignnode::Int = alignnode
    newi::Int = i
    newj::Int = j

    numAlignStates::Int = size(hmmparameters.aligntransprobs,1)

    while !(newalignnode == START && newi == 0 && newj == 0)

      choice = Float64[-Inf for i=1:numAlignStates]

      for prevalignnode=1:numAlignStates
        if hmmparameters.aligntransprobs[prevalignnode, newalignnode] > 0.0
          choice[prevalignnode] = tkf92forward(aapairnode, seqpair, t, cache, hmmparameters,newi,newj, prevalignnode, cornercut)+log(hmmparameters.aligntransprobs[prevalignnode, newalignnode])
        end
      end
      newalignnode = GumbelSample(rng, choice)

      if newalignnode == MATCH
        unshift!(align1, newi)
        unshift!(align2, newj)
        newi = newi-1
        newj = newj-1
      elseif newalignnode == XINSERT
        unshift!(align1, newi)
        unshift!(align2, 0)
        newi = newi-1
        newj = newj
      elseif newalignnode == YINSERT
        unshift!(align1, 0)
        unshift!(align2, newj)
        newi = newi
        newj = newj-1
      end
    end
  end

  function tkf92forward(rng::AbstractRNG, aapairnode::AAPairNode, seqpair::SequencePair, cornercut::Int, params::PairParameters)
    n = seqpair.seq1.length
    m = seqpair.seq2.length
    cache = HMMCache(n,m, 1,cornercut, false, false)
    aligntransprobs = get_alignment_transition_probabilities(params.lambda,params.mu,params.r,params.t)
    hmmparameters = HMMParameters(aligntransprobs, zeros(Float64,2), zeros(Float64,2,2))
    ll = tkf92forward(aapairnode, seqpair, params.t, cache, hmmparameters, n, m, END, cornercut)
    align1 = Int[]
    align2 = Int[]
    tkf92sample(aapairnode, seqpair, params.t, rng, cache, hmmparameters, n, m, END, align1, align2, cornercut)
    return ll, align1, align2
  end

  function maxdistance(align1::Array{Int,1}, align2::Array{Int,1})
    len = length(align1)
    maxdist = 0
    for a=1:len
      if align1[a] > 0 && align2[a] > 0
          maxdist = max(maxdist, abs(align2[a]-align1[a]))
      end
    end
    return maxdist
  end

export findcornertcutbound
function findcornertcutbound(seqpair::SequencePair, inputalign1::Array{Int,1}, inputalign2::Array{Int,1}, incornercut::Int=-1)
    rng = MersenneTwister(242402531025555)
    aapairnode = AAPairNode()
    load_parameters(aapairnode, "resources/lg_LG.PAML.txt")



    n = seqpair.seq1.length
    m = seqpair.seq2.length

    cornercut = 100 + abs(n-m)
    if incornercut > 0
      cornercut = incornercut
    end
    pairparams = PairParameters()

    lower1, upper1 = ones(Int,n)*10000000, zeros(Int,n)
    lower2, upper2 = ones(Int,m)*10000000, zeros(Int,m)

    tunemcmc = TuneMCMC(4)
    current = PairParameters()
    proposed = PairParameters()
    maxk = 4
    currentll, align1, align2 = tkf92forward(rng, aapairnode, seqpair, cornercut, current)
    maxdist = 0
    for k=1:maxk
      logger = AcceptanceLogger()
      maxiter = 60
      if k == maxk
        maxiter = 150
      end
      for iter=1:maxiter
        move = UtilsModule.sample(rng, Float64[1.0, 1.0, 1.0, 1.0])
        if move == 1
          sigma = 0.2*getfactor(tunemcmc, 1)
          d1 = Truncated(Normal(current.lambda, sigma), 0.0, Inf)
          proposed.lambda = rand(d1)
          d2 = Truncated(Normal(proposed.lambda, sigma), 0.0, Inf)
          propratio = logpdf(d2, current.lambda) - logpdf(d1, proposed.lambda)
          movename = "lambda"
        elseif move == 2
          sigma = 0.15*getfactor(tunemcmc, 2)
          d1 = Truncated(Normal(current.ratio, sigma), 0.0, 1.0)
          proposed.ratio = rand(d1)
          d2 = Truncated(Normal(proposed.ratio, sigma), 0.0, 1.0)
          propratio = logpdf(d2, current.ratio) - logpdf(d1, proposed.ratio)
          movename = "ratio"
        elseif move == 3
          sigma = 0.25*getfactor(tunemcmc, 3)
          d1 = Truncated(Normal(current.r, sigma), 0.0, 1.0)
          proposed.r = rand(d1)
          d2 = Truncated(Normal(proposed.r, sigma), 0.0, 1.0)
          propratio = logpdf(d2, current.r) - logpdf(d1, proposed.r)
          movename = "r"
        elseif move == 4
          sigma = 0.1*getfactor(tunemcmc, 4)
          movename = "t"
          d1 = Truncated(Normal(current.t, sigma), 0.0, Inf)
          proposed.t = rand(d1)
          d2 = Truncated(Normal(proposed.t, sigma), 0.0, Inf)
          propratio = logpdf(d2, current.t) - logpdf(d1, proposed.t)
        end

        proposed.mu = proposed.lambda/proposed.ratio
        if(proposed.lambda > 0.0 && proposed.mu > 0.0 && proposed.lambda < proposed.mu && 0.001 < proposed.ratio < 0.999 && 0.001 < proposed.r < 0.999 && proposed.t > 0.0)
          proposedll, palign1, palign2 = tkf92forward(rng, aapairnode, seqpair, cornercut, proposed)
          if(exp(proposedll-currentll+propratio) > rand(rng))
            currentll = proposedll
            current = PairParameters(proposed)
            align1 = palign1
            align2 = palign2
            logAccept!(logger, movename)
          else
            proposed = PairParameters(current)
            logReject!(logger, movename)
          end
        end

        if k == maxk && iter % 1 == 0
          #println(getaminoacidalignment(seqpair.seq1, align1))
          #println(getaminoacidalignment(seqpair.seq2, align2))

          maxdist = max(maxdist, maxdistance(align1,align2))

          for a=1:length(align1)
            if align1[a] > 0  && align2[a] > 0
              lower1[align1[a]] = min(lower1[align1[a]], align2[a])
              upper1[align1[a]] = max(upper1[align1[a]], align2[a])
              lower2[align2[a]] = min(lower2[align2[a]], align1[a])
              upper2[align2[a]] = max(upper2[align2[a]], align1[a])
            end
          end
          #println(currentll,"\t", maxdist)
        end
      end
      logacceptance(tunemcmc, 1, getacceptanceratio(logger, "lambda"))
      logacceptance(tunemcmc, 2, getacceptanceratio(logger, "ratio"))
      logacceptance(tunemcmc, 3, getacceptanceratio(logger, "r"))
      logacceptance(tunemcmc, 4, getacceptanceratio(logger, "t"))
      println(UtilsModule.list(logger))
    end
    cc = max(20, maxdistance(inputalign1,inputalign2)+10)
    return maxdist, max(cc, max(20,(maxdist-abs(n-m))*2)+abs(n-m)), lower1, upper1, lower2, upper2
  end
end
#=
rng = MersenneTwister(242402531025555)


inputsamples = shuffle!(rng, load_sequences_and_alignments("data/data_diverse.txt"))
pairs = SequencePair[sample.seqpair for sample in inputsamples]

#seqpair = pairs[4]
for seqpair in pairs
  tic()
  println(seqpair.seq1.length, "\t", seqpair.seq2.length)
  cornercut = Cornercut.findcornertcutbound(seqpair, zeros(Int,1), zeros(Int,1))
  #println("Z",cornercut)
  lower1 = cornercut[3]
  upper1 = cornercut[4]
  lower2 = cornercut[5]
  upper2 = cornercut[6]
  for i=1:length(lower1)
    println(i,"\t",lower1[i],"\t",upper1[i])
  end
  toc()
end
=#
