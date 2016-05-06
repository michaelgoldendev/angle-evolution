#include("UtilsModule.jl")
#include("NodesModule.jl")


module AlignmentHMM

  using Formatting
  using Distributions
  using DataStructures
  using UtilsModule
  using NodesModule

  MATCH = 1
  XINSERT = 2
  YINSERT = 3
  START = 4
  END = 5

  export get_alignment_transition_probabilities
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

  export HMMCache
  type HMMCache
    cache::Array{Float64,3}
    n::Int
    m::Int
    numHiddenStates::Int
    cornercut::Int
    usecornercut::Bool

    function HMMCache(n::Int, m::Int, numHiddenStates::Int, cornercut::Int)
      usecornercut = true
      cache = ones(1,1,1)
      width = 1
      if n*m < (cornercut*2+3)*m
        cache = ones((n+1)*(m+1),3,numHiddenStates)*-Inf
        usecornercut = false
      else
        cache = ones((cornercut*2+4)*m,3,numHiddenStates)*-Inf
      end

      new(cache, n+1,m+1,numHiddenStates,cornercut, usecornercut)
    end
  end

  export putvalue
  function putvalue(cache::HMMCache, i::Int, j::Int, alignnode::Int, h::Int, v::Float64)
    if cache.usecornercut
      index = (i-j+cache.cornercut+1)*cache.m+j
      if index > size(cache.cache,1)
        println("resizing cache 1")
        newcache = ones(Float64,index+10,3,cache.numHiddenStates)*-Inf
        for a=1:size(cache.cache,1)
          for b=1:size(cache.cache,2)
            for c=1:size(cache.cache,3)
              newcache[a,b,c] = cache.cache[a,b,c]
            end
          end
        end
        cache.cache = newcache
      end
      cache.cache[index,alignnode,h] = v
    else
      cache.cache[(i-1)*cache.m+j,alignnode,h] = v
    end
  end

  export getvalue
  function getvalue(cache::HMMCache, i::Int, j::Int, alignnode::Int, h::Int)
    if cache.usecornercut
      index = (i-j+cache.cornercut+1)*cache.m+j
      if index > size(cache.cache,1)
        return -Inf
        #=
        println("resizing cache 2")
        newcache = ones(Float64,index+10,3,cache.numHiddenStates)*-Inf
        for a=1:size(cache.cache,1)
          for b=1:size(cache.cache,2)
            for c=1:size(cache.cache,3)
              newcache[a,b,c] = cache.cache[a,b,c]
            end
          end
        end
        cache.cache = newcache
        =#
      end
      return cache.cache[index,alignnode,h]
    else
      return cache.cache[(i-1)*cache.m+j,alignnode,h]
    end
  end

  export resetcache
  function resetcache(cache::HMMCache)
    fill!(cache.cache, -Inf)
  end

  type DatalikCache
    caches::Array{Dict{Int, Float64},1}
    n::Int
    m::Int
    numHiddenStates::Int
    t::Float64
    evocache::Array{Float64,3}
    tcache::Array{Float64,3}
    seq1cache::Array{Float64,2}
    seq2cache::Array{Float64,2}

    function DatalikCache(n::Int, m::Int, numHiddenStates::Int)
      caches = Dict{Int,Float64}[]
      for h=1:numHiddenStates
        d = Dict{Int,Float64}()
        push!(caches, d)
      end
      new(caches, n,m,numHiddenStates, -1.0, ones(Float64, n, m, numHiddenStates)*Inf, ones(Float64, n, m, numHiddenStates)*Inf, ones(Float64, n, numHiddenStates)*Inf, ones(Float64, m, numHiddenStates)*Inf)
    end
  end

  function get_data_lik_cache(cache::DatalikCache, obsnodes::Array{ObservationNode,1}, h::Int, seq1::Sequence, seq2::Sequence, k::Int, l::Int, t::Float64)
    i = k
    j = l
    s1 = seq1
    s2 = seq2
    if cache.evocache[i,j,h] == Inf || cache.tcache[i,j,h] != t
      cache.tcache[i,j,h] = t
      cache.evocache[i,j,h] = get_data_lik(obsnodes[h], s1, s2, i,j, t)
    else
      return cache.evocache[i,j,h]
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

  function tkf92forward(seqpair::SequencePair, t::Float64,  datalikcache::DatalikCache, cache::HMMCache, cache2::Array{Float64,3}, obsnodes::Array{ObservationNode,1}, hmmparameters::HMMParameters, cornercut::Int=1000000, fixAlignment::Bool=false, align1::Array{Int,1}=zeros(Int,1), align2::Array{Int,1}=zeros(Int,1), fixStates::Bool=false, states::Array{Int,1}=zeros(Int,1),partialAlignment::Int=0, startIndex::Int=1, endIndex::Int=1, B::Float64=1.0)
    n = seqpair.seq1.length
    m = seqpair.seq2.length
    numHiddenStates = hmmparameters.numHiddenStates

    tempprevcells = ones(Float64,4,numHiddenStates)*-Inf
    tempprevcells2 = zeros(Float64, numHiddenStates)
    tempdatalik = ones(Float64,numHiddenStates)*-Inf

    if fixStates
      preva = 0
      prevb = 0
      a = 1
      b = 1
      prevalignnode = START
      prevh = 0
      len = length(align1)
      ll = 0.0
      for z=1:len
        preva = a
        prevb = b
        h = states[z]

        hmmtransloglik = 0.0
        if prevh == 0
          hmmtransloglik = hmmparameters.loghmminitprobs[h]
        else
          hmmtransloglik = hmmparameters.loghmmtransprobs[prevh,h]
        end

        if align1[z] != 0 && align2[z] != 0 # MATCH
          a += 1
          b += 1
          logaligntransprobs = hmmparameters.logaligntransprobs[prevalignnode, MATCH]
          datalik = get_data_lik_cache(datalikcache, obsnodes, h, seqpair.seq1, seqpair.seq2, preva, prevb, t)
          ll += logaligntransprobs + hmmtransloglik + datalik
          prevalignnode = MATCH
          prevh = h
        elseif align2[z] == 0 # XINSERT
          a += 1
          logaligntransprobs = hmmparameters.logaligntransprobs[prevalignnode, XINSERT]
          datalik = get_data_lik_cache_seq1(datalikcache, obsnodes, h, seqpair.seq1, preva)
          ll += logaligntransprobs + hmmtransloglik + datalik
          prevalignnode = XINSERT
          prevh = h
        elseif align1[z] == 0 # YINSERT
          b += 1
          logaligntransprobs = hmmparameters.logaligntransprobs[prevalignnode, YINSERT]
          datalik = get_data_lik_cache_seq2(datalikcache, obsnodes, h, seqpair.seq2, prevb)
          ll += logaligntransprobs + hmmtransloglik + datalik
          prevalignnode = YINSERT
          prevh = h
        end
      end

      ll += hmmparameters.logaligntransprobs[prevalignnode, END]

      return ll
    elseif fixAlignment
      for h=1:numHiddenStates
        cache2[1,MATCH,h] = hmmparameters.loghmminitprobs[h]
        cache2[1,XINSERT,h] = hmmparameters.loghmminitprobs[h]
        cache2[1,YINSERT,h] = hmmparameters.loghmminitprobs[h]
      end

      preva = 0
      prevb = 0
      a = 1
      b = 1
      alignnode = 0
      prevalignnode= START
      len = length(align1)
      for z=1:len
        preva = a
        prevb = b
        if align1[z] != 0 && align2[z] != 0 # MATCH
          a += 1
          b += 1
          alignnode = MATCH
        elseif align2[z] == 0 # XINSERT
          a += 1
          alignnode = XINSERT
        elseif align1[z] == 0 # YINSERT
          b += 1
          alignnode = YINSERT
        end

        if preva > 0 && prevb > 0
          maxdatalik = -Inf
          maxsum = -Inf
          if alignnode == MATCH
            for h=1:numHiddenStates
              tempdatalik[h] = get_data_lik_cache(datalikcache, obsnodes, h, seqpair.seq1, seqpair.seq2, preva, prevb, t)
              maxdatalik = max(maxdatalik,tempdatalik[h])
            end
          elseif alignnode == XINSERT
            for h=1:numHiddenStates
              tempdatalik[h] = get_data_lik_cache_seq1(datalikcache, obsnodes, h, seqpair.seq1, preva)
              maxdatalik = max(maxdatalik,tempdatalik[h])
            end
          elseif  alignnode == YINSERT
            for h=1:numHiddenStates
              tempdatalik[h] = get_data_lik_cache_seq2(datalikcache, obsnodes, h, seqpair.seq2, prevb)
              maxdatalik = max(maxdatalik,tempdatalik[h])
            end
          end

          if a == n+1 && b == m+1
            for h=1:numHiddenStates
              tempprevcells[MATCH,h] = cache2[z,MATCH,h]+hmmparameters.logaligntransprobs[MATCH, END]
              maxsum = max(maxsum, tempprevcells[MATCH,h])
              tempprevcells[XINSERT,h] = cache2[z,XINSERT,h]+hmmparameters.logaligntransprobs[XINSERT, END]
              maxsum = max(maxsum, tempprevcells[XINSERT,h])
              tempprevcells[YINSERT,h] = cache2[z,YINSERT,h]+hmmparameters.logaligntransprobs[YINSERT, END]
              maxsum = max(maxsum, tempprevcells[YINSERT,h])
            end
            for h=1:numHiddenStates
              @fastmath tempprevcells[MATCH,h] = exp(tempprevcells[MATCH,h]-maxsum)
              @fastmath tempprevcells[XINSERT,h] = exp(tempprevcells[XINSERT,h]-maxsum)
              @fastmath tempprevcells[YINSERT,h] = exp(tempprevcells[YINSERT,h]-maxsum)
              @fastmath tempprevcells2[h] = tempprevcells[MATCH,h]+tempprevcells[XINSERT,h]+tempprevcells[YINSERT,h]
            end
          elseif preva == 1 && prevb == 1
            for h=1:numHiddenStates
              tempprevcells2[h] = cache2[z,MATCH,h]+hmmparameters.logaligntransprobs[START, alignnode]
              maxsum = max(maxsum, tempprevcells2[h])
            end
            for h=1:numHiddenStates
              tempprevcells2[h] = exp(tempprevcells2[h]-maxsum)
            end
          else
            for h=1:numHiddenStates
              tempprevcells[MATCH,h] = cache2[z,MATCH,h]+hmmparameters.logaligntransprobs[MATCH, alignnode]
              maxsum = max(maxsum, tempprevcells[MATCH,h])
              tempprevcells[XINSERT,h] = cache2[z,XINSERT,h]+hmmparameters.logaligntransprobs[XINSERT, alignnode]
              maxsum = max(maxsum, tempprevcells[XINSERT,h])
              tempprevcells[YINSERT,h] = cache2[z,YINSERT,h]+hmmparameters.logaligntransprobs[YINSERT, alignnode]
              maxsum = max(maxsum, tempprevcells[YINSERT,h])
            end
            for h=1:numHiddenStates
              @fastmath tempprevcells[MATCH,h] = exp(tempprevcells[MATCH,h]-maxsum)
              @fastmath tempprevcells[XINSERT,h] = exp(tempprevcells[XINSERT,h]-maxsum)
              @fastmath tempprevcells[YINSERT,h] = exp(tempprevcells[YINSERT,h]-maxsum)
              @fastmath tempprevcells2[h] = tempprevcells[MATCH,h]+tempprevcells[XINSERT,h]+tempprevcells[YINSERT,h]
            end
          end

          for h=1:numHiddenStates
            expsum = 0.0
            @simd for prevh=1:numHiddenStates
              @fastmath expsum += tempprevcells2[prevh]*hmmparameters.hmmtransprobs[prevh,h]
            end

            if expsum > 0.0
              if cache2[z+1,alignnode,h] == -Inf
                @fastmath cache2[z+1,alignnode,h] = maxsum  + tempdatalik[h] + log(expsum)
              else
                @fastmath cache2[z+1,alignnode,h] = logsumexp(cache2[z+1,alignnode,h],maxsum  + tempdatalik[h] + log(expsum))
              end
            end
          end
        end
      end


      ll = -Inf
      for h=1:numHiddenStates
        ll = logsumexp(ll, cache2[len+1,h])
      end
      return ll
    else
      for h=1:numHiddenStates
        putvalue(cache,1,1,MATCH,h,hmmparameters.loghmminitprobs[h])
        putvalue(cache,1,1,XINSERT,h,hmmparameters.loghmminitprobs[h])
        putvalue(cache,1,1,YINSERT,h,hmmparameters.loghmminitprobs[h])
        #cache[1,1,MATCH,h] = hmmparameters.loghmminitprobs[h]
        #cache[1,1,XINSERT,h] = hmmparameters.loghmminitprobs[h]
        #cache[1,1,YINSERT,h] = hmmparameters.loghmminitprobs[h]
      end
      if partialAlignment == 1 || partialAlignment == 2
        a = 1
        b = 1
        alignnode = MATCH
        len = length(align1)
        fixAlignment = true
        for z=1:len
          if align1[z] != 0 && align2[z] != 0 # MATCH
            a += 1
            b += 1
            alignnode = MATCH
            if partialAlignment == 1 && (a < startIndex || a > endIndex)
              putvalue(cache,a,b,MATCH,states[z],0.0)
            end
            if partialAlignment == 2 && (b < startIndex || b > endIndex)
              putvalue(cache,a,b,MATCH,states[z],0.0)
            end
          elseif align2[z] == 0 # XINSERT
            a += 1
            alignnode = XINSERT
            if partialAlignment == 1 && (a < startIndex || a > endIndex)
              putvalue(cache,a,b,XINSERT,states[z],0.0)
            end
            if partialAlignment == 2 && (b < startIndex || b > endIndex)
              putvalue(cache,a,b,XINSERT,states[z],0.0)
            end
          elseif align1[z] == 0 # YINSERT
            b += 1
            alignnode = YINSERT
            if partialAlignment == 1 && (a < startIndex || a > endIndex)
              putvalue(cache,a,b,YINSERT,states[z],0.0)
            end
            if partialAlignment == 2 && (b < startIndex || b > endIndex)
              putvalue(cache,a,b,YINSERT,states[z],0.0)
            end
          end
        end
      end

      starta = 1
      enda = n+1
      startbx = 1
      endbx = m+1
      if partialAlignment == 1
        starta = startIndex
        enda = endIndex+1
      elseif partialAlignment == 2
        startbx = startIndex
        endbx = endIndex+1
      end


      for a=starta:enda
        startb = max(startbx,a-cornercut)
        endb = min(endbx,a+cornercut)
        for b=startb:endb
          for alignnode=1:3
            preva = a
            prevb = b
            if alignnode == MATCH
              preva -= 1
              prevb -= 1
            elseif alignnode == XINSERT
              preva -= 1
            elseif alignnode == YINSERT
              prevb -= 1
            end

            if preva > 0 && prevb > 0
              maxdatalik = -Inf
              maxsum = -Inf
              if alignnode == MATCH
                for h=1:numHiddenStates
                  tempdatalik[h] = get_data_lik_cache(datalikcache, obsnodes, h, seqpair.seq1, seqpair.seq2, preva, prevb, t)
                  maxdatalik = max(maxdatalik,tempdatalik[h])
                end
              elseif alignnode == XINSERT
                for h=1:numHiddenStates
                  tempdatalik[h] = get_data_lik_cache_seq1(datalikcache, obsnodes, h, seqpair.seq1, preva)
                  maxdatalik = max(maxdatalik,tempdatalik[h])
                end
              elseif  alignnode == YINSERT
                for h=1:numHiddenStates
                  tempdatalik[h] = get_data_lik_cache_seq2(datalikcache, obsnodes, h, seqpair.seq2, prevb)
                  maxdatalik = max(maxdatalik,tempdatalik[h])
                end
              end

              if a == n+1 && b == m+1
                for h=1:numHiddenStates
                  tempprevcells[MATCH,h] = getvalue(cache,preva,prevb,MATCH,h)+hmmparameters.logaligntransprobs[MATCH, END]
                  maxsum = max(maxsum, tempprevcells[MATCH,h])
                  tempprevcells[XINSERT,h] = getvalue(cache,preva,prevb,XINSERT,h)+hmmparameters.logaligntransprobs[XINSERT, END]
                  maxsum = max(maxsum, tempprevcells[XINSERT,h])
                  tempprevcells[YINSERT,h] = getvalue(cache,preva,prevb,YINSERT,h)+hmmparameters.logaligntransprobs[YINSERT, END]
                  maxsum = max(maxsum, tempprevcells[YINSERT,h])
                end
                for h=1:numHiddenStates
                  @fastmath tempprevcells[MATCH,h] = exp(tempprevcells[MATCH,h]-maxsum)
                  @fastmath tempprevcells[XINSERT,h] = exp(tempprevcells[XINSERT,h]-maxsum)
                  @fastmath tempprevcells[YINSERT,h] = exp(tempprevcells[YINSERT,h]-maxsum)
                  @fastmath tempprevcells2[h] = tempprevcells[MATCH,h]+tempprevcells[XINSERT,h]+tempprevcells[YINSERT,h]
                end
              elseif preva == 1 && prevb == 1
                for h=1:numHiddenStates
                  tempprevcells2[h] = getvalue(cache,preva,prevb,MATCH,h)+hmmparameters.logaligntransprobs[START, alignnode]
                  maxsum = max(maxsum, tempprevcells2[h])
                end
                for h=1:numHiddenStates
                  tempprevcells2[h] = exp(tempprevcells2[h]-maxsum)
                end
              else
                for h=1:numHiddenStates

                  tempprevcells[MATCH,h] = getvalue(cache,preva,prevb,MATCH,h)+hmmparameters.logaligntransprobs[MATCH, alignnode]
                  maxsum = max(maxsum, tempprevcells[MATCH,h])
                  tempprevcells[XINSERT,h] = getvalue(cache,preva,prevb,XINSERT,h)+hmmparameters.logaligntransprobs[XINSERT, alignnode]
                  maxsum = max(maxsum, tempprevcells[XINSERT,h])
                  tempprevcells[YINSERT,h] = getvalue(cache,preva,prevb,YINSERT,h)+hmmparameters.logaligntransprobs[YINSERT, alignnode]
                  maxsum = max(maxsum, tempprevcells[YINSERT,h])
                end
                for h=1:numHiddenStates
                  @fastmath tempprevcells[MATCH,h] = exp(tempprevcells[MATCH,h]-maxsum)
                  @fastmath tempprevcells[XINSERT,h] = exp(tempprevcells[XINSERT,h]-maxsum)
                  @fastmath tempprevcells[YINSERT,h] = exp(tempprevcells[YINSERT,h]-maxsum)
                  @fastmath tempprevcells2[h] = tempprevcells[MATCH,h]+tempprevcells[XINSERT,h]+tempprevcells[YINSERT,h]
                end
              end

              for h=1:numHiddenStates
                expsum = 0.0
                @simd for prevh=1:numHiddenStates
                  @fastmath expsum += tempprevcells2[prevh]*hmmparameters.hmmtransprobs[prevh,h]
                end

                if expsum > 0.0
                  if getvalue(cache,a,b,alignnode,h) == -Inf
                    @fastmath putvalue(cache,a,b,alignnode,h, maxsum  + tempdatalik[h] + log(expsum))
                  else
                    @fastmath putvalue(cache,a,b,alignnode,h, logsumexp(getvalue(cache,a,b,alignnode,h),maxsum  + tempdatalik[h] + log(expsum)))
                  end
                end
              end
            end

          end
        end
      end
    end
    i = n+1
    j = m+1
    ll = -Inf
    for h=1:numHiddenStates
      for alignnode=1:3
        ll = logsumexp(ll, getvalue(cache,i,j,alignnode,h))
      end
    end
    return ll
  end

  export tkf92sample
  function tkf92sample(rng::AbstractRNG, seqpair::SequencePair, cache::HMMCache, params::PairParameters, modelparams::ModelParameters, B::Float64)
    prior = modelparams.prior
    obsnodes = modelparams.obsnodes
    hmminitprobs = modelparams.hmminitprobs
    hmmtransprobs = modelparams.hmmtransprobs
    numHiddenStates = length(hmminitprobs)
    aligntransprobs = get_alignment_transition_probabilities(params.lambda,params.mu,params.r,params.t)
    hmmparameters = HMMParameters(aligntransprobs, hmminitprobs, hmmtransprobs)

    n = seqpair.seq1.length
    m = seqpair.seq2.length
    numHiddenStates = hmmparameters.numHiddenStates

    i = n+1
    j = m+1
    choice = zeros(Float64,3*numHiddenStates)
    for h=1:numHiddenStates
      for alignnode=1:3
        choice[(alignnode-1)*numHiddenStates+h] = getvalue(cache,i,j,alignnode,h)
      end
    end
    for c=1:length(choice)
      if choice[c] != -Inf
        choice[c] *= B
      end
    end
    r = GumbelSample(rng,choice)
    alignnode = div(r-1,numHiddenStates) + 1
    h = ((r-1) % numHiddenStates)+1

    align1 = Int[]
    align2 = Int[]
    states = Int[]

    while true
      a = i
      b = j
      if alignnode == MATCH
        i -= 1
        j -= 1
      elseif alignnode == XINSERT
        i -= 1
      elseif alignnode == YINSERT
        j -= 1
      end
      #println(i,"\t",j, "\t", alignnode,"\t", h)
      if i == 0 || j == 0
        break
      end
      if alignnode == MATCH
        unshift!(align1, i)
        unshift!(align2, j)
      elseif alignnode == XINSERT
        unshift!(align1, i)
        unshift!(align2, 0)
      elseif alignnode == YINSERT
        unshift!(align1, 0)
        unshift!(align2, j)
      end
      unshift!(states, h)


      for prevh=1:numHiddenStates
        for prevalignnode=1:3
          if a == n+1 && b == m+1
            choice[(prevalignnode-1)*numHiddenStates+prevh] = getvalue(cache,i,j,prevalignnode,prevh) + hmmparameters.logaligntransprobs[prevalignnode, END] + hmmparameters.loghmmtransprobs[prevh,h]
          elseif i == 1 && j == 1
            choice[(prevalignnode-1)*numHiddenStates+prevh] = getvalue(cache,i,j,prevalignnode,prevh) + hmmparameters.logaligntransprobs[START, alignnode] + hmmparameters.loghmmtransprobs[prevh,h]
          else
            choice[(prevalignnode-1)*numHiddenStates+prevh] = getvalue(cache,i,j,prevalignnode,prevh) + hmmparameters.logaligntransprobs[prevalignnode, alignnode]+ hmmparameters.loghmmtransprobs[prevh,h]
          end
        end
      end
      for c=1:length(choice)
        if choice[c] != -Inf
          choice[c] *= B
        end
      end
      r = GumbelSample(rng,choice)
      alignnode = div(r-1,numHiddenStates) + 1
      h = ((r-1) % numHiddenStates)+1
    end

    return align1, align2, states
  end

   export tkf92sample2
  function tkf92sample2(rng::AbstractRNG, seqpair::SequencePair, cache::Array{Float64,3}, align1::Array{Int,1}, align2::Array{Int,1}, params::PairParameters, modelparams::ModelParameters, B::Float64)
    prior = modelparams.prior
    obsnodes = modelparams.obsnodes
    hmminitprobs = modelparams.hmminitprobs
    hmmtransprobs = modelparams.hmmtransprobs
    numHiddenStates = length(hmminitprobs)
    aligntransprobs = get_alignment_transition_probabilities(params.lambda,params.mu,params.r,params.t)
    hmmparameters = HMMParameters(aligntransprobs, hmminitprobs, hmmtransprobs)

    n = seqpair.seq1.length
    m = seqpair.seq2.length
    numHiddenStates = hmmparameters.numHiddenStates

    i = n+1
    j = m+1
    len = length(align1)+1
    choice = zeros(Float64,3*numHiddenStates)
    for h=1:numHiddenStates
      for alignnode=1:3
        choice[(alignnode-1)*numHiddenStates+h] = cache[len,alignnode,h]
      end
    end
    for c=1:length(choice)
      if choice[c] != -Inf
        choice[c] *= B
      end
    end

    r = GumbelSample(rng,choice)
    alignnode = div(r-1,numHiddenStates) + 1
    h = ((r-1) % numHiddenStates)+1

    states = Int[]

    z = len
    while true
      a = i
      b = j
      if alignnode == MATCH
        i -= 1
        j -= 1
      elseif alignnode == XINSERT
        i -= 1
      elseif alignnode == YINSERT
        j -= 1
      end
      z -= 1
      #println(i,"\t",j, "\t", alignnode,"\t", h)
      if i == 0 || j == 0
        break
      end
      unshift!(states, h)


      for prevh=1:numHiddenStates
        for prevalignnode=1:3
          if a == n+1 && b == m+1
            choice[(prevalignnode-1)*numHiddenStates+prevh] = cache[z,prevalignnode,prevh] + hmmparameters.logaligntransprobs[prevalignnode, END] + hmmparameters.loghmmtransprobs[prevh,h]
          elseif i == 1 && j == 1
            choice[(prevalignnode-1)*numHiddenStates+prevh] = cache[z,prevalignnode,prevh] + hmmparameters.logaligntransprobs[START, alignnode] + hmmparameters.loghmmtransprobs[prevh,h]
          else
            choice[(prevalignnode-1)*numHiddenStates+prevh] = cache[z,prevalignnode,prevh] + hmmparameters.logaligntransprobs[prevalignnode, alignnode]+ hmmparameters.loghmmtransprobs[prevh,h]
          end
        end
      end
      for c=1:length(choice)
        if choice[c] != -Inf
          choice[c] *= B
        end
      end
      r = GumbelSample(rng,choice)
      alignnode = div(r-1,numHiddenStates) + 1
      h = ((r-1) % numHiddenStates)+1
    end

    return copy(align1), copy(align2), states
  end

  export tkf92
  function tkf92(nsamples::Int, rng::AbstractRNG, cache::HMMCache, datalikcache::DatalikCache, seqpair::SequencePair, params::PairParameters, modelparams::ModelParameters, B::Float64, cornercut::Int=1000000, fixAlignment::Bool=false, align1::Array{Int,1}=zeros(Int,1), align2::Array{Int,1}=zeros(Int,1), fixStates::Bool=false, states::Array{Int,1}=zeros(Int,1), partialAlignment::Int=0, startIndex::Int=1, endIndex::Int=1, includeAlignmentProbability::Bool=true)
    C = B
    D = 1.0
    if B == -1.0
      C  = 1.0
      D = 0.0
    end

    prior = modelparams.prior
    obsnodes = modelparams.obsnodes
    hmminitprobs = modelparams.hmminitprobs
    hmmtransprobs = modelparams.hmmtransprobs
    numHiddenStates = length(hmminitprobs)
    aligntransprobs = get_alignment_transition_probabilities(params.lambda,params.mu,params.r,params.t)
    if !includeAlignmentProbability
      fill!(aligntransprobs,1.0)
    end
    #println(aligntransprobs)
    hmmparameters = HMMParameters(aligntransprobs, hmminitprobs, hmmtransprobs)
    samples = SequencePairSample[]
    if fixStates
      ll = C*tkf92forward(seqpair, params.t, datalikcache, cache, ones(Float64,1,1,1), obsnodes, hmmparameters, cornercut, fixAlignment, align1, align2, fixStates, states, partialAlignment, startIndex, endIndex)
      ll += D*logprior(modelparams.prior, params)
      sample = SequencePairSample(seqpair, params)
      sample.align1, sample.align2, sample.states = copy(align1), copy(align2), copy(states)
      push!(samples, sample)
      return ll, samples, cache
    else
      ll = 0.0
      if fixAlignment
        cache2 = ones(length(align1)+1,3, modelparams.numHiddenStates)*-Inf
        ll = C*tkf92forward(seqpair, params.t, datalikcache, cache, cache2, obsnodes, hmmparameters, cornercut, fixAlignment, align1, align2, fixStates, states, partialAlignment, startIndex, endIndex)
        ll += D*logprior(modelparams.prior, params)
        for i=1:nsamples
          sample = SequencePairSample(seqpair, params)
          sample.align1, sample.align2, sample.states = AlignmentHMM.tkf92sample2(rng, seqpair, cache2, align1, align2, params, modelparams, B)
          push!(samples, sample)
        end
        return ll, samples, cache2
      else
        resetcache(cache)
        ll = C*tkf92forward(seqpair, params.t, datalikcache, cache, ones(Float64,1,1,1), obsnodes, hmmparameters, cornercut, fixAlignment, align1, align2, fixStates, states, partialAlignment, startIndex, endIndex)
        ll += D*logprior(modelparams.prior, params)
        for i=1:nsamples
          sample = SequencePairSample(seqpair, params)
          sample.align1, sample.align2, sample.states = AlignmentHMM.tkf92sample(rng, seqpair, cache, params, modelparams, B)
          push!(samples, sample)
        end
        return ll, samples, cache
      end
    end
  end
end
#=

using Formatting
using Distributions
using DataStructures
using UtilsModule
using NodesModule

srand(98418108751401)
rng = MersenneTwister(242402531025555)
benchmarksfilepath = "data/holdout_data_diverse.txt"
benchmarksname = basename(benchmarksfilepath)

pairs = shuffle!(rng, load_sequences_and_alignments(benchmarksfilepath))
modelfile = "models/pairhmm32_switching_n200_fixalignment=false.jls"
#modelfile = "models/pairhmm4_switching_n5_fixalignment=false.jls"
ser = open(modelfile,"r")
modelparams = deserialize(ser)
close(ser)

inputsamples = shuffle!(rng, load_sequences_and_alignments("data/data_diverse.txt"))
pairs = SequencePair[sample.seqpair for sample in inputsamples]
params = PairParameters()

cornercut = 30
seqpair = pairs[2]
datalikcache = AlignmentHMM.DatalikCache(seqpair.seq1.length, seqpair.seq2.length, modelparams.numHiddenStates)
ll, samples, cache = AlignmentHMM.tkf92(1,rng,datalikcache, seqpair,params, modelparams, cornercut)
println(ll)
ll, samples, cache = AlignmentHMM.tkf92(1,rng,datalikcache, seqpair,params, modelparams, cornercut)
println(ll)

sample = samples[1]
println(string(getaminoacidalignment(seqpair.seq1, sample.align1)))
println(string(getaminoacidalignment(seqpair.seq2, sample.align2)))
println(string(getstatestring(sample.states)))

println(seqpair.seq1.length,"\t", seqpair.seq2.length)
ll, samples, cache = AlignmentHMM.tkf92(1,rng, datalikcache, seqpair,params, modelparams, cornercut, true, sample.align1, sample.align2, true, sample.states)
println(ll)

ll, samples, cache = AlignmentHMM.tkf92(1,rng, datalikcache, seqpair,params, modelparams, cornercut, false, sample.align1, sample.align2, false, sample.states, 1, 1,143)
sample = samples[1]
println(string(getaminoacidalignment(seqpair.seq1, sample.align1)))
println(string(getaminoacidalignment(seqpair.seq2, sample.align2)))
println(string(getstatestring(sample.states)))
ll, samples, cache = AlignmentHMM.tkf92(1,rng, datalikcache, seqpair,params, modelparams, cornercut, false, sample.align1, sample.align2, false, sample.states, 1, 1,143)
sample = samples[1]
println(string(getaminoacidalignment(seqpair.seq1, sample.align1)))
println(string(getaminoacidalignment(seqpair.seq2, sample.align2)))
println(string(getstatestring(sample.states)))
ll, samples, cache = AlignmentHMM.tkf92(1,rng, datalikcache, seqpair,params, modelparams, cornercut, false, sample.align1, sample.align2, false, sample.states, 1, 1,143)
sample = samples[1]
println(string(getaminoacidalignment(seqpair.seq1, sample.align1)))
println(string(getaminoacidalignment(seqpair.seq2, sample.align2)))
println(string(getstatestring(sample.states)))
ll, samples, cache = AlignmentHMM.tkf92(1,rng, datalikcache, seqpair,params, modelparams, cornercut, false, sample.align1, sample.align2, false, sample.states, 1, 1,143)
sample = samples[1]
println(string(getaminoacidalignment(seqpair.seq1, sample.align1)))
println(string(getaminoacidalignment(seqpair.seq2, sample.align2)))
println(string(getstatestring(sample.states)))
ll, samples, cache = AlignmentHMM.tkf92(1,rng, datalikcache, seqpair,params, modelparams, cornercut, false, sample.align1, sample.align2, false, sample.states, 1, 1,143)
sample = samples[1]
println(string(getaminoacidalignment(seqpair.seq1, sample.align1)))
println(string(getaminoacidalignment(seqpair.seq2, sample.align2)))
println(string(getstatestring(sample.states)))
ll, samples, cache = AlignmentHMM.tkf92(1,rng, datalikcache, seqpair,params, modelparams, cornercut, false, sample.align1, sample.align2, false, sample.states, 1, 1,143)
sample = samples[1]
println(string(getaminoacidalignment(seqpair.seq1, sample.align1)))
println(string(getaminoacidalignment(seqpair.seq2, sample.align2)))
println(string(getstatestring(sample.states)))
ll, samples, cache = AlignmentHMM.tkf92(1,rng, datalikcache, seqpair,params, modelparams, cornercut, false, sample.align1, sample.align2, false, sample.states, 1, 1,143)
sample = samples[1]
println(string(getaminoacidalignment(seqpair.seq1, sample.align1)))
println(string(getaminoacidalignment(seqpair.seq2, sample.align2)))
println(string(getstatestring(sample.states)))=#
