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
  counts /= Float64(nsamples)
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
