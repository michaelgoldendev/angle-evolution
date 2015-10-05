using Distributions

include("VonMisesDensity.jl")

aminoacids = "ACDEFGHIKLMNPQRSTVWY"
MISSING_ANGLE = -1000.0
ANGLE_ERROR_KAPPA = 500.0


type Sequence
  length::Int
  seq::Array{Int, 1}
  phi::Array{Float64, 1}
  psi::Array{Float64, 1}
  phi_error::Array{Float64, 1}
  psi_error::Array{Float64, 1}
  angle_error_kappa::Float64
  error_distribution::VonMisesDensity

  function Sequence(length::Int)
    new(length, zeros(Int,length), ones(Float64,length)*MISSING_ANGLE, ones(Float64,length)*MISSING_ANGLE, ones(Float64,length)*MISSING_ANGLE, ones(Float64,length)*MISSING_ANGLE, ANGLE_ERROR_KAPPA, VonMisesDensity(0.0, ANGLE_ERROR_KAPPA))
  end

  function Sequence(seq::ASCIIString)
    len = length(seq)
    s = zeros(Int,len)
    for i=1:len
      s[i] = search(aminoacids, seq[i])
    end

    return new(len, s, ones(Float64,len)*MISSING_ANGLE, ones(Float64,len)*MISSING_ANGLE, ones(Float64,len)*MISSING_ANGLE, ones(Float64,len)*MISSING_ANGLE, ANGLE_ERROR_KAPPA, VonMisesDensity(0.0, ANGLE_ERROR_KAPPA))
  end

  function Sequence(seq::ASCIIString, phi::Array{Float64,1}, psi::Array{Float64,1})
    len = length(seq)
    s = zeros(Int,len)
    for i=1:len
      s[i] = search(aminoacids, seq[i])
    end

    return new(len, s, phi, psi, copy(phi), copy(psi), ANGLE_ERROR_KAPPA, VonMisesDensity(0.0, ANGLE_ERROR_KAPPA))
  end

  function Sequence(sequence::Sequence)
    return new(sequence.length, copy(sequence.seq), copy(sequence.phi), copy(sequence.psi), copy(sequence.phi_error), copy(sequence.psi_error), sequence.angle_error_kappa, VonMisesDensity(0.0, sequence.angle_error_kappa))
  end
end

type SequencePair
  id::Int
  seq1::Sequence
  seq2::Sequence
  t::Float64

  function SequencePair(id::Int, seq1::Sequence, seq2::Sequence)
    return new(id, seq1, seq2, 1.0)
  end

  function SequencePair(pair::SequencePair)
    return new(pair.id,pair.seq1, pair.seq2, pair.t)
  end
end


type PairParameters
  lambda::Float64
  mu::Float64
  ratio::Float64
  r::Float64
  t::Float64

  function PairParameters()
    return new(0.1,0.2,0.5,0.5,0.1)
  end

  function PairParameters(x::PairParameters)
    return new(x.lambda, x.mu, x.ratio, x.r, x.t)
  end

  function PairParameters(x::Array{Float64,1})
    lambda = x[1]
    ratio = x[2]
    mu = lambda/ratio
    r = x[3]
    t = x[4]
    return new(lambda,mu, ratio, r, t)
  end
end

type SequencePairSample
  seqpair::SequencePair
  params::PairParameters
  align1::Array{Int,1}
  align2::Array{Int,1}
  states::Array{Int,1}

  function SequencePairSample(seqpair::SequencePair, params::PairParameters)
    align1 = Int[]
    align2 = Int[]
    states = Int[]
    return new(seqpair, PairParameters(params), align1, align2, states)
  end

  function SequencePairSample(sample::SequencePairSample)
    return new(SequencePair(sample.seqpair), PairParameters(sample.params), copy(sample.align1), copy(sample.align2), copy(sample.states))
  end
end

function getconfigurations(align1::Array{Int,1},align2::Array{Int,1})
  conf1 = Int[]
  conf2 = Int[]
  for i=1:length(align1)
    if align1[i] > 0
      push!(conf1, align2[i])
    end
    if align2[i] > 0
      push!(conf2, align1[i])
    end
  end
  return conf1, conf2
end

function getalignmentpath(n::Int, m::Int, align1::Array{Int,1},align2::Array{Int,1}, states::Array{Int,1})
  matrix::SparseMatrixCSC{Int64,Int64} = spzeros(Int, n+1, m+1)
  hindex = length(states)
  matrix[n+1,m+1] = states[hindex]
  if states[hindex] == 0
    matrix[n+1,m+1] = 1000000
  end
  i = n+1
  j = m+1
  for (a,b) in zip(reverse(align1), reverse(align2))
    if a > 0 && b > 0
      i -= 1
      j -= 1
    elseif a > 0
      i -= 1
    elseif b > 0
      j -= 1
    end

    hindex -= 1
    if i > 0 && j > 0
      if hindex > 0
        matrix[i,j] = states[hindex]
      else
        matrix[i,j] = 1000000
      end
    end
  end
  return matrix
end

function getsequencestates(align1::Array{Int,1}, align2::Array{Int,1}, states::Array{Int,1})
  states1 = Int[]
  states2 = Int[]
  for i=1:length(align1)
    if align1[i] > 0
      push!(states1, states[i])
    end
    if align2[i] > 0
      push!(states2, states[i])
    end
  end
  return states1, states2
end

function getalignment(seq1::Sequence, align1::Array{Int,1})
  s1 = ""
  index = 1
  for i=1:length(align1)
    if align1[i] > 0
      s1 = string(s1, aminoacids[seq1.seq[index]])
      index += 1
    else
       s1 = string(s1, "-")
    end
  end

  return s1
end

type PriorDistribution
    lambdaprior::Gamma
    muprior::Gamma
    rprior::Beta
    tprior::Gamma
    params::Array{Float64,1}

    function PriorDistribution()
      new(Gamma(), Gamma(), Beta(), Gamma(), zeros(Float64,8))
    end

    function PriorDistribution(params::Array{Float64, 1})
      new(Gamma(params[1], params[2]), Gamma(params[3], params[4]), Beta(params[5], params[6]), Gamma(params[7], params[8]), params)
    end
end

function logprior(prior::PriorDistribution, pairparams::PairParameters)
      ll = 0.0
      ll += logpdf(prior.lambdaprior, pairparams.lambda)
      ll += logpdf(prior.muprior, pairparams.mu)
      ll += logpdf(prior.rprior, pairparams.r)
      ll += logpdf(prior.tprior, pairparams.t)
      return ll
end

function logprior(prior::PriorDistribution, samples::Array{SequencePairSample,1})
  ll = 0.0
  for sample in samples
    ll += logprior(prior, sample.params)
  end
  return ll
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
