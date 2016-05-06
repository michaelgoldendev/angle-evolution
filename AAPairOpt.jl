include("ObservationNode.jl")
include("Sequence.jl")

type AAPairdOpt
  fixh::Int
  samples::Array{SequencePairSample,1}
  obsnodes::Array{ObservationNode, 1}
  hmminitprobs::Array{Float64,1}
  hmmtransprobs::Array{Float64,2}

  function AAPairOpt(fixh::Int, samples::Array{SequencePairSample-,1}, obsnodes::Array{ObservationNode, 1}, hmminitprobs::Array{Float64,1}, hmmtransprobs::Array{Float64})
    new(fixh, samples, obsnodes, hmminitprobs, hmmtransprobs)
  end


  function computell()
    ll = 0.0
    for sample in samples
      seqpair = sample.seqpair
      align1 = sample.align1
      align2 = sample.align2
      states = sample.states
      for a=1:length(align1)
        i = align1[a]
        j = align2[a]
        h = states[a]
        if h == fixh
          t = sample.params.t
          if i == 0
            ll += get_data_lik(obsnodes[h], seqpair.seq2,j, t)
          elseif j == 0
            ll += get_data_lik(obsnodes[h], seqpair.seq1,i, t)
          else
            ll += get_data_lik(obsnodes[h], seqpair.seq1, seqpair.seq2, i, j, t)
          end
        end
      end
    end

    return ll
  end
end