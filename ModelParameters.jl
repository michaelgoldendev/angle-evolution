include("ObservationNode.jl")

type ModelParameters
  prior::PriorDistribution
  obsnodes::Array{ObservationNode,1}
  hmminitprobs::Array{Float64,1}
  hmmtransprobs::Array{Float64,2}
  numHiddenStates::Int

  function ModelParameters(prior::PriorDistribution, obsnodes::Array{ObservationNode,1}, hmminitprobs::Array{Float64,1}, hmmtransprobs::Array{Float64,2})
      return new(prior, obsnodes, hmminitprobs, hmmtransprobs, length(hmminitprobs))
  end
end

function readmodel(modelfile)
    ser = open(modelfile,"r")
    modelparams::ModelParameters = deserialize(ser)
    close(ser)
    return modelparams
end

function write_hiddenstates(modelio::ModelParameters, filename)
  out = open(filename, "w")
  if modelio.obsnodes[1].useswitching
    for h=1:modelio.numHiddenStates
      write(out, string(get_parameters(modelio.obsnodes[h].switching.diffusion_r1)),"\n")
      write(out, string(modelio.obsnodes[h].switching.aapairnode_r1.eqfreqs),"\n")
      write(out, string(get_parameters(modelio.obsnodes[h].switching.diffusion_r2)),"\n")
      write(out, string(modelio.obsnodes[h].switching.aapairnode_r2.eqfreqs),"\n")
    end
  else
    for h=1:modelio.numHiddenStates
      write(out, string(get_parameters(modelio.obsnodes[h].diffusion)),"\n")
      write(out, string(modelio.obsnodes[h].aapairnode.eqfreqs),"\n")
    end
  end
  close(out)
end

function aic(ll::Float64, freeParameters::Int)
	return 2.0*freeParameters - 2.0*ll
end
