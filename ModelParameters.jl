include("ObservationNode.jl")

export ModelParameters
type ModelParameters
  prior::PriorDistribution
  obsnodes::Array{ObservationNode,1}
  hmminitprobs::Array{Float64,1}
  hmmtransprobs::Array{Float64,2}
  numHiddenStates::Int
  useerrordistribution::Bool
  samples::Array{SequencePairSample,1}

  function ModelParameters(prior::PriorDistribution, obsnodes::Array{ObservationNode,1}, hmminitprobs::Array{Float64,1}, hmmtransprobs::Array{Float64,2},useerrordistribution::Bool, samples::Array{SequencePairSample,1})
      return new(prior, obsnodes, hmminitprobs, hmmtransprobs, length(hmminitprobs),useerrordistribution, samples)
  end
end

export readmodel
function readmodel(modelfile)
    ser = open(modelfile,"r")
    modelparams::ModelParameters = deserialize(ser)
    close(ser)
    return modelparams
end

export write_hiddenstates
function write_hiddenstates(modelio::ModelParameters, filename)
  out = open(filename, "w")

  if modelio.obsnodes[1].useswitching
    for h=1:modelio.numHiddenStates
      write(out, string(get_parameters(modelio.obsnodes[h].switching.diffusion_r1)),"\n")
      write(out, string(modelio.obsnodes[h].switching.aapairnode_r1.eqfreqs),"\n")
      write(out, string(modelio.obsnodes[h].switching.ss_r1.ctmc.eqfreqs),"\n")
      write(out, string(get_parameters(modelio.obsnodes[h].switching.diffusion_r2)),"\n")
      write(out, string(modelio.obsnodes[h].switching.aapairnode_r2.eqfreqs),"\n")
      write(out, string(modelio.obsnodes[h].switching.ss_r2.ctmc.eqfreqs),"\n")
    end
  else
    for h=1:modelio.numHiddenStates
      write(out, string(get_parameters(modelio.obsnodes[h].diffusion)),"\n")
      write(out, string(modelio.obsnodes[h].aapairnode.eqfreqs),"\n")
      write(out, string(modelio.obsnodes[h].ss.ctmc.eqfreqs),"\n")
    end
  end
  close(out)
end

using JSON

export export_json
function export_json(modelio::ModelParameters, filename)
  jsondict = Dict()
  jsondict["numHiddenStates"] = modelio.numHiddenStates
  jsondict["switching"] = modelio.obsnodes[1].useswitching
  if modelio.obsnodes[1].useswitching
    for h=1:modelio.numHiddenStates
      jsondict[string("diffusion_r1.",h)] = get_parameters(modelio.obsnodes[h].switching.diffusion_r1)
      jsondict[string("aapairnode_r1.eqfreqs.",h)] = modelio.obsnodes[h].switching.aapairnode_r1.eqfreqs
      jsondict[string("ss_r1.eqfreqs.",h)] = modelio.obsnodes[h].switching.ss_r1.ctmc.eqfreqs
      jsondict[string("diffusion_r2.",h)] = get_parameters(modelio.obsnodes[h].switching.diffusion_r2)
      jsondict[string("aapairnode_r2.eqfreqs.",h)] = modelio.obsnodes[h].switching.aapairnode_r2.eqfreqs
      jsondict[string("ss_r2.eqfreqs.",h)] = modelio.obsnodes[h].switching.ss_r2.ctmc.eqfreqs

      obsnode = modelio.obsnodes[h]
      jsondict[string("switching.alpha.",h)] =  obsnode.switching.alpha
      jsondict[string("switching.pi_r1.",h)] =  obsnode.switching.pi_r1
    end
  else
    for h=1:modelio.numHiddenStates
      jsondict[string("diffusion_r1.",h)] = get_parameters(modelio.obsnodes[h].diffusion)
      jsondict[string("aapairnode_r1.eqfreqs.",h)] = modelio.obsnodes[h].aapairnode.eqfreqs
      jsondict[string("ss_r1.eqfreqs.",h)] = modelio.obsnodes[h].ss.ctmc.eqfreqs
    end
  end

  jsonout = open(filename,"w")
  JSON.print(jsonout, jsondict)
  close(jsonout)
end

export print_hiddenstates
function print_hiddenstates(modelio::ModelParameters, filename)
  #out = open(filename, "w")
  if modelio.obsnodes[1].useswitching
    for h=1:modelio.numHiddenStates
      println(string(modelio.obsnodes[h].switching.diffusion_r1.alpha_phi,"\t",modelio.obsnodes[h].switching.diffusion_r1.sigma_phi,"\t", modelio.obsnodes[h].switching.diffusion_r1.mu_phi))
      println(string(modelio.obsnodes[h].switching.diffusion_r1.alpha_psi,"\t",modelio.obsnodes[h].switching.diffusion_r1.sigma_psi,"\t", modelio.obsnodes[h].switching.diffusion_r1.mu_psi))
      println(string(modelio.obsnodes[h].switching.diffusion_r2.alpha_phi,"\t",modelio.obsnodes[h].switching.diffusion_r2.sigma_phi,"\t", modelio.obsnodes[h].switching.diffusion_r2.mu_phi))
      println(string(modelio.obsnodes[h].switching.diffusion_r2.alpha_psi,"\t",modelio.obsnodes[h].switching.diffusion_r2.sigma_psi,"\t", modelio.obsnodes[h].switching.diffusion_r2.mu_psi))
    end
  else
    for h=1:modelio.numHiddenStates
      write(out, string(get_parameters(modelio.obsnodes[h].diffusion)),"\n")
      write(out, string(modelio.obsnodes[h].aapairnode.eqfreqs),"\n")
      write(out, string(modelio.obsnodes[h].ss.ctmc.eqfreqs),"\n")
    end
  end
  #close(out)
end

export aic
function aic(ll::Float64, freeParameters::Int)
	return 2.0*freeParameters - 2.0*ll
end
