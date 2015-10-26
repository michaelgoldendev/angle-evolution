include("CTMC.jl")

export SecondaryStructureNode
type SecondaryStructureNode
  ctmc::CTMC

  function SecondaryStructureNode()
    eqfreqs = ones(Float64,3)/3.0
    a = 0.5
    b = 1.0
    c = 1.5
    t = 1.0
    S = zeros(Float64, 3, 3)
    S[1,1] = -a -b
    S[1,2] = a
    S[1,3] = b
    S[2,1] = a
    S[2,2] = -a -c
    S[2,3] = c
    S[3,1] = b
    S[3,2] = c
    S[3,3] = -b -c
    ctmc = CTMC(eqfreqs, S, t)
    new(ctmc)
  end

  function SecondaryStructureNode(eqfreqs::Array{Float64,1}, a::Float64, b::Float64, c::Float64, t::Float64)
    S = zeros(Float64, 3, 3)
    S[1,1] = -a -b
    S[1,2] = a
    S[1,3] = b
    S[2,1] = a
    S[2,2] = -a -c
    S[2,3] = c
    S[3,1] = b
    S[3,2] = c
    S[3,3] = -b -c
    ctmc = CTMC(eqfreqs, S, t)
    new(ctmc)
  end
end

export set_parameters
function set_parameters(ss::SecondaryStructureNode, eqfreqs::Array{Float64,1}, a::Float64, b::Float64, c::Float64, t::Float64)
    S = zeros(Float64, 3, 3)
    S[1,1] = -a -b
    S[1,2] = a
    S[1,3] = b
    S[2,1] = a
    S[2,2] = -a -c
    S[2,3] = c
    S[3,1] = b
    S[3,2] = c
    S[3,3] = -b -c
    set_parameters(ss.ctmc, eqfreqs,  S, t)
end

function set_parameters(ss::SecondaryStructureNode, eqfreqs::Array{Float64,1}, t::Float64)
    set_parameters(ss.ctmc, eqfreqs,  ss.ctmc.S, t)
end

function set_parameters(ss::SecondaryStructureNode, a::Float64, b::Float64, c::Float64, t::Float64)
    S = zeros(Float64, 3, 3)
    S[1,1] = -a -b
    S[1,2] = a
    S[1,3] = b
    S[2,1] = a
    S[2,2] = -a -c
    S[2,3] = c
    S[3,1] = b
    S[3,2] = c
    S[3,3] = -b -c
    set_parameters(ss.ctmc, ss.ctmc.eqfreqs,  S, t)
end
