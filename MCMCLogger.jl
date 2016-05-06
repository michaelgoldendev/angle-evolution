using DataStructures
using Formatting

export MCMCLogger
type MCMCLogger
    dict::Dict{AbstractString, Array{Float64,1}}

    function MCMCLogger()
        return new(Dict())
    end
end

function logvalue(logger::MCMCLogger, key::AbstractString, v::Float64)
  if !haskey(logger.dict, key)
    logger.dict[key] = Float64[]
  end
  push!(logger.dict[key],v)
end

#=
logger = MCMCLogger()

logvalue(logger, "t", 0.5)
logvalue(logger, "t", 0.6)
logvalue(logger, "t", 0.55)
logvalue(logger, "t", 0.52)
println(logger.dict)

for key in keys(logger.dict)
  println("key:",key)
end=#