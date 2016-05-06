#=
addprocs(2)

#addprocs(CPU_CORES)


@everywhere include("UtilsModule.jl")
@everywhere include("NodesModule.jl")
@everywhere include("Cornercut.jl")
@everywhere include("StatisticalAlignmentHMM.jl")
@everywhere include("PairHMM.jl")

train()=#

#include("PairHMM.jl")
#train()

#include("PairHMM.jl")
#mlalign()
#=
include("UtilsModule.jl")
include("NodesModule.jl")
include("Cornercut.jl")
include("StatisticalAlignmentHMM.jl")
include("PairHMM.jl")

#simulatestationary()
#analysejumps()
=#
#=
addprocs(2)
@everywhere include("AcceptanceLogger.jl")
@everywhere include("UtilsModule.jl")
@everywhere include("NodesModule.jl")
@everywhere include("Cornercut.jl")
@everywhere include("StatisticalAlignmentHMM.jl")
@everywhere include("PairHMM.jl")
=#

include("AcceptanceLogger.jl")
include("UtilsModule.jl")
include("NodesModule.jl")
include("Cornercut.jl")
include("StatisticalAlignmentHMM.jl")
include("PairHMM.jl")

computemarginallikelihoods()
