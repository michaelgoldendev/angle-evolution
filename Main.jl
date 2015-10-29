#=
using ProfileView

#addprocs(CPU_CORES)
@everywhere include("PairHMM.jl")
#train()
@profile train()
ProfileView.svgwrite("profile_results.svg")
profilewriter = open("profile.log", "w")
Profile.print(profilewriter)
=#
#push!(LOAD_PATH, pwd())
#println(pwd())

addprocs(CPU_CORES)

@everywhere include("UtilsModule.jl")
@everywhere include("NodesModule.jl")
@everywhere include("PairHMM.jl")
train()

#include("PairHMM.jl")
#train()

#include("PairHMM.jl")
#mlalign()

#=
include("UtilsModule.jl")
include("NodesModule.jl")
include("PairHMM.jl")

test()
=#
#train()
