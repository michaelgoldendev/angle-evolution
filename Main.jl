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

addprocs(4)
@everywhere include("PairHMM.jl")
train()

#include("PairHMM.jl")
#train()

#include("PairHMM.jl")
#mlalign()


#include("PairHMM.jl")
#test()
#train()
