#mlalign()
#test()

addprocs(CPU_CORES)
#include("PairHMM.jl")
@everywhere include("PairHMM.jl")
train()
