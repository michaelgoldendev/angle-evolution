module NodesModule
  using Distributions
  include("AAPairNode.jl")
  include("DiffusionNode.jl")
  include("SwitchingNode.jl")
  include("MCMCmoves.jl")
  include("SecondaryStructureNode.jl")
  include("Sequence.jl")
  include("ObservationNode.jl")
  include("ModelParameters.jl")
  include("ModelOptimization.jl")
end
