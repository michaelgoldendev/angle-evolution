using UtilsModule

export AAPairNode
type AAPairNode
  eqfreqs::Array{Float64, 1}
  logeqfreqs::Array{Float64, 1}
  S::Array{Float64,2}
  Q::Array{Float64,2}
  D::Array{Float64,1}
  V::Array{Float64,2}
  Vi::Array{Float64,2}
  t::Float64
  Pt::Array{Float64,2}
  logPt::Array{Float64,2}

  function AAPairNode()
    eqfreqs = ones(Float64,20)*0.05
    logeqfreqs = log(eqfreqs)
    S = ones(Float64, 20, 20)
    Q = ones(Float64, 20, 20)
    t = 1.0
    Pt = expm(Q*t)
    logPt = log(Pt)
    return new(eqfreqs, logeqfreqs, S, Q, zeros(Float64,1), zeros(Float64,1,1), zeros(Float64,1,1), t, Pt, logPt)
  end

  function AAPairNode(node::AAPairNode)
    new(copy(node.eqfreqs), copy(node.logeqfreqs), copy(node.S), copy(node.Q), copy(node.D), copy(node.V), copy(node.Vi), node.t, copy(node.Pt), copy(node.logPt))
  end
end

export set_parameters
function set_parameters(node::AAPairNode, eqfreqs::Array{Float64, 1},  S::Array{Float64,2}, t::Float64)
  node.eqfreqs = eqfreqs
  node.logeqfreqs = log(eqfreqs)
  node.S = S
  node.Q = zeros(Float64,20,20)
  for i=1:20
    for j=1:20
      node.Q[i,j] = S[i,j]*eqfreqs[j]
    end
  end
  for i=1:20
    node.Q[i,i] = 0.0
    for j=1:20
      if i != j
        node.Q[i,i] -= node.Q[i,j]
      end
    end
  end

  node.D, node.V = eig(node.Q)
  node.Vi = inv(node.V)
  node.Pt = node.V*Diagonal(exp(node.D*t))*node.Vi
  for i=1:20
    for j=1:20
      if node.Pt[i,j] > 0.0
        node.logPt[i,j] = log(node.Pt[i,j])
      else
        node.logPt[i,j] = -1e10
        node.Pt[i,j] = 0.0
      end
    end
  end
end

function set_parameters(node::AAPairNode, eqfreqs::Array{Float64, 1}, t::Float64)
  set_parameters(node, eqfreqs, node.S, t)
end

function set_parameters(node::AAPairNode, t::Float64)
  if(t != node.t)
    node.t = t
    node.Pt = node.V*Diagonal(exp(node.D*t))*node.Vi
    for i=1:20
      for j=1:20
        if node.Pt[i,j] > 0.0
          node.logPt[i,j] = log(node.Pt[i,j])
        else
          node.logPt[i,j] = -1e10
          node.Pt[i,j] = 0.0
        end
      end
    end
  end
end

export load_parameters
function load_parameters(node::AAPairNode, parameter_file)
  f = open(parameter_file)
  lines = readlines(f)
  close(f)

  S = zeros(Float64,20, 20)

  for i=1:20
   spl = split(lines[i])
   for j=1:length(spl)
     S[i+1,j] = parse(Float64, spl[j])
     S[j,i+1] = S[i+1,j]
   end
  end

  for i=1:20
    for j=1:20
      if i != j
        S[i,i] -= S[i,j]
      end
    end
  end

  eqfreqs = zeros(Float64,20)
  spl = split(lines[21])
  for i=1:20
    eqfreqs[i] = parse(Float64, spl[i])
  end

  set_parameters(node, eqfreqs, S, 1.0)
end

export get_data_lik
function get_data_lik(node::AAPairNode, x0::Int)
  if x0 > 0
    return node.logeqfreqs[x0]
  else
    return 0.0
  end
end

function get_data_lik(node::AAPairNode, x0::Int, xt::Int, t::Float64)
  if x0 > 0 && xt > 0
    set_parameters(node, t)
    return node.logeqfreqs[x0] + node.logPt[x0,xt]
  elseif x0 > 0
    return node.logeqfreqs[x0]
  elseif xt > 0
    return node.logeqfreqs[xt]
  end

  return 0.0
end

export sample
function sample(node::AAPairNode, rng::AbstractRNG, x0::Int, xt::Int, t::Float64)
  a = x0
  b = xt
  set_parameters(node, t)
  if a <= 0 && b <= 0
    a = sample(rng, node.eqfreqs)
    b = sample(rng, node.Pt[a,:])
  elseif a <= 0
    a = sample(rng, node.Pt[b,:])
  elseif b <= 0
    b = sample(rng, node.Pt[a,:])
  end
  return a,b
end

#pairnode = AAPairNode()
#load_parameters(pairnode, "/home/michael/dev/moju/lg_LG.PAML.txt")

#=
rng = MersenneTwister(2204104198511)
println(sample(pairnode, rng, 0, 0,1.0))
println(sample(pairnode,rng,  1, 0,1.0))
println(sample(pairnode,rng,  0, 5,1.0))
println(sample(pairnode, rng,  8, 5,1.0))
=#
#println(pairnode.Q)
#println(pairnode.Pt)
#=
set_parameters(pairnode,1.0)
println(pairnode.Pt)
set_parameters(pairnode,10.0)
println(pairnode.Pt)
set_parameters(pairnode,100.0)
println(pairnode.Pt)
=#
