include("Utils.jl")

type CTMC
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

  function CTMC(eqfreqs::Array{Float64,1}, S::Array{Float64,2}, t::Float64)
    n = length(eqfreqs)
    Q = ones(Float64, n, n)
    for i=1:n
      for j=1:n
        Q[i,j] = S[i,j]*eqfreqs[j]
      end
    end
    for i=1:n
      Q[i,i] = 0.0
      for j=1:n
        if i != j
          Q[i,i] -= Q[i,j]
        end
      end
    end
    D, V = eig(Q)
    Vi = inv(V)
    Pt = V*Diagonal(exp(D*t))*Vi
    logPt = log(Pt)
    return new(eqfreqs, log(eqfreqs), S, Q, D,V, Vi, t, Pt, logPt)
  end
end

function set_parameters(node::CTMC, eqfreqs::Array{Float64, 1},  S::Array{Float64,2}, t::Float64)
  node.eqfreqs = eqfreqs
  node.logeqfreqs = log(eqfreqs)
  node.S = S
  n = length(node.eqfreqs)
  node.Q = zeros(Float64,n,n)
  for i=1:n
    for j=1:n
      node.Q[i,j] = S[i,j]*eqfreqs[j]
    end
  end
  for i=1:n
    node.Q[i,i] = 0.0
    for j=1:n
      if i != j
        node.Q[i,i] -= node.Q[i,j]
      end
    end
  end

  node.D, node.V = eig(node.Q)
  node.Vi = inv(node.V)
  node.Pt = node.V*Diagonal(exp(node.D*t))*node.Vi
  n = length(node.eqfreqs)
  for i=1:n
    for j=1:n
      if node.Pt[i,j] > 0.0
        node.logPt[i,j] = log(node.Pt[i,j])
      else
        node.logPt[i,j] = -1e10
        node.Pt[i,j] = 0.0
      end
    end
  end
end

function set_parameters(node::CTMC, eqfreqs::Array{Float64, 1}, t::Float64)
  set_parameters(node, eqfreqs, node.S, t)
end

function set_parameters(node::CTMC, t::Float64)
  if(t != node.t)
    node.t = t
    node.Pt = node.V*Diagonal(exp(node.D*t))*node.Vi
    n = length(node.eqfreqs)
    for i=1:n
      for j=1:n
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

function get_data_lik(node::CTMC, x0::Int)
  if x0 > 0
    return node.logeqfreqs[x0]
  else
    return 0.0
  end
end

function get_data_lik(node::CTMC, x0::Int, xt::Int, t::Float64)
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

function sample(node::CTMC, rng::AbstractRNG, x0::Int, xt::Int, t::Float64)
  a = x0
  b = xt
  if a <= 0 && b <= 0
    set_parameters(node, t)
    a = sample(rng, node.eqfreqs)
    b = sample(rng, node.Pt[a,:])
  elseif a <= 0
    a = sample(rng, node.Pt[b,:])
  elseif b <= 0
    b = sample(rng, node.Pt[a,:])
  end
  return a,b
end
