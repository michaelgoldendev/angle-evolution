using DataStructures
using Formatting

module UtilsModule
  include("AcceptanceLogger.jl")
  include("AngleUtils.jl")

  export quickExp
  export safelog
  export orderpair
  export logsumexp
  export sample
  export GumbelSample

  function quickExp(v::Float64)
      if v < -57.0
          return 0.0
      elseif v == 0.0
          return 1.0
      end

      return exp(v)
  end


  function safelog(x::Float64)
    if x < 0.0
      println("X=",x)
      return -Inf
    else
      return log(x)
    end
  end


  function orderpair(a::Float64, b::Float64)
    if a < b
      return a,b
    end
    return b,a
  end


  function logsumexp(a::Float64, b::Float64)
      if a == -Inf
        return b
      elseif b == -Inf
        return a
      elseif a < b
        return b + log1p(quickExp(a-b))
      else
        return a + log1p(quickExp(b-a))
      end

      #=
      mn, mx = orderpair(a,b)
      #mx = max(a,b)
      #mn = min(a,b)
      return mx + log1p(quickExp(mn-mx))
      #return mx + log(quickExp(a-mx)+quickExp(b-mx))
      =#
  end



  function logsumexp(v::Array{Float64,1})
      sum = -Inf
      for a in v
        sum = logsumexp(sum, a)
      end
      return sum
  end

  function sample(rng::AbstractRNG, v::Array{Float64})
      s = sum(v)
      n = length(v)
      r = rand(rng)
      cumsum = 0.0
      for i=1:n
          cumsum += v[i] / s
          if cumsum >= r
              return i
          end
      end

      return n
  end

  function sample(rng::AbstractRNG, v::Array{Float64}, s::Float64)
      n = length(v)
      r = rand(rng)*s
      cumsum = 0.0
      for i=1:n
          @inbounds cumsum += v[i]
          if cumsum >= r
              return i
          end
      end

      return n
  end

  function GumbelSample(rng::AbstractRNG, v::Array{Float64})
      n = length(v)
      index = 1
      max = v[1] -log(-log(rand(rng)))
      for i=2:n
        if v[i] != -Inf
          y = v[i] -log(-log(rand(rng)))
          if y > max
            max = y
            index = i
          end
        end
      end

      return index
  end
end
