using DataStructures
#using Formatting


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
      return -1e10
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
      #=
      if a == -Inf || isnan(a)
        if isnan(b)
          return -1e10
        end
        return b
      elseif b == -Inf || isnan(b)
        if isnan(a)
          return -1e10
        end
        return a
      elseif a < b
        return b + log1p(quickExp(a-b))
      else
        return a + log1p(quickExp(b-a))
      end=#

      if a == -Inf || isnan(a)
        if isnan(b)
          return -Inf
        end
        return b
      elseif b == -Inf || isnan(b)
        if isnan(a)
          return -Inf
        end
        return a
      else
        v = a - b
        if v < 0.0
          if v < -20.0
            return b
          else
            return b + log1p(exp(v))
          end
        else
          if v > 20.0
            return a
          else
            return a + log1p(exp(-v))
          end
        end
      end
  end



  function logsumexp(v::Array{Float64,1})
      sum = -Inf
      for a in v
        sum = logsumexp(sum, a)
      end
      return sum
  end

  export logsumexpstable
  function logsumexpstable(v::Array{Float64,1}, start::Int, stop::Int)
    if start == stop
      return v[start]
    elseif start+1 == stop
      a = v[start]
      b = v[stop]
      v = a - b
      if v < 0.0
        return b + log1p(exp(v))
      else
        return a + log1p(exp(-v))
      end
    else
      len = div(stop-start,2)
      a = logsumexpstable(v,start,start+len-1)
      b = logsumexpstable(v,start+len,stop)
      v = a - b
      if v < 0.0
        return b + log1p(exp(v))
      else
        return a + log1p(exp(-v))
      end
    end
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
      # returns a sample from a log categorical distribution
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
