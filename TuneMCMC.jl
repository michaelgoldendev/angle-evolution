export TuneMCMC
type TuneMCMC
  Nmoves::Int
  hlength::Int
  history::Array{Int,2}
  factor::Float64
  factors::Array{Float64,1}

  function TuneMCMC(Nmoves::Int)
    hlength = 20
    history = zeros(Int, Nmoves, hlength+1)
    new(Nmoves,hlength,history,1.5, ones(Float64,Nmoves))
  end

  function TuneMCMC(mcmcmoves::TuneMCMC)
    new(mcmcmoves.Nmoves, mcmcmoves.hlength, deepcopy(mcmcmoves.history), mcmcmoves.factor, deepcopy(mcmcmoves.factors))
  end
end

export logacceptance
function logacceptance(mcmc::TuneMCMC, move::Int, acceptancerate::Float64)
  mcmc.history[move,1] += 1
  if mcmc.history[move,1] <= 1 || mcmc.history[move,1] == mcmc.hlength+2
    mcmc.history[move,1] = 2
  end
  h = mcmc.history[move,1]
  if acceptancerate < 0.2
    mcmc.history[move,h] = 1
  elseif acceptancerate > 0.3
    mcmc.history[move,h] = 3
  else
    mcmc.history[move,h] = 2
  end

  mcmc.factors[move] *= computefactor(mcmc,move)
end

export computefactor
function computefactor(mcmc::TuneMCMC,move::Int)
  sum = 0.0
  total = 0.0
  p = 0.8
  for i=1:mcmc.hlength
    if mcmc.history[move,1] != 0
      index = ((mcmc.history[move,1]+ i - 2) % mcmc.hlength) + 2
      d = mcmc.hlength - i
      #println("C",mcmc.history[move,1],"\t", i,"\t",index,"\t",mcmc.history[move,index],"\t",d)
      if mcmc.history[move,index] == 1
        v = p^d
        sum += -v
        total += v
      elseif mcmc.history[move,index] == 2
        v = p^d
        sum += 0.0
        total += v*2.0
      elseif mcmc.history[move,index] == 3
        v = p^d
        sum += v
        total += v
      end
    end
  end

  a = sum / total
  println("OOO ",a)
  #println("Z",a)
  if a > 0.25
    return mcmc.factor
  elseif a < -0.25
    return 1.0/mcmc.factor
  else
    return 1.0
  end
end

export getfactor
function getfactor(mcmc::TuneMCMC, move::Int)
  return mcmc.factors[move]
end

export gethistory
function gethistory(mcmc::TuneMCMC, move::Int)
  s = ""
  for i=2:mcmc.hlength+1
    s = string(s, mcmc.history[move,i])
  end
  return s
end

mcmc = TuneMCMC(5)
logacceptance(mcmc,1,0.2)
println(getfactor(mcmc,1))
logacceptance(mcmc,1,0.4)
println(getfactor(mcmc,1))
logacceptance(mcmc,1,0.1)
println(getfactor(mcmc,1))
logacceptance(mcmc,1,0.2)
println(getfactor(mcmc,1))
logacceptance(mcmc,1,0.1)
println(getfactor(mcmc,1))
logacceptance(mcmc,1,0.1)
println(getfactor(mcmc,1))