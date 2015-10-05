using Memoize

@memoize function fib(n::Int64)
    if n <= 2
      return 1
    else
      return fib(n-2)+fib(n-1)
  end
end

START = 1
MATCH = 2
XINSERT = 3
YINSERT = 4
END = 5
N1 = 6
N2 = 7
N3 = 8
N4 = 9

function pairhmm(delta::Float64, tau::Float64, epsilon::Float64)
  aligntransprobs = zeros(Float64, 5, 5)
  aligntransprobs[START,MATCH] = 1.0 - 2.0*delta - tau
  aligntransprobs[START,XINSERT] = delta
  aligntransprobs[START,YINSERT] = delta
  aligntransprobs[START,END] = tau
  aligntransprobs[MATCH,MATCH] = 1.0 - 2.0*delta - tau
  aligntransprobs[MATCH,XINSERT] = delta
  aligntransprobs[MATCH,YINSERT] = delta
  aligntransprobs[MATCH,END] = tau
  aligntransprobs[XINSERT,MATCH] = 1.0 - epsilon - tau
  aligntransprobs[XINSERT,XINSERT] = epsilon
  aligntransprobs[XINSERT,END] = tau
  aligntransprobs[YINSERT,MATCH] = 1.0 - epsilon - tau
  aligntransprobs[YINSERT,YINSERT] = epsilon
  aligntransprobs[YINSERT,END] = tau
  aligntransprobs[END,END] = 1.0
  hmmtransprobs = eye(30)

  n = 6
  m = 5
  sum = 0.0
  sum += tau*pairhmm(aligntransprobs,hmmtransprobs,n,m,MATCH,1)
  sum += tau*pairhmm(aligntransprobs,hmmtransprobs,n,m,XINSERT,1)
  sum += tau*pairhmm(aligntransprobs,hmmtransprobs,n,m,YINSERT,1)


end

function pairhmm(aligntransprobs::Array{Float64,2}, hmmtransprobs::Array{Float64,2}, i::Int, j::Int, alignnode::Int, h::Int)
  if i < 0 || j < 0
    return 0.0
  elseif i == 0 && j == 0
    if alignnode == MATCH
      return 1.0
    elseif alignnode == XINSERT || alignnode == YINSERT
      return 0.0
    end
  end

  sum = 0.0
  prevh = 1
  ret = ""
  for prevalignnode=1:5
    transprob = aligntransprobs[prevalignnode, alignnode]
    if transprob > 0.0
        prevlik = 1.0
        currlik = 1.0
        if alignnode == MATCH
          prevlik = pairhmm(aligntransprobs, hmmtransprobs, i-1, j-1, prevalignnode, prevh)
          currlik = 1.0
        elseif alignnode == XINSERT
          prevlik = pairhmm(aligntransprobs, hmmtransprobs, i-1, j, prevalignnode, prevh)
          currlik = 1.0
        elseif alignnode == YINSERT
          prevlik =  pairhmm(aligntransprobs, hmmtransprobs, i, j-1, prevalignnode, prevh)
          currlik = 1.0
        end
        sum += prevlik*transprob*currlik
        ret = string(ret,"\n", prevalignnode,"\t", alignnode,"\t",sum,"\t", prevlik,"\t", transprob,"\t", currlik)
    end
  end
  #println(aligntransprobs)
  #println(ret)
  return sum
end

function tkf92(lambda::Float64, mu::Float64, r::Float64, t::Float64)
  Bt = (1.0 - exp((lambda-mu)*t))/(mu - lambda*exp((lambda-mu)*t))

  aligntransprobs = zeros(Float64, 9, 9)
  aligntransprobs[START,N1] = 1.0

  aligntransprobs[MATCH,MATCH] = r
  aligntransprobs[MATCH,N1] = 1.0-r

  aligntransprobs[XINSERT,XINSERT] = r
  aligntransprobs[XINSERT,N3] = 1.0-r

  aligntransprobs[YINSERT,YINSERT] = r + (1.0-r)*(lambda*Bt)
  aligntransprobs[YINSERT,N2] = (1.0-r)*(1.0-lambda*Bt)

  aligntransprobs[END,END] = 0.0

  aligntransprobs[N1,YINSERT] =  lambda*Bt
  aligntransprobs[N1,N2] = 1.0 - lambda*Bt

  aligntransprobs[N2,END] = 1.0 - (lambda/mu)
  aligntransprobs[N2,N4] = lambda/mu

  aligntransprobs[N3,YINSERT] = (1.0 - mu*Bt - exp(-mu*t))/(1.0-exp(-mu*t))
  aligntransprobs[N3,N2] = (mu*Bt)/(1.0-exp(-mu*t))

  aligntransprobs[N4,MATCH] = exp(-mu*t)
  aligntransprobs[N4,XINSERT] = 1.0 - exp(-mu*t)

  println(aligntransprobs)

  hmmtransprobs = eye(30)

  n = 4
  m = 4
  return tkf92lik(aligntransprobs,hmmtransprobs,n,m,END,1)
end

function tkf92lik(aligntransprobs::Array{Float64,2}, hmmtransprobs::Array{Float64,2}, i::Int, j::Int, alignnode::Int, h::Int)
  if i < 0 || j < 0
    return 0.0
  elseif i == 0 && j == 0
    if alignnode == MATCH
      return 1.0
    elseif alignnode == XINSERT || alignnode == YINSERT
      return 0.0
    end
  end

  sum = 0.0
  prevh = 1
  ret = ""
  for prevalignnode=1:9
    transprob = aligntransprobs[prevalignnode, alignnode]
    if transprob > 0.0
        prevlik = 1.0
        currlik = 1.0
        if alignnode == MATCH
          prevlik = tkf92lik(aligntransprobs, hmmtransprobs, i-1, j-1, prevalignnode, prevh)
          currlik = 1.0
        elseif alignnode == XINSERT
          prevlik = tkf92lik(aligntransprobs, hmmtransprobs, i-1, j, prevalignnode, prevh)
          currlik = 1.0
        elseif alignnode == YINSERT
          prevlik =  tkf92lik(aligntransprobs, hmmtransprobs, i, j-1, prevalignnode, prevh)
          currlik = 1.0
        else
          prevlik =  tkf92lik(aligntransprobs, hmmtransprobs, i, j, prevalignnode, prevh)
          currlik = 1.0
        end

        sum += prevlik*transprob*currlik
        ret = string(ret,"\n", prevalignnode,"\t", alignnode,"\t",sum,"\t", prevlik,"\t", transprob,"\t", currlik)
    end
  end
  println(ret)
  return sum
end




#lik = pairhmm(0.2,0.1,0.25)
#println(lik)

println(tkf92(0.25, 0.4, 0.15, 0.00))
