
export angular_rmsd
function angular_rmsd(theta1::Array{Float64, 1}, theta2::Array{Float64})

  dist = 0.0
  len = 0
  for i=1:length(theta1)
    if theta1[i] > -100.0 && theta2[i] > -100.0
      x0 = cos(theta1[i])
      x1 = sin(theta1[i])
      y0 = cos(theta2[i])
      y1 = sin(theta2[i])
      c = y0-x0
      s = y1-x1
      dist += c*c + s*s
      len += 1
    end
  end

  return sqrt(dist/float(len))
end

export angular_rmsd
function angular_rmsd(theta1::Float64, theta2::Float64)
  if theta1 > -100.0 && theta2 > -100.0
    #=
    x0 = cos(theta1)
    x1 = sin(theta1)
    y0 = cos(theta2)
    y1 = sin(theta2)
    c = y0-x0
    s = y1-x1
    return sqrt(c*c + s*s)=#
    return min(abs(theta1-theta2), 2.0*pi - abs(theta1-theta2))

  end

  return -1.0
end



function angular_rmsd(theta1::Array{Float64, 1}, theta2::Array{Float64},  align1::Array{Int}, align2::Array{Int})

  dist =0.0
  len = 0
  for (a,b) in zip(align1, align2)
    if a > 0 && b > 0
      if theta1[a] > -100.0 && theta2[b] > -100.0
        x0 = cos(theta1[a])
        x1 = sin(theta1[a])
        y0 = cos(theta2[b])
        y1 = sin(theta2[b])
        c = y0-x0
        s = y1-x1
        dist += c*c + s*s
        len += 1
      end
    end
  end

  return sqrt(dist/float(len))

  #=
  dist =0.0
  len = 0
  for (a,b) in zip(align1, align2)
    if a > 0 && b > 0
      if theta1[a] > -100.0 && theta2[b] > -100.0
        dist += (1.0 - cos(theta2[b]-theta1[a]))/2.0
        len += 1
      end
    end
  end

  return dist/float(len)
  =#
end

export angular_mean
function angular_mean(theta::Array{Float64, 1})
  if length(theta) == 0
    return -1000.0
  end

  c = 0.0
  s = 0.0
  total = float(length(theta))
  for t in theta
    c += cos(t)
    s += sin(t)
  end
  c /= total
  s /= total
  rho = sqrt(c*c + s*s)

  if s > 0
    return acos(c/rho)
  else
    return 2*pi - acos(c / rho)
  end
end

export angular_rmsd
function angular_rmsd(phi0::Float64, phit::Float64, psi0::Float64, psit::Float64)
  if phi0 > -100.0 && phit > -100.0 && psi0 > -100.0 && psit > -100.0
    #d1 = min(abs(phi0-phit), 2.0*pi - abs(phi0-phit))
    #d2 = min(abs(psi0-psit), 2.0*pi - abs(psi0-psit))
    #println(phi0,"\t",phit,"\t",psi0,"\t",psit)
    #println("phi",phi0-phit,"\t",d1)
    #println("psi",psi0-psit,"\t",d2)
    #println("d=",sqrt(d1*d1 + d2*d2))
    #return sqrt(d1*d1 + d2*d2)
    return sqrt(2.0*(2.0 - cos(phi0-phit) - cos(psi0-psit)))
  end
  return -1.0
end

export percentilerank
function percentilerank(v::Array{Float64,1}, x::Float64)
  c = 0.0
  len = length(v)
  for i=1:len
    if v[i] < x
      c += 1.0
    end
  end
  if len > 5
    return c/len
  else
    return -1.0
  end
end

export angular_rmsd_rank
function angular_rmsd_rank(phi0::Array{Float64, 1}, psi0::Array{Float64}, phit::Float64, psit::Float64, d::Float64)
  v = Float64[]
  for i=1:length(phi0)
    if phi0[i] > -100.0 && psi0[i] > -100.0 && phit > -100.0  && psit > -100.0
      push!(v,angular_rmsd(phi0[i], phit, psi0[i], psit))
    end
  end
  return percentilerank(v,d)
end

function angular_rmsd(phi0::Array{Float64, 1}, phit::Array{Float64, 1}, psi0::Array{Float64}, psit::Array{Float64})

  dist =0.0
  len = 0
  for i=1:length(phi0)
    if phi0[i] > -100.0 && phit[i] > -100.0 && psi0[i] > -100.0 && psit[i] > -100.0
      dist += sqrt(2.0*(2.0 - cos(phi0[i]-phit[i]) - cos(psi0[i]-psit[i])))
      len += 1
      #=
      d1 = min(abs(phi0[i]-phit[i]), 2.0*pi - abs(phi0[i]-phit[i]))
      d2 = min(abs(psi0[i]-psit[i]), 2.0*pi - abs(psi0[i]-psit[i]))
      dist += d1*d1 + d2*d2
      len += 1=#
    end
  end

  return dist/float(len)
end

function angular_rmsd(phi0::Array{Float64, 1}, phit::Array{Float64, 1}, psi0::Array{Float64}, psit::Array{Float64},  align1::Array{Int}, align2::Array{Int})

  dist =0.0
  len = 0
  for (a,b) in zip(align1, align2)
    if a > 0 && b > 0
      if phi0[a] > -100.0 && phit[b] > -100.0 && psi0[a] > -100.0 && psit[b] > -100.0
        dist += sqrt(2.0*(2.0 - cos(phi0[a]-phit[b])- cos(psi0[a]-psit[b])))
        len += 1
        #d1 = min(abs(phi0[a]-phit[b]), 2.0*pi - abs(phi0[a]-phit[b]))
        #d2 = min(abs(psi0[a]-psit[b]), 2.0*pi - abs(psi0[a]-psit[b]))
        #dist += d1*d1 + d2*d2
        len += 1
      end
    end
  end

  return dist/float(len)
end

export pimod
function pimod(angle::Float64)
  theta = mod2pi(angle)
  if theta > pi
    return theta -2.0*pi
  else
    return theta
  end
end
