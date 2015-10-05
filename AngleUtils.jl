function angular_rmsd(theta1::Array{Float64, 1}, theta2::Array{Float64})
  dist =0.0
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
end





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

function pimod(angle::Float64)
  theta = mod2pi(angle)
  if theta > pi
    return theta -2.0*pi
  else
    return theta
  end
end
