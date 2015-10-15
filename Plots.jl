using Gadfly
using Compose



function bin2d(phi::Array{Float64, 1}, psi::Array{Float64, 1})
  nbins = 20
  for (a,b) in zip(phi,psi)
    c = int((a / float(pi))*nbins)
    d = int((b / float(pi))*nbins)
    println(c,"\t",d)
  end
end

#=using RDatasets


using Compose, Gadfly

rng = MersenneTwister(2414802720182)
mat = rand(rng,25,25)*0.1

#draw(SVG("spy.svg", 5inch, 5inch), spy(mat))

p = plot(mat, x="Year", y="Country", color="GDP", Geom.rectbin)
draw(SVG("spy.svg", 5inch, 5inch), p)
=#
#Pkg.add("PyPlot")

#=
using PyPlot
x = linspace(0,2*pi,1000); y = sin(3*x + 4*cos(2*x))
p=plot(x, y, color="red", linewidth=2.0, linestyle="--")
savefig("try.svg")
=#
