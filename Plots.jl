using Gadfly
using Compose
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
