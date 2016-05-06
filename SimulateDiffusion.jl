using Gadfly

t = 3.0
steps = 8000
x0 = -float(pi)/2.0
alpha = 7.5
sigma = 0.75
mu = float(pi)/2.0
dt = t / float(steps)
rng = MersenneTwister(9259781854156713)

xt = x0
times1=Float64[]
path1 = Float64[]
for i=1:steps
  push!(times1, (i-1)*dt)
  push!(path1,mod2pi(xt+pi)-pi)
  xt = xt + alpha*sin(mu-xt)*dt + sqrt(dt)*sigma*randn(rng)
end

xt = x0
times2=Float64[]
path2 = Float64[]
for i=1:steps
  push!(times2, (i-1)*dt)
  push!(path2,mod2pi(xt+pi)-pi)
  xt = xt + alpha*sin(mu-xt)*dt + sqrt(dt)*sigma*randn(rng)
end

xt = x0
times3=Float64[]
path3 = Float64[]
for i=1:steps
  push!(times3, (i-1)*dt)
  push!(path3,mod2pi(xt+pi)-pi)
  xt = xt + alpha*sin(mu-xt)*dt + sqrt(dt)*sigma*randn(rng)
end
horx = [0.0, t]
hory = [mu, mu]
p = plot(layer(x=horx, y=hory, Geom.line, Theme(default_color=color("black"))),
         layer(x=times1, y=path1, Geom.point, Theme(default_color=color("red"),default_point_size=1pt,discrete_highlight_color=c->nothing) ),
         layer(x=times2, y=path2, Geom.point, Theme(default_color=color("blue"),default_point_size=1pt,discrete_highlight_color=c->nothing) ),
         layer(x=times3, y=path3, Geom.point, Theme(default_color=color("green"),default_point_size=1pt,discrete_highlight_color=c->nothing) ),
         Coord.Cartesian(ymin=-float(pi), ymax=float(pi)), Guide.xlabel("t"), Guide.ylabel("theta"), Guide.title(string("x0=-π/2",", alpha=",alpha,", mu=π/2",",sigma=",sigma)))
draw(SVG("simulations.svg", 5inch, 5inch), p)