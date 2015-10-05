using Distributions

d = Truncated(Normal(0.1, 0.2), 0.0, Inf)
rng = MersenneTwister(429420294331)

out = open("serialize.jls","w")
serialize(out, d)
close(out)

in = open("serialize.jls","r")
e::Truncated = deserialize(in)
close(in)

println(e)