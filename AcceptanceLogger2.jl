using DataStructures
using Formatting

export AcceptanceLogger
type AcceptanceLogger
    accepted::Accumulator{AbstractString, Int}
    total::Accumulator{AbstractString, Int}

    function AcceptanceLogger()
        return new(counter(AbstractString),counter(AbstractString))
    end
end

export logAccept!
function logAccept!(logger::AcceptanceLogger, move::AbstractString)
    push!(logger.accepted, move)
    push!(logger.total, move)
end

export logReject!
function logReject!(logger::AcceptanceLogger, move::AbstractString)
    push!(logger.total, move)
end

export clear!
function clear!(logger::AcceptanceLogger)
    logger.accepted = counter(AbstractString)
    logger.total = counter(AbstractString)
end

export getacceptanceratio
function getacceptanceratio(logger::AcceptanceLogger, move::AbstractString)
  return logger.accepted[move]/logger.total[move]
end

export list
function list(logger::AcceptanceLogger)
    logkeys = sort([k for k in keys(logger.total)])
    ret = AbstractString[]
    for k in logkeys
        ratio = fmt(".4f", logger.accepted[k]/logger.total[k])
        total = logger.total[k]
        push!(ret, "$k=$ratio ($total)")
    end
    return ret
end
