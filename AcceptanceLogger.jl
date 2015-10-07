using DataStructures
using Formatting

type AcceptanceLogger
    accepted::Accumulator{AbstractString, Int}
    total::Accumulator{AbstractString, Int}

    function AcceptanceLogger()
        return new(counter(AbstractString),counter(AbstractString))
    end
end

function logAccept!(logger::AcceptanceLogger, move::AbstractString)
    push!(logger.accepted, move)
    push!(logger.total, move)
end

function logReject!(logger::AcceptanceLogger, move::AbstractString)
    push!(logger.total, move)
end

function clear!(logger::AcceptanceLogger)
    logger.accepted = counter(AbstractString)
    logger.total = counter(AbstractString)
end

function list(logger::AcceptanceLogger)
    logkeys = sort([k for k in keys(logger.total)])
    ret = String[]
    for k in logkeys
        ratio = fmt(".4f", logger.accepted[k]/logger.total[k])
        total = logger.total[k]
        push!(ret, "$k=$ratio ($total)")
    end
    return ret
end
