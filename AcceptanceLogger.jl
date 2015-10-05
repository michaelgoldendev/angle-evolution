using DataStructures
using Formatting

type AcceptanceLogger
    accepted::Accumulator{ASCIIString, Int}
    total::Accumulator{ASCIIString, Int}

    function AcceptanceLogger()
        return new(counter(ASCIIString),counter(ASCIIString))
    end
end

function logAccept!(logger::AcceptanceLogger, move::ASCIIString)
    push!(logger.accepted, move)
    push!(logger.total, move)
end

function logReject!(logger::AcceptanceLogger, move::ASCIIString)
    push!(logger.total, move)
end

function clear!(logger::AcceptanceLogger)
    logger.accepted = counter(ASCIIString)
    logger.total = counter(ASCIIString)
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