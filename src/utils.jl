function inversion_count(t::NTuple{N}) where {N}
    s = 0
    for i in 1:(N-1), j in (i+1):N
        s += t[i] > t[j]
    end
    return s
end

function diagonal(f, dims, rank, state::Bool=true)::Bool
    state || return false
    for i in 1:dims
        state = f(ntuple(_ -> i, rank), state)::Bool
        state || return false
    end
    return state
end

function visit_cartesian(f, inds, r, dims, state::Bool)::Bool
    state || return false
    iszero(r) && return f(inds, state)::Bool
    for i in 1:dims
        inds = @inbounds Base.setindex(inds, i, r)
        state = visit_cartesian(f, inds, r - 1, dims, state)
        state || return false
    end
    return state
end

function cartesian(f, dims, rank, state::Bool=true)::Bool
    state || return false
    inds = ntuple(one, rank)
    iszero(rank) && return f(inds, state)::Bool
    return visit_cartesian(f, inds, rank, dims, state)
end

function visit_ordered(f, inds, r, ub, gap, state::Bool)::Bool
    state || return false
    iszero(r) && return f(inds, state)::Bool
    for i in 1:ub
        inds = @inbounds Base.setindex(inds, i, r)
        state = visit_ordered(f, inds, r - 1, i - gap, gap, state)
        state || return false
    end
    return state
end

function increasing(f, dims, rank, state::Bool=true)::Bool
    state || return false
    inds = ntuple(identity, rank)
    iszero(rank) && return f(inds, state)::Bool
    rank > dims && return state
    return visit_ordered(f, inds, rank, dims, 1, state)
end

function nondecreasing(f, dims, rank, state::Bool=true)::Bool
    state || return false
    inds = ntuple(one, rank)
    iszero(rank) && return f(inds, state)::Bool
    return visit_ordered(f, inds, rank, dims, 0, state)
end

flatten(args::Tuple{Vararg{Tuple}}) = (Iterators.flatten(args)...,)
flatten(args::Vararg{Tuple})        = flatten(args)
