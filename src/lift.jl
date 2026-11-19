struct Lifted{O <: Operator, K} <: Operator
    op::O
    axes::NTuple{K, Int}

    """
        Lifted(op, axes)
        Lifted(op, axes...)

    Wraps kD operator `op` so that, when lowered at nD positions, the k-th position of `op` is taken along `axes[k]`.
    The axes may be in any order, e.g. `Lifted(op, 3, 1)` applies the first dimension of `op` along the third axis and
    the second along the first.
    """
    function Lifted(op::O, axes::NTuple{K, Integer}) where {O <: Operator, K}
        K > 0 || throw(ArgumentError("at least one lifted axis is required"))
        all(>(0), axes) || throw(ArgumentError("the lifted axes must be positive"))
        allunique(axes) || throw(ArgumentError("the lifted axes must be unique"))
        return new{O, K}(op, map(Int, axes))
    end
end
Lifted(op::Operator, axes::Vararg{Integer, N}) where {N} = Lifted(op, axes)

Base.getindex(op::Operator, axes::Vararg{Integer, N}) where {N} = Lifted(op, axes)

tensorrank(lf::Lifted, args) = tensorrank(lf.op, args)
checkranks(lf::Lifted, args::NTuple{N, DTerm}) where {N} = checkranks(lf.op, args)

# a nullary operator standing for its argument in the rule of a lifted operator: its sample at the positions of the
# rule is the sample of the argument at these positions along the lifted `axes`, and at `inds` along the other axes
struct Placeholder <: Operator
    arg::DTerm
    inds::Tuple{Vararg{Position}}
    axes::Tuple{Vararg{Int}}
end

lower_rule(p::Placeholder, ::Tuple{}, positions::Tuple) = p.arg[insert_positions(p.inds, p.axes, positions)...]

function lower_rule(lf::Lifted, args::Tuple, inds::NTuple{N, Position}) where {N}
    maximum(lf.axes) <= N || throw(ArgumentError("the lifted axes exceed the number of dimensions"))
    # tensor components are not mapped to the lifted axes yet, so only scalars can be lifted
    checkscalar(lf, args)
    return lower_rule(lf.op, map(t -> Placeholder(t, inds, lf.axes)(), args), map(a -> inds[a], lf.axes))::DTerm
end

# the samples of the rule have one position per lifted axis
function insert_positions(inds::NTuple{N, Position}, axes::NTuple{K, Int}, positions::NTuple{K, Position}) where {N, K}
    return ntuple(Val(N)) do a
        j = findfirst(==(a), axes)
        return isnothing(j) ? inds[a] : positions[j]
    end
end
