abstract type BoundaryFunction{F} end

struct ReducedDimensions end
struct FullDimensions end

struct ContinuousBoundaryFunction{F,P,RF} <: BoundaryFunction{F}
    fun::F
    parameters::P
    ContinuousBoundaryFunction{RF}(fun::F, params::P) where {RF,F,P} = new{F,P,RF}(fun, params)
end

struct DiscreteBoundaryFunction{F,P,RF} <: BoundaryFunction{F}
    fun::F
    parameters::P
    DiscreteBoundaryFunction{RF}(fun::F, params::P) where {F,P,RF} = new{F,P,RF}(fun, params)
end

const CBF{F,P,RF} = ContinuousBoundaryFunction{F,P,RF} where {F,P,RF}
const DBF{F,P,RF} = DiscreteBoundaryFunction{F,P,RF} where {F,P,RF}

const CDBF{F,P} = Union{CBF{F,P},DBF{F,P}} where {F,P}

@inline _params(::CDBF{F,Nothing}) where {F} = ()
@inline _params(cbf::CDBF{F}) where {F} = cbf.parameters

@inline _reduce(::Type{ReducedDimensions}, dim, I) = remove_dim(dim, I)
@inline _reduce(::Type{FullDimensions}, dim, I) = I

@propagate_inbounds function (bc::CBF{F,P,RF})(grid, loc, dim, I::Vararg{Integer}) where {F,P,RF}
    bc.fun(_reduce(RF, dim, coord(grid, loc, I...))..., _params(bc)...)
end

@propagate_inbounds function (bc::DBF{F,P,RF})(grid, loc, dim, I::Vararg{Integer}) where {F,P,RF}
    bc.fun(grid, loc, dim, _reduce(RF, dim, I)..., _params(bc)...)
end

@propagate_inbounds function value(bc::FirstOrderBC{<:BoundaryFunction}, grid::SG{N}, loc, dim, I::Vararg{Integer,N}) where {N}
    bc.value(grid, loc, dim, I...)
end

# Create a continous or discrete boundary function
# if discrete = true, the function has signature f(grid, loc, dim, inds...)
# if reduce_dims = false, the boundary condition function accepts the same number of coordinates as the number of indices
function BoundaryFunction(fun::Function; discrete=false, parameters=nothing, reduce_dims=true)
    RF = reduce_dims ? ReducedDimensions : FullDimensions
    discrete ? DiscreteBoundaryFunction{RF}(fun, parameters) : ContinuousBoundaryFunction{RF}(fun, parameters)
end

# constructor for FirstOrderBC supporting functions
function FirstOrderBC{Kind}(f::F) where {Kind,F<:Function}
    bf = ContinuousBoundaryFunction{ReducedDimensions}(f, nothing)
    return FieldBoundaryCondition{Kind}(bf)
end
