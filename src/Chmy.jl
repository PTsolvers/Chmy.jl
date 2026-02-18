module Chmy

using KernelAbstractions
import LinearAlgebra: ⋅, ×, tr, det, diag, transpose

include("utils.jl")
include("staticcoef.jl")

export STerm, SExprHead, Call, Comp, Loc, Ind, SUniform, SRef, SFun, SIndex, SExpr
export isexpr, iscall, isind, isloc, head, children, operation, arguments, argument, arity, indices, location
export value, isstaticzero, isstaticone
include("expressions.jl")

export Space, Segment, Point
export scale, offset
include("spaces.jl")

export Grid, dims
include("grids.jl")

export AbstractDerivative, AbstractPartialDerivative, CentralDifference, StaggeredCentralDifference, PartialDerivative
include("derivatives.jl")

export DifferentialOperator, AbstractGradient, AbstractDivergence, AbstractCurl
export Gradient, Divergence, Curl
include("calculus.jl")

export STensor, SScalar, SVec
export SSymTensor, SAltTensor, SDiagTensor, SZeroTensor, SIdTensor
export Tensor, SymTensor, AltTensor, DiagTensor, Vec
export tensorrank, tensorkind, name
export isalternating, issymmetric, isdiag
include("tensors.jl")

# re-exported from LinearAlgebra
export ⋅, ×, tr, det, diag, transpose

export ⊡, ⊗, sym, asym, adj, gram, cogram
include("operators.jl")

export AbstractRule, Passthrough, Prewalk, Postwalk, Fixpoint
export stencil_rule, lower_stencil, lift, subs
include("rewriters.jl")

export Binding, push, binding_types
include("binding.jl")

export compute, to_expr
include("compute.jl")

export canonicalize, seval
include("canonicalize.jl")

include("show.jl")

end # module Chmy
