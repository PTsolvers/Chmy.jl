module Chmy

using KernelAbstractions
import Adapt

import LinearAlgebra: ⋅, ×, tr, det, diag, transpose

import Base: broadcasted

# re-export from LinearAlgebra
export ⋅, ×, tr, det, diag, transpose

include("utils.jl")
include("staticcoef.jl")

export STerm, SExprHead, Call, Comp, Loc, Ind, SLiteral, SRef, SFun, SIndex, SExpr
export isexpr, iscall, isind, isloc, head, children, operation, arguments, argument, arity, indices, location
export isuniform, isliteral
export value, isstaticzero, isstaticone
include("expressions.jl")

export AbstractRule, Passthrough, Chain, Prewalk, Postwalk, Fixpoint
include("rewriters.jl")

export Space, Segment, Point
export scale, offset
include("spaces.jl")

export Grid, dims
include("grids.jl")

export STensor, SUTensor, SScalar, SUScalar, SVec, SUVec
export SSymTensor, SUSymTensor, SAltTensor, SUAltTensor, SDiagTensor, SUDiagTensor, SZeroTensor, SIdTensor
export Tensor, SymTensor, AltTensor, DiagTensor, Vec, ZeroTensor, IdTensor
export tensorrank, tensorkind, name
include("tensors.jl")

export SNode, node, unwrap
include("snode.jl")

export @scalars, @vectors, @tensors, @uniform, @sym, @diag, @alt, @id, @zero
include("macros.jl")

export ⊡, ⊗, sym, asym, adj, gram, cogram
include("operators.jl")

export AbstractDerivative, AbstractPartialDerivative, CentralDifference, StaggeredCentralDifference, PartialDerivative
include("derivatives.jl")

export AbstractAveraging, AbstractPartialAveraging, StaggeredLinearAveraging, PartialAveraging
include("averaging.jl")

export DifferentialOperator
export Gradient, Divergence, Curl
include("calculus.jl")

export stencil_rule
include("lowering.jl")

export Binding, push, binding_types, pairstuple
include("binding.jl")

export isless_lex
include("isless_lex.jl")

export canonicalize, simplify
include("canonicalize.jl")

export evaluate
include("evaluate.jl")

export compute, to_expr
include("compute.jl")

export lift
include("lift.jl")

export subs
include("subs.jl")

include("show.jl")

end # module Chmy
