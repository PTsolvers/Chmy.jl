module Chmy

using KernelAbstractions

using Moshi: Data.@data, Data.isa_variant, Match.@match

using PrettyTables: pretty_table, TextTableFormat
import Adapt

import LinearAlgebra: ⋅, ×, tr, det, diag, transpose
import Base: broadcasted

# re-export from LinearAlgebra
export ⋅, ×, tr, det, diag, transpose

export Operator

export ==ₛ

"""
Abstract supertype for all Chmy operators.
"""
abstract type Operator end

export 𝓅, 𝓈
include("locations.jl")

module Kind

struct Sym end
struct Alt end
struct Diag end
struct None end

end

const TensorKind = Union{Kind.Sym,Kind.Alt,Kind.Diag,Kind.None}

export isexpr, iscall
export isliteral, isindex, istensor, iscomp, islocs, isinds, isuniform
export head, args, operation, arguments
export value, argument, arity, isunary, isbinary
export components, locations, indices
export tensorrank, tensorname, tensorkind

export DTerm, ==ₛ
include("dterm.jl")

export 𝑖, 𝑗, 𝑘

const 𝑖 = Index(1)
const 𝑗 = Index(2)
const 𝑘 = Index(3)

export ⊡, ⊗, sym, asym, adj, gram, cogram
include("operators.jl")

# include("operators.jl")

include("utils.jl")
# include("staticcoef.jl")

# export STerm, SExprHead, Call, Comp, Loc, Ind, SLiteral, SRef, SFun, SIndex, SExpr
# export isexpr, iscall, isind, isloc, head, children, operation, arguments, argument, arity, indices, location
# export isuniform, isliteral
# export value, isstaticzero, isstaticone
# include("expressions.jl")

# export AbstractRule, Passthrough, Chain, Prewalk, Postwalk, Fixpoint
# include("rewriters.jl")

# export Space, Segment, Point
# export offset
# include("spaces.jl")

# export Grid, dims
# include("grids.jl")

# export STensor, SUTensor, SScalar, SUScalar, SVec, SUVec
# export SSymTensor, SUSymTensor, SAltTensor, SUAltTensor, SDiagTensor, SUDiagTensor, SZeroTensor, SIdTensor
# export Tensor, SymTensor, AltTensor, DiagTensor, Vec, ZeroTensor, IdTensor
# export tensorrank, tensorkind, name
include("tensors.jl")

include("tensor_components.jl")

# export @scalars, @vectors, @tensors, @uniform, @sym, @diag, @alt, @id, @zero
# include("macros.jl")

# export AbstractDerivative, AbstractPartialDerivative, CentralDifference, StaggeredCentralDifference, PartialDerivative
# include("derivatives.jl")

# export AbstractAveraging, AbstractPartialAveraging, StaggeredLinearAveraging, PartialAveraging
# include("averaging.jl")

# export DifferentialOperator
# export Gradient, Divergence, Curl
# include("calculus.jl")

# export stencil_rule
# include("lowering.jl")

# export Binding, push, binding_types, pairstuple
# include("binding.jl")

# export Shift, CartesianShift, AxisFace, Lower, Upper, Span, Face, Stencil, Nonuniforms
# export δ, adjacent_faces, dim, codim, nonuniforms, reach, facet, facets, isfacet
# export BoundaryNormal, BoundaryTangent, BasisVector
# export GridOperator, operator
# include("grid_operator.jl")

# export HaloArray, halowidths, interior, halo
# include("halo_array.jl")

# export isless_lex
# include("isless_lex.jl")

# export canonicalize, simplify
# include("canonicalize.jl")

# export evaluate
# include("evaluate.jl")

# export compute, to_expr
# include("compute.jl")

# export lift
# include("lift.jl")

# export subs
# include("subs.jl")

include("show.jl")

end # module Chmy
