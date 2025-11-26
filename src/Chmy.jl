module Chmy

using KernelAbstractions, StaticArrays
import LinearAlgebra

export STerm, SExprHead, Call, Ind, Loc, SUniform, Tag, SRef, SFun, SIndex, SExpr
export isexpr, iscall, isind, isloc, head, children, operation, arguments, argument, indices, location
include("expressions.jl")

export Space, Segment, Point
export scale, offset
include("spaces.jl")

export Grid, dims
include("grids.jl")

export AbstractRule, Passthrough, Postwalk, Fixpoint
export stencil_rule, lower_stencil, lift
include("rewriters.jl")

export AbstractDerivative, AbstractPartialDerivative, CentralDifference, StaggeredCentralDifference, PartialDerivative
include("derivatives.jl")

export Binding, push, binding_types
include("binding.jl")

export compute, to_expr
include("compute.jl")

export AbstractTensor, AbstractPermutationGroup, IdentityGroup, SymmetricGroup, SymmetricTensor, AsymmetricTensor, Vec
export order, dimensions, symmetry, dcontract, ⊡
include("tensors.jl")

include("show.jl")

end # module Chmy
