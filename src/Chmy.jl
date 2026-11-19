module Chmy

using KernelAbstractions

using Moshi: Data.@data, Data.isa_variant, Match.@match

using OrderedCollections: LittleDict
using PrettyTables: pretty_table, TextTableFormat
import Adapt

import LinearAlgebra: ⋅, ×, tr, det, diag, transpose
import Base: broadcasted

using StaticArrays: SVector, SMatrix

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# re-export from LinearAlgebra
export ⋅, ×, tr, det, diag, transpose

export DTerm, Literal, Coordinate, Operator, Fun
export 𝓅, 𝓈, ↓, ↑, Coord, Trace, Position
export AnyNormal, AnyTangent, 𝐧, 𝐭
include("types.jl")

export isexpr, iscall
export isliteral, iscoordinate, istensor, iscomp, isinds, ispack, isuniform
export head, children, operation, arguments
export value, argument, arity, isunary, isbinary
export components, entries, indices
export tensorrank, tensorname, tensorkind
export ==ₛ, !=ₛ, <ₛ, isstaticzero, isstaticone
export 𝒪, ℐ
include("dterm.jl")

const ZERO = Literal(0)
const ONE = Literal(1)

export ξ¹, ξ², ξ³
const ξ¹ = Coordinate(1)
const ξ² = Coordinate(2)
const ξ³ = Coordinate(3)

export ⊡, ⊗, sym, asym, cof, adj, gram, cogram
include("operators.jl")

export evaluate, component
include("evaluate.jl")

include("utils.jl")

export AbstractRule, Passthrough, Chain, Prewalk, Postwalk, Fixpoint
include("rewriters.jl")

export pack
include("tensors.jl")

export @scalars, @vectors, @tensors, @uniform, @sym, @diag, @alt
include("macros.jl")

export Interpolate, Extrapolate, InterpolateFrom
export WallPolicy, ReduceOrder, KeepOrder, ReadGhosts
export locations, interpolate
include("interpolate.jl")

export Grids
include("grids.jl")

export CentralDifference, ForwardDifference, BackwardDifference
export StaggeredCentralDifference, StaggeredForwardDifference, StaggeredBackwardDifference
include("finite_difference.jl")

export DifferentialOperator
export Gradient, Divergence, Curl
include("calculus.jl")

include("topology.jl")

export lower_rule, Project
include("lowering.jl")

export Lifted
include("lift.jl")

export FacePattern, WallPattern, RegionPattern
export Wall, AlongAxis, AnyWall, AnyCorner, LeftWall, RightWall, BottomWall, TopWall, BackWall, FrontWall
export Ghost, Boundary, Interior, Within
export OverridePredicate, OverrideMatch, FaceMatch, FacePredicate, WallPredicate, signed_axes, signed_distances
export Override, override, 𝒷
export BoundaryRule, Add, Scale, WallData, WallValue, WallDerivative, boundary_rule
include("override.jl")

export subs
include("subs.jl")

export Binding
export findkey, push, pairstuple, bindings
include("binding.jl")

export nonuniforms
include("nonuniforms.jl")

export compile
include("compile.jl")

include("show.jl")

end # module Chmy
