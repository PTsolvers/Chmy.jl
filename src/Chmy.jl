module Chmy

using MacroTools
using KernelAbstractions

export Dim, Side, Left, Right
export remove_dim, insert_dim

include("macros.jl")
include("utils.jl")

include("Architectures.jl")
include("Grids/Grids.jl")
include("GridOperators/GridOperators.jl")
include("Fields/Fields.jl")

include("BoundaryConditions/BoundaryConditions.jl")

include("Workers.jl")
include("Distributed/Distributed.jl")

include("KernelLaunch.jl")

using .Grids
using .Fields
using .Architectures

end # module Chmy
