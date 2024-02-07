module Chmy

using KernelAbstractions

export remove_dim, insert_dim

include("macros.jl")
include("utils.jl")
include("kernel_abstractions.jl")

include("Architectures.jl")
include("Grids/Grids.jl")
include("GridOperators/GridOperators.jl")
include("Fields/Fields.jl")

include("BoundaryConditions/BoundaryConditions.jl")

include("Workers.jl")
include("Distributed/Distributed.jl")

using .Grids
using .GridOperators
using .Fields

end # module Chmy
