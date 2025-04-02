module Chmy

using MacroTools
using KernelAbstractions

export
    # utils
    Dim, Side, Left, Right, remove_dim, insert_dim, Offset,

    # Architectures
    Architecture, SingleDeviceArchitecture, Arch, get_backend, get_device, activate!, set_device!,
    heuristic_groupsize, pointertype, disable_task_sync!, enable_task_sync!, with_no_task_sync!,

    # BoundaryConditions
    FieldBoundaryCondition, FirstOrderBC, Dirichlet, Neumann, bc!,
    BoundaryFunction,
    DimSide,
    AbstractBatch, FieldBatch, ExchangeBatch, EmptyBatch, BatchSet, batch,

    # Distributed
    CartesianTopology, global_rank, shared_rank, node_name, cart_comm, shared_comm,
    dims, cart_coords, neighbors, neighbor, has_neighbor, global_size, node_size,
    DistributedArchitecture, topology,
    exchange_halo!, gather!,

    # DoubleBuffer
    DoubleBuffer, swap!, front, back,

    # Fields
    AbstractField, Field, VectorField, TensorField, ConstantField, ZeroField, OneField, ValueField, FunctionField,
    location, halo, interior, set!,
    divg,

    # Grids
    Location, Center, Vertex, flip,
    Connectivity, Bounded, Connected, Periodic, Flat,
    AbstractAxis, UniformAxis, FunctionAxis,
    StructuredGrid, UniformGrid,
    nvertices, ncenters, spacing, inv_spacing, Δ, iΔ, volume, inv_volume,
    coord, coords, center, vertex, centers, vertices,
    origin, extent, bounds, axis,
    direction, axes_names, expand_loc,
    connectivity,

    Δx, Δy, Δz,
    xcoord, ycoord, zcoord,
    xcoords, ycoords, zcoords,
    xvertex, yvertex, zvertex,
    xcenter, ycenter, zcenter,
    xvertices, yvertices, zvertices,
    xcenters, ycenters, zcenters,

    # GridOperators
    left, right, δ, ∂,
    InterpolationRule, Linear, HarmonicLinear,
    itp, lerp, hlerp,
    divg, vmag,
    AbstractMask, FieldMask, FieldMask1D, FieldMask2D, FieldMask3D, at,

    leftx, rightx, δx, ∂x,
    lefty, righty, δy, ∂y,
    leftz, rightz, δz, ∂z,

    leftx_masked, rightx_masked, δx_masked, ∂x_masked,
    lefty_masked, righty_masked, δy_masked, ∂y_masked,
    leftz_masked, rightz_masked, δz_masked, ∂z_masked,

    # KernelLaunch
    Launcher,
    worksize, outer_width, inner_worksize, inner_offset, outer_worksize, outer_offset,

    # Workers
    Worker

include("macros.jl")
include("utils.jl")

include("Architectures.jl")
include("DoubleBuffers.jl")
include("Grids/Grids.jl")
include("Fields/Fields.jl")
include("GridOperators/GridOperators.jl")

include("BoundaryConditions/BoundaryConditions.jl")

include("Workers.jl")
include("Distributed/Distributed.jl")

include("KernelLaunch.jl")

using .Architectures
using .BoundaryConditions
using .Distributed
using .DoubleBuffers
using .Fields
using .Grids
using .GridOperators
using .KernelLaunch
using .Workers

end # module Chmy
