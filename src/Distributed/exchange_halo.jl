# copy between a halo buffer and a view on the device with a kernel on the task's stream: it does not block the host (copyto! of
# a contiguous view waits for the copy on AMDGPU), and its small workgroups run next to the kernels of the inner points (the copy
# kernels of AMDGPU and GPUArrays use workgroups of 512 and 1024 work-items, which wait until those end); host buffers (MPI
# without GPU support) are copied with copyto!
@kernel function copy_halo_kernel!(dst, src)
    I = @index(Global, Linear)
    @inbounds dst[I] = src[I]
end

copy_halo!(backend::Backend, dst, src) = (copy_halo_kernel!(backend, 256, length(dst))(dst, src); return)
copy_halo!(::CPU, dst, src) = (copyto!(dst, src); return)

exchange_backend(arch::DistributedArchitecture) = is_gpu_aware(arch) ? Architectures.get_backend(arch) : CPU()

# post the receives of one side and pack its send buffers
function post_receives!(side::Side{S}, dim::Dim{D}, arch::DistributedArchitecture, fields::Tuple{Vararg{Field}}) where {S,D}
    isempty(fields) && return
    nbrank = neighbor(topology(arch), D, S)
    @assert nbrank != MPI.PROC_NULL "no neighbor to communicate"
    tle = task_local_exchanger(side)
    reset_allocators!(tle)
    init!(tle, exchange_backend(arch), dim, side, fields)
    for idx in eachindex(fields)
        tle.recv_reqs[idx] = MPI.Irecv!(tle.recv_bufs[idx], cart_comm(topology(arch)); source=nbrank)
        copy_halo!(exchange_backend(arch), tle.send_bufs[idx], get_send_view(side, dim, fields[idx]))
    end
    return
end

# post the sends of one side, once its send buffers are packed
function post_sends!(side::Side{S}, dim::Dim{D}, arch::DistributedArchitecture, fields::Tuple{Vararg{Field}}) where {S,D}
    isempty(fields) && return
    tle = task_local_exchanger(side)
    for idx in eachindex(fields)
        tle.send_reqs[idx] = MPI.Isend(tle.send_bufs[idx], cart_comm(topology(arch)); dest=neighbor(topology(arch), D, S))
    end
    return
end

# unpack the messages of one side that have arrived; true once all its receives and sends are complete
function test_exchange!(side::Side, dim::Dim, arch::DistributedArchitecture, fields::Tuple{Vararg{Field}})
    isempty(fields) && return true
    tle  = task_local_exchanger(side)
    done = true
    for idx in eachindex(fields)
        if !tle.recv_done[idx] && MPI.Test(tle.recv_reqs[idx])
            copy_halo!(exchange_backend(arch), get_recv_view(side, dim, fields[idx]), tle.recv_bufs[idx])
            tle.recv_done[idx] = true
        end
        done &= tle.recv_done[idx] & MPI.Test(tle.send_reqs[idx])
    end
    return done
end

"""
    exchange_halo!(dim::Dim, arch, grid, fields1::Tuple, fields2::Tuple)

Perform the halo exchange of both sides of dimension `dim` at once: `fields1` on side 1 and `fields2` on side 2 (an empty tuple for
a side without exchange). The receives and sends of both sides are in flight together.
"""
function exchange_halo!(dim::Dim, arch::DistributedArchitecture, ::StructuredGrid,
                        fields1::Tuple{Vararg{Field}}, fields2::Tuple{Vararg{Field}})
    isempty(fields1) && isempty(fields2) && return
    backend = Architectures.get_backend(arch)

    # initiate non-blocking MPI receives and device-to-device copies to the send buffers
    post_receives!(Side(1), dim, arch, fields1)
    post_receives!(Side(2), dim, arch, fields2)
    KernelAbstractions.synchronize(backend)

    # initiate non-blocking MPI sends
    post_sends!(Side(1), dim, arch, fields1)
    post_sends!(Side(2), dim, arch, fields2)

    # test send and receive requests, initiating device-to-device copies from the receive buffers as receives complete
    while !(test_exchange!(Side(1), dim, arch, fields1) & test_exchange!(Side(2), dim, arch, fields2))
        yield()
    end

    KernelAbstractions.synchronize(backend)

    return
end

"""
    exchange_halo!(side::Side, dim::Dim, arch, grid, fields...)

Perform halo exchange communication between neighboring processes in a distributed architecture.

# Arguments
- `side`: The side of the grid where the halo exchange is performed.
- `dim`: The dimension along which the halo exchange is performed.
- `arch`: The distributed architecture used for communication.
- `grid`: The structured grid on which the halo exchange is performed.
- `fields...`: The fields to be exchanged.
"""
function exchange_halo!(::Side{S}, dim::Dim, arch::DistributedArchitecture, grid::StructuredGrid,
                        fields::Vararg{Field,K}) where {S,K}
    S == 1 ? exchange_halo!(dim, arch, grid, fields, ()) : exchange_halo!(dim, arch, grid, (), fields)
    return
end

"""
    exchange_halo!(arch, grid, fields...)

Perform halo exchange for the given architecture, grid, and fields.

# Arguments
- `arch`: The distributed architecture to perform halo exchange on.
- `grid`: The structured grid on which halo exchange is performed.
- `fields`: The fields on which halo exchange is performed.
"""
function exchange_halo!(arch::DistributedArchitecture, grid::StructuredGrid{N}, fields::Vararg{Field}) where {N}
    ntuple(Val(N)) do D
        Base.@_inline_meta
        if connectivity(grid, Dim(N - D + 1), Side(1)) isa Connected
            exchange_halo!(Side(1), Dim(N - D + 1), arch, grid, fields...)
        end
        if connectivity(grid, Dim(N - D + 1), Side(2)) isa Connected
            exchange_halo!(Side(2), Dim(N - D + 1), arch, grid, fields...)
        end
    end
    return
end

"""
    BoundaryConditions.bc!(side::Side, dim::Dim,
                                arch::DistributedArchitecture,
                                grid::StructuredGrid,
                                batch::ExchangeBatch)

Apply boundary conditions on a distributed grid with halo exchange performed internally.

# Arguments
- `side`: The side of the grid where the halo exchange is performed.
- `dim`: The dimension along which the halo exchange is performed.
- `arch`: The distributed architecture used for communication.
- `grid`: The structured grid on which the halo exchange is performed.
- `batch`: The batch set to apply boundary conditions to.
"""
function BoundaryConditions.bc!(side::Side,
                                dim::Dim,
                                arch::DistributedArchitecture,
                                grid::StructuredGrid,
                                batch::ExchangeBatch)
    exchange_halo!(side, dim, arch, grid, batch.fields...)
    return
end

exchange_fields(batch::ExchangeBatch) = batch.fields
exchange_fields(::AbstractBatch)      = ()

"""
    BoundaryConditions.bc!(dim::Dim, arch::DistributedArchitecture, grid::StructuredGrid,
                           batch::Tuple{AbstractBatch,AbstractBatch})

Apply the boundary conditions of both sides of dimension `dim` on a distributed grid: the boundary condition kernels of the sides
without neighbor, then the halo exchanges of both sides at once.
"""
function BoundaryConditions.bc!(dim::Dim, arch::DistributedArchitecture, grid::StructuredGrid,
                                batch::Tuple{AbstractBatch,AbstractBatch})
    batch[1] isa ExchangeBatch || bc!(Side(1), dim, arch, grid, batch[1])
    batch[2] isa ExchangeBatch || bc!(Side(2), dim, arch, grid, batch[2])
    exchange_halo!(dim, arch, grid, exchange_fields(batch[1]), exchange_fields(batch[2]))
    return
end
