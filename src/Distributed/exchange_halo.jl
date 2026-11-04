# copy between the halo buffers and the views of all fields of one side with one kernel on the task's stream: it does not block the
# host (copyto! of a contiguous view waits for the copy on AMDGPU), and its small workgroups run next to the kernels of the inner
# points (the copy kernels of AMDGPU and GPUArrays use workgroups of 512 and 1024 work-items, which wait until those end). One
# launch for all fields: a launch costs the host about as much as the MPI calls of a message. Buffers and views are indexed
# linearly. Host buffers (MPI without GPU support) are copied with copyto!.
@inline copy_halos_at!(I, ::Tuple{}, ::Tuple{}) = nothing
@inline function copy_halos_at!(I, dsts::Tuple, srcs::Tuple)
    dst = first(dsts)
    if I <= length(dst)
        @inbounds dst[I] = first(srcs)[I]
    end
    copy_halos_at!(I, Base.tail(dsts), Base.tail(srcs))
    return
end

@kernel function copy_halos_kernel!(dsts, srcs)
    I = @index(Global, Linear)
    copy_halos_at!(I, dsts, srcs)
end

copy_halos!(backend::Backend, dsts, srcs) = (copy_halos_kernel!(backend, 256, maximum(length, dsts))(dsts, srcs); return)
copy_halos!(::CPU, dsts, srcs) = (foreach(copyto!, dsts, srcs); return)

# the first K buffers of the current exchange as a tuple (for the copy kernel)
buffer_tuple(bufs::Vector{AbstractArray}, ::Val{K}) where {K} = ntuple(idx -> bufs[idx], Val(K))

exchange_backend(arch::DistributedArchitecture) = is_gpu_aware(arch) ? Architectures.get_backend(arch) : CPU()

# start the receives of one side and pack its send buffers
function post_receives!(ex::Exchanger, side::Side{S}, dim::Dim{D}, arch::DistributedArchitecture,
                        fields::Tuple{Vararg{Field,K}}) where {S,D,K}
    isempty(fields) && return
    nbrank = neighbor(topology(arch), D, S)
    @assert nbrank != MPI.PROC_NULL "no neighbor to communicate"
    init!(ex, exchange_backend(arch), dim, side, fields, cart_comm(topology(arch)), nbrank)
    for idx in 1:K
        MPI.Start(ex.recv_reqs[idx])
    end
    copy_halos!(exchange_backend(arch), buffer_tuple(ex.send_bufs, Val(K)), map(f -> get_send_view(side, dim, f), fields))
    return
end

# start the sends of one side, once its send buffers are packed
function post_sends!(ex::Exchanger, ::Side, ::Dim, ::DistributedArchitecture, fields::Tuple{Vararg{Field,K}}) where {K}
    isempty(fields) && return
    for idx in 1:K
        MPI.Start(ex.send_reqs[idx])
    end
    return
end

# unpack the messages of one side once all have arrived; true once all its receives and sends are complete
function test_exchange!(ex::Exchanger, side::Side, dim::Dim, arch::DistributedArchitecture, fields::Tuple{Vararg{Field,K}}) where {K}
    isempty(fields) && return true
    if !ex.received[]
        received = true
        for idx in 1:K
            received &= MPI.Test(ex.recv_reqs[idx])
        end
        received || return false
        copy_halos!(exchange_backend(arch), map(f -> get_recv_view(side, dim, f), fields), buffer_tuple(ex.recv_bufs, Val(K)))
        ex.received[] = true
    end
    sent = true
    for idx in 1:K
        sent &= MPI.Test(ex.send_reqs[idx])
    end
    return sent
end

"""
    exchange_halo!(dim::Dim, arch, grid, fields1::Tuple, fields2::Tuple; exchangers=nothing)

Perform the halo exchange of both sides of dimension `dim` at once: `fields1` on side 1 and `fields2` on side 2 (an empty tuple for
a side without exchange). The receives and sends of both sides are in flight together. `exchangers` is the state of the exchange
(requests, buffers) from [`halo_exchangers`](@ref), by default the calling task's.
"""
function exchange_halo!(dim::Dim, arch::DistributedArchitecture, ::StructuredGrid,
                        fields1::Tuple{Vararg{Field}}, fields2::Tuple{Vararg{Field}}; exchangers=nothing)
    isempty(fields1) && isempty(fields2) && return
    backend = Architectures.get_backend(arch)
    ex1, ex2 = isnothing(exchangers) ? task_local_exchangers() : exchangers

    # initiate non-blocking MPI receives and device-to-device copies to the send buffers
    post_receives!(ex1, Side(1), dim, arch, fields1)
    post_receives!(ex2, Side(2), dim, arch, fields2)
    KernelAbstractions.synchronize(backend)

    # initiate non-blocking MPI sends
    post_sends!(ex1, Side(1), dim, arch, fields1)
    post_sends!(ex2, Side(2), dim, arch, fields2)

    # test send and receive requests, initiating device-to-device copies from the receive buffers as receives complete
    while !(test_exchange!(ex1, Side(1), dim, arch, fields1) & test_exchange!(ex2, Side(2), dim, arch, fields2))
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
                           batch::Tuple{AbstractBatch,AbstractBatch}; exchangers=nothing)

Apply the boundary conditions of both sides of dimension `dim` on a distributed grid: the boundary condition kernels of the sides
without neighbor, then the halo exchanges of both sides at once, with the state `exchangers` (see [`exchange_halo!`](@ref)).
"""
function BoundaryConditions.bc!(dim::Dim, arch::DistributedArchitecture, grid::StructuredGrid,
                                batch::Tuple{AbstractBatch,AbstractBatch}; exchangers=nothing)
    batch[1] isa ExchangeBatch || bc!(Side(1), dim, arch, grid, batch[1])
    batch[2] isa ExchangeBatch || bc!(Side(2), dim, arch, grid, batch[2])
    exchange_halo!(dim, arch, grid, exchange_fields(batch[1]), exchange_fields(batch[2]); exchangers)
    return
end
