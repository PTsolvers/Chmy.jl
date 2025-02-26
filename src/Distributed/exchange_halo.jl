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
function exchange_halo!(side::Side{S}, dim::Dim{D},
                        arch::DistributedArchitecture,
                        ::StructuredGrid,
                        fields::Vararg{Field,K}) where {S,D,K}
    comm = cart_comm(topology(arch))
    nbrank = neighbor(topology(arch), D, S)
    @assert nbrank != MPI.PROC_NULL "no neighbor to communicate"

    tle = task_local_exchanger()

    reset_allocators!(tle)
    init!(tle, Architectures.get_backend(arch), dim, side, fields)

    # initiate non-blocking MPI receive and device-to-device copy to the send buffer
    for idx in eachindex(fields)
        tle.recv_reqs[idx] = MPI.Irecv!(tle.recv_bufs[idx], comm; source=nbrank)
        send_view = get_send_view(Side(S), Dim(D), fields[idx])
        copyto!(tle.send_bufs[idx], send_view)
    end
    KernelAbstractions.synchronize(Architectures.get_backend(arch))

    # initiate non-blocking MPI send
    for idx in eachindex(fields)
        tle.send_reqs[idx] = MPI.Isend(tle.send_bufs[idx], comm; dest=nbrank)
    end

    recv_ready = falses(K)
    send_ready = falses(K)

    # test send and receive requests, initiating device-to-device copy
    # to the receive buffer if the receive is complete
    while !(all(recv_ready) && all(send_ready))
        for idx in eachindex(fields)
            if MPI.Test(tle.recv_reqs[idx]) && !recv_ready[idx]
                recv_view = get_recv_view(Side(S), Dim(D), fields[idx])
                copyto!(recv_view, tle.recv_bufs[idx])
                recv_ready[idx] = true
            end
            send_ready[idx] = MPI.Test(tle.send_reqs[idx])
        end
        yield()
    end

    KernelAbstractions.synchronize(Architectures.get_backend(arch))

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
function BoundaryConditions.bc!(side::Side, dim::Dim,
                                arch::DistributedArchitecture,
                                grid::StructuredGrid,
                                batch::ExchangeBatch,
                                workers)
    put!(workers) do
        exchange_halo!(side, dim, arch, grid, batch.fields...)
    end
    return
end
