function BoundaryConditions.bc!(side::Val{S}, dim::Val{D},
                                arch::DistributedArchitecture,
                                ::StructuredGrid,
                                batch::ExchangeBatch{K}; async=false) where {S,D,K}
    comm = cart_comm(topology(arch))
    nbrank = neighbor(topology(arch), D, S)
    @assert nbrank != MPI.PROC_NULL "no neighbor to communicate"

    tle = task_local_exchanger()

    init!(tle, batch, backend(arch), dim, side)

    # initiate non-blocking MPI recieve and device-to-device copy to the send buffer
    for idx in eachindex(batch.fields)
        tle.recv_reqs[idx] = MPI.Irecv!(tle.recv_bufs[idx], comm; source=nbrank)
        send_view = get_send_view(Val(S), Val(D), batch.fields[idx])
        copyto!(tle.send_bufs[idx], send_view)
    end
    KernelAbstractions.synchronize(backend(arch))

    # initiate non-blocking MPI send
    for idx in eachindex(batch.fields)
        tle.send_reqs[idx] = MPI.Isend(tle.send_bufs[idx], comm; dest=nbrank)
    end

    recv_ready = falses(K)
    send_ready = falses(K)

    # test send and receive requests, initiating device-to-device copy
    # to the receive buffer if the receive is complete
    while !(all(recv_ready) && all(send_ready))
        for idx in eachindex(batch.fields)
            if MPI.Test(tle.recv_reqs[idx]) && !recv_ready[idx]
                recv_view = get_recv_view(Val(S), Val(D), batch.fields[idx])
                copyto!(recv_view, tle.recv_bufs[idx])
                recv_ready[idx] = true
            end
            send_ready[idx] = MPI.Test(tle.send_reqs[idx])
        end
        yield()
    end

    reset_allocators!(tle)
    async || KernelAbstractions.synchronize(backend(arch))

    return
end

# TODO: remove Exchange bcs where there are no neighbors

function BoundaryConditions.batch(::DistributedArchitecture,
                                  grid::StructuredGrid{N},
                                  f_bcs::Vararg{BoundaryConditions.FieldAndBC,K}; replace::Bool=false) where {N,K}
    fs, bcs = zip(f_bcs...)
    bcs_reg = map(x -> BoundaryConditions.regularise(grid, x), bcs) |> BoundaryConditions.reorder
    if replace
        return _batch_with_replace(fs, bcs_reg, grid)
    else
        return _batch_with_remove(fs, bcs_reg, grid)
    end
end

# ntuple version is type unstable for some reason
@generated function _batch_with_remove(fs::NTuple{K,Field{<:Any,N}}, bcs::Tuple, grid::StructuredGrid{N}) where {N,K}
    quote
        @inline
        Base.Cartesian.@ntuple $N D -> begin
            Base.Cartesian.@ntuple 2 S -> begin
                if bcs[D][S] isa Tuple{Vararg{Nothing}} || connectivity(grid, Val(D), Val(S)) isa Connected
                    EmptyBatch()
                else
                    FieldBatch(BoundaryConditions.prune(fs, bcs[D][S])...)
                end
            end
        end
    end
end

# ntuple version is type unstable for some reason
@generated function _batch_with_replace(fs::NTuple{K,Field{<:Any,N}}, bcs::Tuple, grid::StructuredGrid{N}) where {N,K}
    quote
        @inline
        Base.Cartesian.@ntuple $N D -> begin
            Base.Cartesian.@ntuple 2 S -> begin
                if bcs[D][S] isa Tuple{Vararg{Nothing}}
                    EmptyBatch()
                elseif connectivity(grid, Val(D), Val(S)) isa Connected
                    exchanges = Base.Cartesian.@ntuple $K _ -> Exchange()
                    ExchangeBatch(fs, exchanges)
                else
                    FieldBatch(BoundaryConditions.prune(fs, bcs[D][S])...)
                end
            end
        end
    end
end
