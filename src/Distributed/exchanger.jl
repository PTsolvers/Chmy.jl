# state of the halo exchange of one side of a dimension, reused from one exchange to the next: requests, buffers, bookkeeping
struct Exchanger
    storage::Dict{Backend,StackAllocator}
    send_reqs::Vector{MPI.Request}
    recv_reqs::Vector{MPI.Request}
    send_bufs::Vector{AbstractArray}
    recv_bufs::Vector{AbstractArray}
    recv_done::Vector{Bool}

    function Exchanger()
        storage   = Dict{Backend,StackAllocator}()
        send_reqs = MPI.Request[]
        recv_reqs = MPI.Request[]
        send_bufs = AbstractArray[]
        recv_bufs = AbstractArray[]
        recv_done = Bool[]
        return new(storage, send_reqs, recv_reqs, send_bufs, recv_bufs, recv_done)
    end
end

"""
    halo_exchangers(arch)

State of the halo exchanges of both sides of a dimension on `arch` (one per side: both are in flight at the same time), owned by an
overlapped `Launcher` and passed to [`exchange_halo!`](@ref); `nothing` for an architecture without halo exchange. The state of
one exchange must not be used by another exchange at the same time.
"""
halo_exchangers(::Architecture) = nothing
halo_exchangers(::DistributedArchitecture) = (Exchanger(), Exchanger())

# the state of the calling task, for the exchanges called without one (`exchange_halo!(arch, grid, fields...)`, `bc!` of a batch set)
function task_local_exchangers()
    return get!(() -> (Exchanger(), Exchanger()), task_local_storage(), :chmy_exchangers)::NTuple{2,Exchanger}
end

function get_allocator(ex::Exchanger, backend::Backend)
    return get!(() -> StackAllocator(backend), ex.storage, backend)
end

function Base.resize!(ex::Exchanger, count::Integer)
    resize!(ex.send_reqs, count)
    resize!(ex.recv_reqs, count)
    resize!(ex.send_bufs, count)
    resize!(ex.recv_bufs, count)
    resize!(ex.recv_done, count)
    return
end

function reset_reqs!(ex::Exchanger)
    fill!(ex.send_reqs, MPI.REQUEST_NULL)
    fill!(ex.recv_reqs, MPI.REQUEST_NULL)
    fill!(ex.recv_done, false)
    return
end

function init!(ex::Exchanger, backend::Backend, dim, side, fields::Tuple{Vararg{Field}})
    resize!(ex, length(fields))
    reset_reqs!(ex)

    alloc_size = 0
    foreach(fields) do field
        send_view = get_send_view(side, dim, field)
        recv_view = get_recv_view(side, dim, field)
        alloc_size += prod(size(send_view)) * sizeof(eltype(field))
        alloc_size += prod(size(recv_view)) * sizeof(eltype(field))
    end

    tla = get_allocator(ex, backend)
    resize!(tla, alloc_size)

    foreach(enumerate(fields)) do (idx, field)
        send_view = get_send_view(side, dim, field)
        recv_view = get_recv_view(side, dim, field)
        ex.send_bufs[idx] = allocate(tla, eltype(field), size(send_view))
        ex.recv_bufs[idx] = allocate(tla, eltype(field), size(recv_view))
    end

    return
end

reset_allocators!(ex::Exchanger) = foreach(reset!, values(ex.storage))
