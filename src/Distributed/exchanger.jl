# the send and receive buffers of one field in the exchanges of dimension `dim`: vectors of the length of its faces, kept from one
# exchange to the next (the pack and unpack copies index faces and buffers linearly)
struct HaloBuffers
    dim::Int
    eltype::DataType
    backend::Backend
    nsend::Int
    nrecv::Int
    send::AbstractVector
    recv::AbstractVector
end

# state of the halo exchange of one side of a dimension, reused from one exchange to the next: requests, buffers, bookkeeping
struct Exchanger
    buffers::Vector{HaloBuffers}     # every buffer pair of this side so far (dimension, element type and lengths)
    send_reqs::Vector{MPI.Request}
    recv_reqs::Vector{MPI.Request}
    send_bufs::Vector{AbstractArray} # the buffers of the current exchange, one per field
    recv_bufs::Vector{AbstractArray}
    received::Base.RefValue{Bool}    # all messages of the current exchange received and unpacked

    function Exchanger()
        buffers   = HaloBuffers[]
        send_reqs = MPI.Request[]
        recv_reqs = MPI.Request[]
        send_bufs = AbstractArray[]
        recv_bufs = AbstractArray[]
        return new(buffers, send_reqs, recv_reqs, send_bufs, recv_bufs, Ref(false))
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

function Base.resize!(ex::Exchanger, count::Integer)
    resize!(ex.send_reqs, count)
    resize!(ex.recv_reqs, count)
    resize!(ex.send_bufs, count)
    resize!(ex.recv_bufs, count)
    return
end

function reset_reqs!(ex::Exchanger)
    fill!(ex.send_reqs, MPI.REQUEST_NULL)
    fill!(ex.recv_reqs, MPI.REQUEST_NULL)
    ex.received[] = false
    return
end

function halo_buffer(backend::Backend, T::DataType, n::Integer)
    buffer = KernelAbstractions.allocate(backend, T, n)
    # used by one exchange at a time, after the previous one has completed: the implicit synchronization would only order its uses
    # after complete work (and add a wait when the exchanging task's stream changes)
    disable_task_sync!(buffer)
    return buffer
end

# the buffers of field `idx`: a pair of this dimension, element type and lengths that no field before it in this exchange has
# taken, or a new one
function take_buffers!(ex::Exchanger, backend::Backend, ::Dim{D}, side::Side, field::Field, idx::Integer) where {D}
    T     = eltype(field)
    nsend = length(get_send_view(side, Dim(D), field))
    nrecv = length(get_recv_view(side, Dim(D), field))
    k = findfirst(ex.buffers) do b
        b.dim == D && b.eltype === T && b.nsend == nsend && b.nrecv == nrecv && b.backend == backend &&
            !any(j -> ex.send_bufs[j] === b.send, 1:(idx-1))
    end
    if isnothing(k)
        push!(ex.buffers, HaloBuffers(D, T, backend, nsend, nrecv, halo_buffer(backend, T, nsend), halo_buffer(backend, T, nrecv)))
        k = lastindex(ex.buffers)
    end
    ex.send_bufs[idx] = ex.buffers[k].send
    ex.recv_bufs[idx] = ex.buffers[k].recv
    return
end

function init!(ex::Exchanger, backend::Backend, dim::Dim, side::Side, fields::Tuple{Vararg{Field,K}}) where {K}
    resize!(ex, K)
    reset_reqs!(ex)
    ntuple(Val(K)) do idx
        Base.@_inline_meta
        take_buffers!(ex, backend, dim, side, fields[idx], idx)
    end
    return
end
