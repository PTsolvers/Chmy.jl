# the send and receive buffers of one field in the exchanges of dimension `dim` with rank `peer`: vectors of the length of its faces
# (the pack and unpack copies index faces and buffers linearly), and persistent requests for them (MPI_Send_init/MPI_Recv_init,
# started at every exchange), kept from one exchange to the next: no allocation per exchange
struct HaloBuffers
    dim::Int
    eltype::DataType
    backend::Backend
    nsend::Int
    nrecv::Int
    comm::MPI.Comm
    peer::Int
    send::AbstractVector
    recv::AbstractVector
    send_req::MPI.Request
    recv_req::MPI.Request
end

function HaloBuffers(dim, T, backend, nsend, nrecv, comm, peer)
    send = halo_buffer(backend, T, nsend)
    recv = halo_buffer(backend, T, nrecv)
    send_req = MPI.Send_init(send, comm; dest=peer)
    recv_req = MPI.Recv_init(recv, comm; source=peer)
    return HaloBuffers(dim, T, backend, nsend, nrecv, comm, peer, send, recv, send_req, recv_req)
end

# state of the halo exchange of one side of a dimension, reused from one exchange to the next: requests, buffers, bookkeeping
struct Exchanger
    buffers::Vector{HaloBuffers}     # every buffer pair of this side so far (dimension, element type and lengths)
    send_reqs::Vector{MPI.Request}   # the requests and buffers of the current exchange, one per field
    recv_reqs::Vector{MPI.Request}
    send_bufs::Vector{AbstractArray}
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

function halo_buffer(backend::Backend, T::DataType, n::Integer)
    buffer = KernelAbstractions.allocate(backend, T, n)
    # used by one exchange at a time, after the previous one has completed: the implicit synchronization would only order its uses
    # after complete work (and add a wait when the exchanging task's stream changes)
    disable_task_sync!(buffer)
    return buffer
end

# the buffers and requests of field `idx`: a pair of this dimension, element type, lengths and peer that no field before it in this
# exchange has taken, or a new one
function take_buffers!(ex::Exchanger, backend::Backend, ::Dim{D}, side::Side, field::Field, idx::Integer, comm::MPI.Comm,
                       peer::Integer) where {D}
    T     = eltype(field)
    nsend = length(get_send_view(side, Dim(D), field))
    nrecv = length(get_recv_view(side, Dim(D), field))
    k = findfirst(ex.buffers) do b
        b.dim == D && b.eltype === T && b.nsend == nsend && b.nrecv == nrecv && b.backend == backend && b.comm === comm &&
            b.peer == peer && !any(j -> ex.send_bufs[j] === b.send, 1:(idx-1))
    end
    if isnothing(k)
        push!(ex.buffers, HaloBuffers(D, T, backend, nsend, nrecv, comm, peer))
        k = lastindex(ex.buffers)
    end
    b = ex.buffers[k]
    ex.send_bufs[idx], ex.recv_bufs[idx] = b.send, b.recv
    ex.send_reqs[idx], ex.recv_reqs[idx] = b.send_req, b.recv_req
    return
end

# the buffers and requests of an exchange of `fields` with rank `peer` on `comm`
function init!(ex::Exchanger, backend::Backend, dim::Dim, side::Side, fields::Tuple{Vararg{Field,K}}, comm::MPI.Comm,
               peer::Integer) where {K}
    resize!(ex, K)
    ex.received[] = false
    ntuple(Val(K)) do idx
        Base.@_inline_meta
        take_buffers!(ex, backend, dim, side, fields[idx], idx, comm, peer)
    end
    return
end
