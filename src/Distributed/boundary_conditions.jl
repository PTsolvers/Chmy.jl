"""
    ExchangeData

Structure containing the information to exchange halos of one side of an N-dimensional array.
"""
mutable struct ExchangeData{SB,RB}
    send_buffer::SB
    recv_buffer::RB
    send_request::MPI.Request
    recv_request::MPI.Request
    ExchangeData(send_buf, recv_buf) = new{typeof(send_buf),typeof(recv_buf)}(send_buf, recv_buf, MPI.REQUEST_NULL, MPI.REQUEST_NULL)
end

"""
    ExchangeData(::Val{S}, ::Val{D}, field::Field) where {S,D}

Create `ExchangeData` to exchange halos on side `S` in the spatial direction `D`.
"""
function ExchangeData(::Val{S}, ::Val{D}, field::Field) where {S,D}
    send_view = get_send_view(Val(S), Val(D), field)
    recv_view = get_recv_view(Val(S), Val(D), field)
    send_buffer = similar(parent(send_view), eltype(send_view), size(send_view))
    recv_buffer = similar(parent(recv_view), eltype(recv_view), size(recv_view))
    return ExchangeData(send_buffer, recv_buffer)
end

const EBC = ExchangeData

"""
    bc!(::Val{S}, ::Val{D},
        arch::DistributedArchitecture,
        grid::StructuredGrid,
        fields::NTuple{N,Field},
        exchange_datas::NTuple{N,ExchangeData}; async=false) where {S,D,N}

Perform a non-blocking MPI exchange for a set of fields.
"""
function BoundaryConditions.bc!(::Val{S}, ::Val{D},
                                arch::DistributedArchitecture,
                                grid::StructuredGrid,
                                fields::NTuple{N,Field},
                                exchange_datas::NTuple{N,EBC}; async=false) where {S,D,N}
    comm = cart_comm(topology(arch))
    nbrank = neighbor(topology(arch), D, S)
    @assert nbrank != MPI.PROC_NULL "no neighbor to communicate"

    # initiate non-blocking MPI recieve and device-to-device copy to the send buffer
    for idx in eachindex(fields)
        data = exchange_datas[idx]
        data.recv_request = MPI.Irecv!(data.recv_buffer, comm; source=nbrank)
        send_view = get_send_view(Val(S), Val(D), fields[idx])
        copyto!(data.send_buffer, send_view)
    end
    KernelAbstractions.synchronize(backend(arch))

    # initiate non-blocking MPI send
    for idx in eachindex(fields)
        info = exchange_datas[idx]
        info.send_request = MPI.Isend(info.send_buffer, comm; dest=nbrank)
    end

    recv_ready = falses(N)
    send_ready = falses(N)

    # test send and receive requests, initiating device-to-device copy
    # to the receive buffer if the receive is complete
    while !(all(recv_ready) && all(send_ready))
        for idx in eachindex(fields)
            data = exchange_datas[idx]
            if MPI.Test(data.recv_request) && !recv_ready[idx]
                recv_view = get_recv_view(Val(S), Val(D), fields[idx])
                copyto!(recv_view, data.recv_buffer)
                recv_ready[idx] = true
            end
            send_ready[idx] = MPI.Test(data.send_request)
        end
        yield()
    end
    async || KernelAbstractions.synchronize(backend(arch))

    return
end

overlap(::Vertex) = 1
overlap(::Center) = 0

get_recv_view(side::Val{S}, dim::Val{D}, f::Field) where {S,D} = get_recv_view(side, dim, parent(f), halo(f))

function get_send_view(side::Val{S}, dim::Val{D}, f::Field) where {S,D}
    get_send_view(side, dim, parent(f), halo(f), overlap(location(f, dim)))
end

function get_recv_view(::Val{1}, ::Val{D}, array::AbstractArray, halo_width::Integer) where {D}
    recv_range = (halo_width+1):(2halo_width)
    indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_recv_view(::Val{2}, ::Val{D}, array::AbstractArray, halo_width::Integer) where {D}
    recv_range = (size(array, D)-2halo_width+1):(size(array, D)-halo_width)
    indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_send_view(::Val{1}, ::Val{D}, array::AbstractArray, halo_width::Integer, overlap::Integer) where {D}
    send_range = (overlap+2halo_width+1):(overlap+3halo_width)
    indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_send_view(::Val{2}, ::Val{D}, array::AbstractArray, halo_width::Integer, overlap::Integer) where {D}
    send_range = (size(array, D)-overlap-3halo_width+1):(size(array, D)-overlap-2halo_width)
    indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end
