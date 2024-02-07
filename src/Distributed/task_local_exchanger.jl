struct TaskLocalExchanger
    storage::Dict{Backend,StackAllocator}
    send_reqs::Vector{MPI.Request}
    recv_reqs::Vector{MPI.Request}
    send_bufs::Vector{AbstractArray}
    recv_bufs::Vector{AbstractArray}

    function TaskLocalExchanger()
        storage   = Dict{Backend,StackAllocator}()
        send_reqs = MPI.Request[]
        recv_reqs = MPI.Request[]
        send_bufs = AbstractArray[]
        recv_bufs = AbstractArray[]
        return new(storage, send_reqs, recv_reqs, send_bufs, recv_bufs)
    end
end

function task_local_exchanger()
    return get!(() -> TaskLocalExchanger(), task_local_storage(), :chmy_comm_storage)
end

function get_allocator(tle::TaskLocalExchanger, backend::Backend)
    return get!(() -> StackAllocator(backend), tle.storage, backend)
end

function Base.resize!(tle::TaskLocalExchanger, count::Integer)
    resize!(tle.send_reqs, count)
    resize!(tle.recv_reqs, count)
    resize!(tle.send_bufs, count)
    resize!(tle.recv_bufs, count)
    return
end

function reset_reqs!(tle::TaskLocalExchanger)
    fill!(tle.send_reqs, MPI.REQUEST_NULL)
    fill!(tle.recv_reqs, MPI.REQUEST_NULL)
    return
end

@inline function pick_send_buffer(::TaskLocalExchanger, ex::Exchange{AbstractArray,<:Any}, T, dims, backend)
    return ex.send_buffer
end

@inline function pick_recv_buffer(::TaskLocalExchanger, ex::Exchange{<:Any,AbstractArray}, T, dims, backend)
    return ex.recv_buffer
end

@inline function pick_send_buffer(tle::TaskLocalExchanger, ::Exchange{Nothing,<:Any}, T, dims, backend)
    return allocate(get_allocator(tle, backend), T, dims)
end

@inline function pick_recv_buffer(tle::TaskLocalExchanger, ::Exchange{<:Any,Nothing}, T, dims, backend)
    return allocate(get_allocator(tle, backend), T, dims)
end

function init!(tle::TaskLocalExchanger, batch::ExchangeBatch, backend::Backend, dim, side)
    resize!(tle, length(batch.fields))
    reset_reqs!(tle)

    for idx in eachindex(batch.fields)
        field = batch.fields[idx]
        exch = batch.exchanges[idx]
        send_view = get_send_view(side, dim, field)
        recv_view = get_recv_view(side, dim, field)
        # use user-provided MPI buffers if any, otherwise use stack allocator
        tle.send_bufs[idx] = pick_send_buffer(tle, exch, eltype(field), size(send_view), backend)
        tle.recv_bufs[idx] = pick_recv_buffer(tle, exch, eltype(field), size(recv_view), backend)
    end

    return
end

reset_allocators!(tle::TaskLocalExchanger) = foreach(reset!, values(tle.storage))
