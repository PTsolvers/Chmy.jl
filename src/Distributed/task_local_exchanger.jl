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

function init!(tle::TaskLocalExchanger, backend::Backend, dim, side, fields::Tuple{Vararg{Field}})
    resize!(tle, length(fields))
    reset_reqs!(tle)

    alloc_size = 0
    foreach(fields) do field
        send_view = get_send_view(side, dim, field)
        recv_view = get_recv_view(side, dim, field)
        alloc_size += prod(size(send_view)) * sizeof(eltype(field))
        alloc_size += prod(size(recv_view)) * sizeof(eltype(field))
    end

    tla = get_allocator(tle, backend)
    resize!(tla, alloc_size)

    foreach(enumerate(fields)) do (idx, field)
        send_view = get_send_view(side, dim, field)
        recv_view = get_recv_view(side, dim, field)
        tle.send_bufs[idx] = allocate(tla, eltype(field), size(send_view))
        tle.recv_bufs[idx] = allocate(tla, eltype(field), size(recv_view))
    end

    return
end

reset_allocators!(tle::TaskLocalExchanger) = foreach(reset!, values(tle.storage))
