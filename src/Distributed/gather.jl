"""
    gather!(dst, src, comm::MPI.Comm; root=0)

Gather local array `src` into a global array `dst`.
Size of the global array `size(dst)` should be equal to the product of the size of a local array `size(src)` and the dimensions of a Cartesian communicator `comm`.
The array will be gathered on the process with id `root` (`root=0` by default).
Note that the memory for a global array should be allocated only on the process with id `root`, on other processes `dst` can be set to `nothing`.
"""
function gather!(dst::Union{AbstractArray{T,N},Nothing}, src::AbstractArray{T,N}, comm::MPI.Comm; root=0) where {T,N}
    dims, _, _ = MPI.Cart_get(comm)
    dims = Tuple(dims)
    if MPI.Comm_rank(comm) == root
        # make subtype for gather
        offset  = Tuple(0 for _ in 1:N)
        subtype = MPI.Types.create_subarray(size(dst), size(src), offset, MPI.Datatype(eltype(dst)))
        subtype = MPI.Types.create_resized(subtype, 0, size(src, 1) * Base.elsize(dst))
        MPI.Types.commit!(subtype)
        # make VBuffer for collective communication
        counts  = fill(Cint(1), reverse(dims)) # gather one subarray from each MPI rank
        displs  = zeros(Cint, reverse(dims))   # reverse dims since MPI Cart comm is row-major
        csizes  = cumprod(size(src)[2:end] .* dims[1:end-1])
        strides = (1, csizes...)
        for I in CartesianIndices(displs)
            offset = reverse(Tuple(I - oneunit(I)))
            displs[I] = sum(offset .* strides)
        end
        recvbuf = MPI.VBuffer(dst, vec(counts), vec(displs), subtype)
        MPI.Gatherv!(src, recvbuf, comm; root)
    else
        MPI.Gatherv!(src, nothing, comm; root)
    end
    return
end

"""
    gather!(arch, dst, src::Field; kwargs...)

Gather the interior of a field `src` into a global array `dst`.
"""
function gather!(arch::DistributedArchitecture, dst, src::Field; kwargs...)
    gather!(dst, interior(src), cart_comm(topology(arch)); kwargs...)
end
