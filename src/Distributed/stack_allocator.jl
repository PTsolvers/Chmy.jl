"""
    mutable struct StackAllocator

Simple stack (a.k.a. bump/arena) allocator.
Maintains an internal buffer that grows dynamically if the requested allocation exceeds current buffer size.
"""
mutable struct StackAllocator{B<:AbstractVector{UInt8}}
    buffer::B
    offset::UInt
    nallocs::Int
end

"""
    StackAllocator(backend::Backend)

Create a stack allocator using the specified backend to store allocations.
"""
function StackAllocator(backend::Backend)
    buffer = KernelAbstractions.allocate(backend, UInt8, 0)
    return StackAllocator(buffer, UInt(0), 0)
end

"""
    reset!(sa::StackAllocator)

Reset the stack allocator by resetting the pointer. Doesn't free the internal memory buffer.
"""
function reset!(sa::StackAllocator)
    sa.offset  = UInt(0)
    sa.nallocs = 0
    return
end

"""
    resize!(sa::StackAllocator, sz::Integer)

Resize the StackAllocator's buffer to capacity of `sz` bytes.
This method will throw an error if any arrays were already allocated using this allocator.
"""
function Base.resize!(sa::StackAllocator, sz::Integer)
    if sa.offset != UInt(0)
        error("reset StackAllocator before resizing")
    end
    if sz > length(sa.buffer)
        resize!(sa.buffer, ceil(Int, 1.5 * sz)) # add extra capacity
    end
    return
end

"""
    nallocs(sa::StackAllocator)

Get the number of allocations made by the given `StackAllocator`.
"""
nallocs(sa::StackAllocator) = sa.nallocs

"""
    allocate(sa::StackAllocator, T::DataType, dims, [align=sizeof(T)])

Allocate a buffer of type `T` with dimensions `dims` using a stack allocator.
The `align` parameter specifies the alignment of the buffer elements.

# Arguments
- `sa::StackAllocator`: The stack allocator object.
- `T::DataType`: The data type of the requested allocation.
- `dims`: The dimensions of the requested allocation.
- `align::Integer`: The alignment of the allocated buffer in bytes.

!!! warning

    Arrays allocated with StackAllocator are not managed by Julia runtime.
    User is responsible for ensuring correct lifetimes, i.e., that the reference to allocator
    outlives all arrays allocated using this allocator.
"""
function allocate(sa::StackAllocator, T::DataType, dims, align::Integer=sizeof(T))
    nbytes  = prod(dims) * sizeof(T)
    aligned = div(sa.offset + align - 1, align) * align
    # reallocate buffer if allocation size is larger than buffer size
    if aligned + nbytes > length(sa.buffer)
        error("not enough memory to allocate")
    end
    # get a slice of the buffer
    backend = KernelAbstractions.get_backend(sa.buffer)
    data_ptr = convert(pointertype(backend, T), pointer(sa.buffer) + aligned)
    sa.offset = aligned + nbytes
    sa.nallocs += 1
    return unsafe_wrap(backend, data_ptr, dims)
end
