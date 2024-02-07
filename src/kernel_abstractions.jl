Base.unsafe_wrap(::CPU, ptr::Ptr, dims) = unsafe_wrap(Array, ptr, dims)

pointertype(::CPU, T::DataType) = Ptr{T}
