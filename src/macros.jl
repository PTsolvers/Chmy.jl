using MacroTools

# for a function with a last parameter being Vararg{Integer}, create additional definition for a CartesianIndex
macro add_cartesian(ex)
    add_cartesian_impl(ex)
end

function add_cartesian_impl(ex)
    func_def = splitdef(ex)
    lastarg = last(func_def[:args])
    @capture(lastarg, arg_::Vararg{Integer,N_}) || error("last argument should be Vararg")

    cart_def = deepcopy(func_def)
    cart_def[:args][end] = :($arg::CartesianIndex{$N})

    call_args = convert(Vector{Any}, map(first ∘ splitarg, func_def[:args]))
    call_args[end] = :(Tuple($(call_args[end]))...)

    cart_def[:body] = quote
        $(func_def[:name])($(call_args...))
    end

    cart_ex = combinedef(cart_def) |> shortdef

    quote
        Base.@__doc__ Base.@propagate_inbounds $ex
        Base.@propagate_inbounds $cart_ex
    end |> esc
end
