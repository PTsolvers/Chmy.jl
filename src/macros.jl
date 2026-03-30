"""
    @scalars x y ...

Declare symbolic scalar fields.

```julia
@scalars a b
@scalars a @uniform(b, c)
```
"""
macro scalars(args...)
    return expand_declaration(Symbol("@scalars"), args)
end

"""
    @vectors u v ...

Declare symbolic vector fields.

```julia
@vectors u v
@uniform @vectors u v
```
"""
macro vectors(args...)
    return expand_declaration(Symbol("@vectors"), args)
end

"""
    @tensors R x y ...

Declare symbolic tensors of rank `R`.

Plain names create generic tensors. Qualifiers such as [`@sym`](@ref),
[`@diag`](@ref), and [`@alt`](@ref) select specialized tensor kinds.

```julia
@tensors 2 A @sym(B, C) @diag(D)
@uniform @tensors 2 @sym(U)
```
"""
macro tensors(args...)
    return expand_declaration(Symbol("@tensors"), args)
end

"""
    @uniform decl
    @uniform begin ... end

Mark declarations as spatially uniform.

`@uniform` can wrap a declaration macro, appear inline inside another declaration,
or wrap a `begin ... end` block containing declaration macros.

```julia
@uniform @scalars a b
@scalars a @uniform(b, c)
@uniform begin
    @scalars a b
    @vectors u v
end
```
"""
macro uniform(args...)
    if length(args) == 1 && args[1] isa Expr && args[1].head === :macrocall && args[1].args[1] in DECLARATION_MACROS
        expr = args[1]
        return expand_declaration(expr.args[1], macro_args(expr); uniform=true)
    end
    if length(args) == 1 && args[1] isa Expr && args[1].head === :block
        return expand_uniform_block(args[1])
    end
    declaration_error("`@uniform` must wrap a declaration macro like `@uniform @scalars x y` or appear inline as `@scalars @uniform(x, y)`.")
end

"""
    @sym x
    @sym(x, y, ...)

Mark tensors as symmetric.

This qualifier is only valid inside [`@tensors`](@ref).

```julia
@tensors 2 @sym(A, B)
@uniform @tensors 2 @sym(C)
```
"""
macro sym(args...)
    declaration_error("`@sym` can only be used inside `@tensors`.")
end

"""
    @diag x
    @diag(x, y, ...)

Mark tensors as diagonal.

This qualifier is only valid inside [`@tensors`](@ref).

```julia
@tensors 2 @diag(D)
@tensors 2 A @uniform(@diag(E))
```
"""
macro diag(args...)
    declaration_error("`@diag` can only be used inside `@tensors`.")
end

"""
    @alt x
    @alt(x, y, ...)

Mark tensors as alternating.

This qualifier is only valid inside [`@tensors`](@ref).

```julia
@tensors 2 @alt(W, Z)
@uniform @tensors 2 @alt(A)
```
"""
macro alt(args...)
    declaration_error("`@alt` can only be used inside `@tensors`.")
end

struct Declaration
    name::Symbol
    rank::Int
    kind::Symbol
    uniform::Bool
end

const DECLARATION_MACROS = (Symbol("@scalars"), Symbol("@vectors"), Symbol("@tensors"))
const KIND_MACROS = Dict(Symbol("@sym") => :SymKind, Symbol("@diag") => :DiagKind, Symbol("@alt") => :AltKind)

const STENSOR_REF = GlobalRef(@__MODULE__, :STensor)

is_macrocall(expr, name::Symbol) = expr isa Expr && expr.head === :macrocall && expr.args[1] === name
macro_args(expr::Expr) = expr.args[3:end]

declaration_error(message) = throw(ArgumentError(message))

function parse_rank(rank)
    rank isa Integer || declaration_error("`@tensors` rank must be a nonnegative integer literal, got `$rank`.")
    rank >= 0 || declaration_error("`@tensors` rank must be a nonnegative integer literal, got `$rank`.")
    return Int(rank)
end

function parse_annotation(expr, rank, uniform)
    args = macro_args(expr)
    macro_name = expr.args[1]
    isempty(args) && declaration_error("`$macro_name(...)` requires at least one name.")
    kind = KIND_MACROS[macro_name]
    return map(args) do arg
        arg isa Symbol || declaration_error("`$macro_name(...)` requires a single name, got `$arg`.")
        Declaration(arg, rank, kind, uniform)
    end
end

function parse_declaration_item!(decls, arg, rank, allow_kinds, uniform)
    if arg isa Symbol
        push!(decls, Declaration(arg, rank, :NoKind, uniform))
        return
    end

    # `@uniform(...)` only changes the flag for the wrapped declarations.
    if is_macrocall(arg, Symbol("@uniform"))
        uniform = true
        inner_args = macro_args(arg)
        isempty(inner_args) && declaration_error("`@uniform(...)` requires at least one name.")
        for inner_arg in inner_args
            parse_declaration_item!(decls, inner_arg, rank, allow_kinds, uniform)
        end
        return
    end

    if arg isa Expr && arg.head === :macrocall && haskey(KIND_MACROS, arg.args[1])
        allow_kinds || declaration_error("tensor kind qualifiers like `$(arg.args[1])(...)` are only valid inside `@tensors`.")
        append!(decls, parse_annotation(arg, rank, uniform))
        return
    end

    declaration_error("declaration targets must be bare symbols, got `$arg`.")
end

function parse_declarations(args, rank, allow_kinds, uniform)
    decls = Declaration[]
    for arg in args
        parse_declaration_item!(decls, arg, rank, allow_kinds, uniform)
    end
    return decls
end

kind_ref(kind::Symbol) = GlobalRef(@__MODULE__, kind)

function constructor_expr(decl::Declaration)
    return Expr(:call, Expr(:curly, STENSOR_REF, decl.rank, kind_ref(decl.kind), decl.uniform, QuoteNode(decl.name)))
end

function emit_declarations(decls::Vector{Declaration})
    # Emit plain assignments so the declared names bind in caller scope.
    assignments = map(decls) do decl
        Expr(:(=), esc(decl.name), constructor_expr(decl))
    end
    return Expr(:block, assignments...)
end

function expand_declaration(macro_name, args; uniform=false)
    if macro_name === Symbol("@scalars")
        isempty(args) && declaration_error("`@scalars` requires at least one name.")
        decls = parse_declarations(args, 0, false, uniform)
        return emit_declarations(decls)
    elseif macro_name === Symbol("@vectors")
        isempty(args) && declaration_error("`@vectors` requires at least one name.")
        decls = parse_declarations(args, 1, false, uniform)
        return emit_declarations(decls)
    elseif macro_name === Symbol("@tensors")
        isempty(args) && declaration_error("`@tensors` requires a rank followed by at least one name.")
        length(args) == 1 && declaration_error("`@tensors` requires at least one name after the tensor rank.")
        rank = parse_rank(first(args))
        decls = parse_declarations(args[2:end], rank, true, uniform)
        return emit_declarations(decls)
    end

    declaration_error("Unsupported declaration macro `$macro_name`.")
end

function expand_uniform_block(expr)
    expr isa Expr && expr.head === :block || declaration_error("`@uniform` blocks must contain only declaration macros.")
    items = Any[]
    # Preserve line nodes so error locations still point back to user code.
    for item in expr.args
        if item isa LineNumberNode
            push!(items, item)
        elseif item isa Expr && item.head === :macrocall && item.args[1] in DECLARATION_MACROS
            push!(items, expand_declaration(item.args[1], macro_args(item); uniform=true))
        elseif item isa Expr && item.head === :block
            push!(items, expand_uniform_block(item))
        else
            declaration_error("`@uniform begin ... end` may only contain declaration macros.")
        end
    end
    return Expr(:block, items...)
end
