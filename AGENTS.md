# Chmy.jl

Chmy v0.2 is a symbolic finite-difference system: tensor-valued expressions are built symbolically, expanded to scalar components, lowered to stencils on a staggered grid, and compiled to Julia functions via RuntimeGeneratedFunctions. KernelAbstractions is used for CPU/GPU execution. `src/Chmy.jl` lists the include order and every export, which is the quickest map of the public API.

## Code style

The code must be formatted with Runic.

Inline code comments should start with a small letter, and shouldn't end with a dot.

Example of a correct inline code comment:

```julia
# fallback implementation for all functions and operators, accepting only scalar arguments
checkranks(op::Operator, args::NTuple{N,DTerm}) where {N} = checkscalar(op, args)
```

## Docstrings

Only exported functions should have docstrings. For Chmy functions, the arguments shouldn't have types defined in the method signature:

```julia
"""
    isliteral(term)

Returns `true` if `term` is a Literal.
"""
isliteral(term::DTerm) = isa_variant(term, Literal)
```

For methods from Base, the docstrings should contain the types:

```julia
"""
    ⋅(a::DTerm, b::DTerm)

A single contraction operator, which contracts the last index of `a` with the first index of `b`.
"""
⋅(a::DTerm, b::DTerm) = makecall(Fun(⋅), a, b)
```

For small getter-like functions or constructors the docstring should be short, i.e. contain no examples or subsections.

## Unit tests

Tests in the `test/` folder must only validate the expected behavior and the type stability using `@inferred` macro. Checking the number allocations and microbenchmarks should be placed to a separate folder and should not be generated unless specifically requested.

The unit test should contain minimal scaffolding. In Chmy.jl, reading unit tests is one way for a user to understand the implementation.

Tests use ParallelTestRunner, which discovers test files relative to the current directory, so run it from `test/` (running `test/runtests.jl` from the repo root also picks up every file in `src/` as a "test"):

```bash
# run the full test suite (one worker process per test file)
cd test && julia --project=. runtests.jl

# run selected test files (names are file names without `.jl`)
cd test && julia --project=. runtests.jl test_lowering test_compile

# list available tests / show runner options
cd test && julia --project=. runtests.jl --list

# equivalent through Pkg, from the repo root
julia --project -e 'using Pkg; Pkg.test(; test_args=["test_simplify"])'

# format with Runic (required); --check only reports unformatted files
runic --inplace src test
runic --check src test

# build docs (Documenter + DocumenterVitepress)
julia --project=docs docs/make.jl
```

Every file in `test/` other than `runtests.jl` is a separate test run in its own process and must be self-contained (`using Test`, `using Chmy`).
