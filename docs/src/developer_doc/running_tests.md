# Running Tests

## CPU tests

To run the Chmy test suite on the CPU, simple run `test` from within the package mode or using `Pkg`:
```julia-repl
julia> using Pkg

julia> Pkg.test("Chmy")
```

## GPU tests

To run the Chmy test suite on CUDA, ROC or Metal backend (Nvidia, AMD or Apple GPUs), respectively, run the tests using `Pkg` adding following `test_args`:

### For CUDA backend (Nvidia GPUs):

```julia-repl
julia> using Pkg

julia> Pkg.test("Chmy"; test_args=["--backend=CUDA"])
```

### For ROC backend (AMD GPUs):

```julia-repl
julia> using Pkg

julia> Pkg.test("Chmy"; test_args=["--backend=AMDGPU"])
```

### For Metal backend (Apple GPUs):

```julia-repl
julia> using Pkg

julia> Pkg.test("Chmy"; test_args=["--backends=Metal"])
```