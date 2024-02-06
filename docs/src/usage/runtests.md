# Running tests

## CPU tests

To run the FastIce test suite on the CPU, simple run `test` from within the package mode or using `Pkg`:
```julia-repl
using Pkg
Pkg.test("FastIce")
```

## GPU tests - needs update!

To run the FastIce test suite on CUDA or ROC backend (Nvidia or AMD GPUs), respectively, run the tests using `Pkg` adding following `test_args`:

### For CUDA backend (Nvidia GPUs):

```julia-repl
using Pkg
Pkg.test("FastIce"; test_args=["--backend=CUDA"])
```

### For ROC backend (AMD GPUs):

```julia-repl
using Pkg
Pkg.test("FastIce"; test_args=["--backend=AMDGPU"])
```
