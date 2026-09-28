using Test
using Chmy

@testset "compilation" begin
    @scalars a
    @tensors 2 A B

    sampled = lower((sin.(A .+ a) .* B .^ 2)[1, 2], (𝑖, 𝑗))
    binding = Binding(A[1, 2] => fill(0.2, 2, 2), a => fill(0.3, 2, 2), B[1, 2] => fill(2.0, 2, 2))
    f = compile(sampled, binding)
    @test (@inferred f(binding.data, (1, 2))) ≈ sin(0.5) * 4
end
