# Exact correspondence between finite MPO automata and TEBD bond densities.

@testset "Nearest-neighbour MPO/TEBD bridge" begin
    N = 4
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    identity2 = Matrix{ComplexF64}(I, 2, 2)
    onsite = [
        (0.07site - 0.19) .* X .+ (0.11 - 0.03site) .* Z
        for site in 1:N
    ]
    couplings = ComplexF64[
        0.31 -0.27 0.14
        -0.09 0.22 -0.17
    ]
    left_ops = [X, Y]
    right_ops = [Z, X]

    mpo = nearest_neighbor_mpo(
        N, onsite, left_ops, right_ops; couplings=couplings,
    )
    densities = nearest_neighbor_bond_hamiltonians(
        N, onsite, left_ops, right_ops; couplings=couplings,
    )
    @test length(densities) == N - 1
    @test all(size(h) == (2, 2, 2, 2) for h in densities)

    dense_from_bonds = zeros(ComplexF64, 2^N, 2^N)
    for bond in 1:(N - 1)
        local_matrix = reshape(
            permutedims(densities[bond], (2, 1, 4, 3)), 4, 4,
        )
        operators = Matrix{ComplexF64}[]
        append!(operators, [identity2 for _ in 1:(bond - 1)])
        push!(operators, local_matrix)
        append!(operators, [identity2 for _ in (bond + 2):N])
        dense_from_bonds .+= reduce(kron, operators)
    end
    @test dense_from_bonds ≈ mpo_matrix(mpo) atol=2.0e-12 rtol=2.0e-12

    state = random_mps(
        N, 2; maxdim=4, rng=MersenneTwister(0xb01d), T=ComplexF64,
    )
    normalize!(state)
    initial = dense_state(state)
    dt = 1.0e-3
    tebd_step!(state, densities, dt; order=2, maxdim=16)
    exact = exp(-im * dt * mpo_matrix(mpo)) * initial
    @test phase_aligned_error(dense_state(state), exact) <= 2.0e-9

    @test_throws ArgumentError nearest_neighbor_bond_hamiltonians(
        1, onsite[1], left_ops, right_ops; couplings=[1, 1],
    )
end
