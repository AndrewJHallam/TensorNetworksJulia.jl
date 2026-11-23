# Finite MPS compression and MPO-application tests.

function _compression_dense_state(psi)
    dimensions = [size(A, 3) for A in psi]
    total_dimension = prod(dimensions)
    state = zeros(ComplexF64, total_dimension)
    configuration = Vector{Int}(undef, length(psi))

    for basis_index in 0:(total_dimension - 1)
        remainder = basis_index
        for site in length(psi):-1:1
            configuration[site] = mod(remainder, dimensions[site]) + 1
            remainder = div(remainder, dimensions[site])
        end

        bond = ComplexF64[1]
        for site in eachindex(psi)
            A = psi[site]
            next_bond = zeros(ComplexF64, size(A, 2))
            for left in axes(A, 1), right in axes(A, 2)
                next_bond[right] +=
                    bond[left] * A[left, right, configuration[site]]
            end
            bond = next_bond
        end
        state[basis_index + 1] = only(bond)
    end
    return state
end

function _compression_gauge_transform(mps, bond, gauge)
    transformed = deepcopy(mps)
    size(gauge, 1) == size(gauge, 2) == size(mps[bond], 2) ||
        throw(DimensionMismatch("invalid test gauge"))
    for physical in axes(transformed[bond], 3)
        transformed[bond][:, :, physical] .=
            transformed[bond][:, :, physical] * gauge
    end
    for physical in axes(transformed[bond + 1], 3)
        transformed[bond + 1][:, :, physical] .=
            gauge \ transformed[bond + 1][:, :, physical]
    end
    return transformed
end

function _compression_product_mpo(local_maps)
    T = promote_type(map(eltype, local_maps)...)
    return [
        Array(reshape(T.(operator), size(operator, 1), size(operator, 2), 1, 1))
        for operator in local_maps
    ]
end

function _compression_fidelity(left, right)
    denominator = real(dot(left, left) * dot(right, right))
    iszero(denominator) && return 0.0
    return abs2(dot(left, right)) / denominator
end

@testset "Finite MPS compression and MPO application" begin
    @testset "Exact arbitrary-gauge compression preserves scale" begin
        rng = MersenneTwister(0xc04e)
        psi = random_mps(5, 2; maxdim=4, rng=rng, T=ComplexF64)
        psi[1] .*= 2.4 - 0.7im
        dimension = size(psi[2], 2)
        Q = Matrix(qr(randn(rng, ComplexF64, dimension, dimension)).Q)
        gauge = Q * Diagonal(range(0.45, 1.9; length=dimension))
        psi = _compression_gauge_transform(psi, 2, gauge)
        state_before = _compression_dense_state(psi)
        norm_before = real(dot(state_before, state_before))

        result = compress!(psi; maxdim=64)
        state_after = _compression_dense_state(psi)
        @test result.state === psi
        @test state_after ≈ state_before atol=2e-10 rtol=2e-10
        @test result.norm_squared_before ≈ norm_before atol=2e-10
        @test result.norm_squared_after_truncation ≈ norm_before atol=2e-10
        @test result.norm_squared_after ≈ norm_before atol=2e-10
        @test result.discarded_weight ≈ 0 atol=2e-13
        @test result.max_discarded_weight ≈ 0 atol=2e-13
        @test length(result.discarded_weights) == length(psi) - 1
        @test result.bond_dimensions == [size(A, 2) for A in psi[1:end-1]]
        @test !result.normalized

        single = product_mps([ComplexF64[2, -im]]; normalize=false)
        single_before = _compression_dense_state(single)
        single_result = compress!(single; maxdim=1)
        @test _compression_dense_state(single) ≈ single_before atol=5e-13
        @test isempty(single_result.discarded_weights)
        @test isempty(single_result.bond_dimensions)

        single_normalized = product_mps([ComplexF64[2, -im]]; normalize=false)
        normalized_result = compress!(single_normalized; maxdim=1, normalize=true)
        @test norm(_compression_dense_state(single_normalized)) ≈ 1 atol=5e-13
        @test normalized_result.norm_squared_before ≈ 5 atol=5e-13
        @test normalized_result.norm_squared_after ≈ 1 atol=5e-13
        @test normalized_result.normalized
    end

    @testset "Hard and discarded-weight truncation" begin
        rng = MersenneTwister(0x7a0c)
        psi = random_mps(6, 2; maxdim=8, rng=rng, T=ComplexF64)
        state_before = _compression_dense_state(psi)
        gauge_dimension = size(psi[3], 2)
        gauge_Q = Matrix(qr(randn(
            rng, ComplexF64, gauge_dimension, gauge_dimension,
        )).Q)
        gauged = _compression_gauge_transform(
            psi,
            3,
            gauge_Q * Diagonal(range(0.5, 1.9; length=gauge_dimension)),
        )
        result = compress!(psi; maxdim=2)
        state_after = _compression_dense_state(psi)
        gauged_result = compress!(gauged; maxdim=2)
        gauged_state_after = _compression_dense_state(gauged)

        @test all(<=(2), result.bond_dimensions)
        @test all(weight -> 0 <= weight <= 1, result.discarded_weights)
        @test 0 <= result.discarded_weight <= 1
        @test result.max_discarded_weight == maximum(result.discarded_weights)
        observed_loss = 1 - result.norm_squared_after_truncation /
                            result.norm_squared_before
        @test result.discarded_weight ≈ observed_loss atol=2e-10 rtol=2e-10
        @test _compression_fidelity(state_before, state_after) > 0.1
        @test gauged_state_after ≈ state_after atol=3e-10 rtol=3e-10
        @test gauged_result.discarded_weights ≈
              result.discarded_weights atol=3e-10 rtol=3e-10

        probabilities = [0.90, 0.09, 0.01]
        known = [zeros(ComplexF64, 1, 3, 3),
                 zeros(ComplexF64, 3, 1, 3)]
        for index in 1:3
            known[1][1, index, index] = sqrt(probabilities[index])
            known[2][index, 1, index] = 1
        end
        cutoff_result = compress!(known; maxdim=3, cutoff=0.011)
        @test cutoff_result.bond_dimensions == [2]
        @test cutoff_result.discarded_weights ≈ [0.01] atol=2e-13
        @test cutoff_result.discarded_weight ≈ 0.01 atol=2e-13
        @test cutoff_result.norm_squared_after_truncation ≈ 0.99 atol=2e-13

        known_normalized = [zeros(ComplexF64, 1, 3, 3),
                            zeros(ComplexF64, 3, 1, 3)]
        for index in 1:3
            known_normalized[1][1, index, index] = sqrt(probabilities[index])
            known_normalized[2][index, 1, index] = 1
        end
        normalized_result = compress!(
            known_normalized; maxdim=2, normalize=true,
        )
        @test normalized_result.discarded_weight ≈ 0.01 atol=2e-13
        @test normalized_result.norm_squared_after_truncation ≈ 0.99 atol=2e-13
        @test normalized_result.norm_squared_after ≈ 1 atol=2e-13
    end

    @testset "Compression validation and zero states" begin
        psi = product_mps(3, 2, 1; T=ComplexF64)
        @test_throws ArgumentError compress!(deepcopy(psi); maxdim=0)
        @test_throws ArgumentError compress!(deepcopy(psi); maxdim=2, cutoff=-1e-4)
        @test_throws ArgumentError compress!(deepcopy(psi); maxdim=2, cutoff=NaN)
        @test_throws DimensionMismatch compress!(
            [zeros(ComplexF64, 2, 1, 2)]; maxdim=1,
        )

        zero_state = product_mps(3, 2, 1; T=ComplexF64)
        fill!(zero_state[1], 0)
        result = compress!(zero_state; maxdim=1)
        @test result.norm_squared_before == 0
        @test result.norm_squared_after == 0
        @test result.discarded_weight == 0
        @test iszero(norm(_compression_dense_state(zero_state)))

        zero_for_normalization = product_mps(3, 2, 1; T=ComplexF64)
        fill!(zero_for_normalization[1], 0)
        tensors_before = deepcopy(zero_for_normalization)
        @test_throws ArgumentError compress!(
            zero_for_normalization; maxdim=1, normalize=true,
        )
        @test all(zero_for_normalization[n] == tensors_before[n]
                  for n in eachindex(zero_for_normalization))
    end

    @testset "MPO application agrees with dense contraction" begin
        rng = MersenneTwister(0xa991)
        N = 4
        psi = random_mps(N, 2; maxdim=3, rng=rng, T=ComplexF64)
        psi[1] .*= -0.8 + 1.3im
        psi_before = deepcopy(psi)
        state = _compression_dense_state(psi)
        mpo = ising_mpo(
            N; J=-0.73, hx=0.21, hy=-0.17, hz=0.09, interaction=:z,
        )
        reference = mpo_matrix(mpo) * state

        result = apply_mpo(mpo, psi)
        actual = _compression_dense_state(result.state)
        @test actual ≈ reference atol=3e-10 rtol=3e-10
        @test _compression_dense_state(psi) ≈ state atol=5e-13
        @test all(psi[n] == psi_before[n] for n in eachindex(psi))
        @test result.norm_squared_before ≈ real(dot(reference, reference)) atol=3e-10
        @test result.norm_squared_after ≈ real(dot(reference, reference)) atol=3e-10
        @test result.input_norm_squared ≈ real(dot(state, state)) atol=2e-10
        @test result.enlarged_bond_dimensions ==
              [size(psi[n], 2) * size(mpo[n], 4) for n in 1:(N - 1)]

        normalized = apply_mpo(mpo, psi; normalize=true)
        normalized_state = _compression_dense_state(normalized.state)
        @test norm(normalized_state) ≈ 1 atol=2e-10
        @test _compression_fidelity(normalized_state, reference) ≈ 1 atol=2e-10

        truncated = apply_mpo(mpo, psi; maxdim=2)
        truncated_state = _compression_dense_state(truncated.state)
        @test all(<=(2), truncated.bond_dimensions)
        @test _compression_fidelity(truncated_state, reference) > 0.1
        @test truncated.discarded_weight ≈
              1 - truncated.norm_squared_after_truncation /
                  truncated.norm_squared_before atol=3e-10
    end

    @testset "Rectangular and heterogeneous local maps" begin
        local_states = [ComplexF64[1, 2im], ComplexF64[1, -1, 0.5im]]
        psi = product_mps(local_states; normalize=false)
        psi_before = deepcopy(psi)
        map1 = ComplexF64[1 0; 0 im; 1 -1]
        map2 = ComplexF64[1 0 2; 0.5im 1 -1]
        mpo = _compression_product_mpo([map1, map2])
        reference = kron(map1, map2) * _compression_dense_state(psi)

        result = apply_mpo(mpo, psi)
        @test [size(A, 3) for A in result.state] == [3, 2]
        @test _compression_dense_state(result.state) ≈ reference atol=5e-12
        @test all(psi[n] == psi_before[n] for n in eachindex(psi))

        single = product_mps([ComplexF64[2, -im]]; normalize=false)
        single_map = ComplexF64[1 2; 0 im; -1 0.5]
        single_result = apply_mpo(
            _compression_product_mpo([single_map]), single,
        )
        @test size(only(single_result.state), 3) == 3
        @test _compression_dense_state(single_result.state) ≈
              single_map * _compression_dense_state(single) atol=5e-13
    end

    @testset "MPO application validation and zero output" begin
        psi = product_mps(2, 2, 1; T=ComplexF64)
        identity2 = ComplexF64[1 0; 0 1]
        identity_mpo = _compression_product_mpo([identity2, identity2])

        @test_throws ArgumentError apply_mpo(identity_mpo, psi; maxdim=0)
        @test_throws ArgumentError apply_mpo(identity_mpo, psi; cutoff=-1)
        @test_throws DimensionMismatch apply_mpo(identity_mpo[1:1], psi)
        wrong_input = _compression_product_mpo([
            ones(ComplexF64, 2, 3), identity2,
        ])
        @test_throws DimensionMismatch apply_mpo(wrong_input, psi)
        broken_bonds = [zeros(ComplexF64, 2, 2, 1, 2),
                        zeros(ComplexF64, 2, 2, 3, 1)]
        @test_throws DimensionMismatch apply_mpo(broken_bonds, psi)

        zero_mpo = _compression_product_mpo([
            zeros(ComplexF64, 2, 2), zeros(ComplexF64, 2, 2),
        ])
        zero_result = apply_mpo(zero_mpo, psi)
        @test iszero(norm(_compression_dense_state(zero_result.state)))
        @test zero_result.norm_squared_before == 0
        @test_throws ArgumentError apply_mpo(zero_mpo, psi; normalize=true)
    end
end
