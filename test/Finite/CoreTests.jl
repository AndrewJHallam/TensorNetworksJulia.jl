# Finite MPS/MPO primitives and DMRG tests.

@testset "TensorNetworksJulia" begin
    @testset "MPS construction, overlap, and normalization" begin
        N = 5
        d = 2
        product = product_mps(N, d, 1)
        @test length(product) == N
        @test all(ndims(A) == 3 for A in product)
        @test all(size(A) == (1, 1, d) for A in product)
        @test overlap(product, product) ≈ 1

        zero = ComplexF64[1, 0]
        one = ComplexF64[0, 1]
        plus = ComplexF64[1, 1] / sqrt(2)
        local_product = product_mps([plus, zero, one])
        @test dense_state(local_product) ≈ reduce(kron, [plus, zero, one])

        psi = random_mps(
            N,
            d;
            maxdim = 3,
            rng = MersenneTwister(0x51a7),
            T = ComplexF64,
        )
        phi = random_mps(
            N,
            d;
            maxdim = 3,
            rng = MersenneTwister(0x7eed),
            T = ComplexF64,
        )
        @test length(psi) == N
        @test size(psi[1], 1) == 1
        @test size(psi[end], 2) == 1
        @test all(size(A, 3) == d for A in psi)
        @test all(size(A, 1) <= 3 && size(A, 2) <= 3 for A in psi)
        @test all(
            size(psi[site], 2) == size(psi[site + 1], 1) for
            site in 1:(N - 1)
        )
        @test overlap(psi, phi) ≈ dot(dense_state(psi), dense_state(phi))

        psi_scaled = deepcopy(psi)
        psi_scaled[1] .*= 3 - 4im
        returned = normalize!(psi_scaled)
        @test returned === psi_scaled
        @test overlap(psi_scaled, psi_scaled) ≈ 1 atol = TEST_ATOL
        @test norm(dense_state(psi_scaled)) ≈ 1 atol = TEST_ATOL
    end

    @testset "Two-site contraction and QR/SVD factorizations" begin
        rng = MersenneTwister(0xcafe)
        A = randn(rng, ComplexF64, 2, 3, 2)
        B = randn(rng, ComplexF64, 3, 4, 2)
        AA = contract_two_sites(A, B)
        @test size(AA) == (2, 4, 2, 2)

        AA_reference = zeros(ComplexF64, 2, 4, 2, 2)
        for left in 1:2, right in 1:4, p1 in 1:2, p2 in 1:2
            AA_reference[left, right, p1, p2] =
                sum(A[left, middle, p1] * B[middle, right, p2] for middle in 1:3)
        end
        @test AA ≈ AA_reference atol = TEST_ATOL

        tensor = randn(rng, ComplexF64, 3, 4, 2)
        C_left, AL = left_orthogonalize(tensor)
        @test absorb_right(AL, C_left) ≈ tensor atol = TEST_ATOL
        @test left_canonical_gram(AL) ≈
              Matrix{ComplexF64}(I, size(AL, 2), size(AL, 2)) atol = TEST_ATOL

        C_right, AR = right_orthogonalize(tensor)
        @test absorb_left(C_right, AR) ≈ tensor atol = TEST_ATOL
        @test right_canonical_gram(AR) ≈
              Matrix{ComplexF64}(I, size(AR, 1), size(AR, 1)) atol = TEST_ATOL

        AL_split, AC_split, discarded_left =
            split_left(AA; maxdim = 32, cutoff = 0.0)
        @test contract_two_sites(AL_split, AC_split) ≈ AA atol = TEST_ATOL
        @test left_canonical_gram(AL_split) ≈
              Matrix{ComplexF64}(I, size(AL_split, 2), size(AL_split, 2)) atol = TEST_ATOL
        @test discarded_left >= -eps(Float64)

        AC_right, AR_split, discarded_right =
            split_right(AA; maxdim = 32, cutoff = 0.0)
        @test contract_two_sites(AC_right, AR_split) ≈ AA atol = TEST_ATOL
        @test right_canonical_gram(AR_split) ≈
              Matrix{ComplexF64}(I, size(AR_split, 1), size(AR_split, 1)) atol = TEST_ATOL
        @test discarded_right >= -eps(Float64)

        AL_truncated, AC_truncated, discarded =
            split_left(AA; maxdim = 2, cutoff = 0.0)
        @test size(AL_truncated, 2) <= 2
        @test size(AC_truncated, 1) == size(AL_truncated, 2)
        @test discarded >= -eps(Float64)
    end

    @testset "Ising MPO agrees with an independent dense Hamiltonian" begin
        N = 3
        parameters = (
            J = -0.73,
            hx = 0.41,
            hy = -0.17,
            hz = 0.23,
            shift = 0.09,
        )
        for interaction in (:x, :z)
            mpo = ising_mpo(N; parameters..., interaction = interaction)
            @test length(mpo) == N
            @test all(ndims(W) == 4 for W in mpo)
            @test all(size(W, 1) == 2 && size(W, 2) == 2 for W in mpo)
            @test size(mpo[1], 3) == 1
            @test size(mpo[end], 4) == 1
            @test all(
                size(mpo[site], 4) == size(mpo[site + 1], 3) for
                site in 1:(N - 1)
            )

            dense_mpo = mpo_matrix(mpo)
            dense_reference = exact_open_ising(
                N;
                parameters...,
                interaction = interaction,
            )
            @test size(dense_mpo) == (2^N, 2^N)
            @test dense_mpo ≈ dense_reference atol = TEST_ATOL rtol = TEST_ATOL
            @test dense_mpo ≈ dense_mpo' atol = TEST_ATOL
        end
    end

    @testset "Environments, expectation values, and effective actions" begin
        N = 4
        parameters = (J = -0.8, hx = 0.37, hy = 0.11, hz = -0.19)
        mpo = ising_mpo(N; parameters..., interaction = :z)
        psi = random_mps(
            N,
            2;
            maxdim = 3,
            rng = MersenneTwister(0xe11),
            T = ComplexF64,
        )
        normalize!(psi)

        left = left_environments(psi, mpo)
        right = right_environments(psi, mpo)
        @test length(left) == N + 1
        @test length(right) == N + 1
        @test size(left[1]) == (1, 1, 1)
        @test size(right[end]) == (1, 1, 1)
        for site in 1:N
            @test size(left[site + 1]) ==
                  (size(psi[site], 2), size(mpo[site], 4), size(psi[site], 2))
            @test size(right[site]) ==
                  (size(psi[site], 1), size(mpo[site], 3), size(psi[site], 1))
            @test left_environment(left[site], psi[site], mpo[site]) ≈
                  left[site + 1] atol = TEST_ATOL
            @test right_environment(psi[site], mpo[site], right[site + 1]) ≈
                  right[site] atol = TEST_ATOL
        end

        state = dense_state(psi)
        dense_hamiltonian = mpo_matrix(mpo)
        energy_reference = dot(state, dense_hamiltonian * state)
        energy_mps = expectation(psi, mpo)
        @test energy_mps ≈ energy_reference atol = 2.0e-10 rtol = 2.0e-10
        @test left[end][1, 1, 1] ≈ energy_mps atol = TEST_ATOL
        @test right[1][1, 1, 1] ≈ energy_mps atol = TEST_ATOL

        site = 2
        HA = apply_heff1(psi[site], left[site], mpo[site], right[site + 1])
        @test size(HA) == size(psi[site])
        @test dot(vec(psi[site]), vec(HA)) ≈ energy_mps atol = 2.0e-10

        AA = contract_two_sites(psi[site], psi[site + 1])
        HAA = apply_heff2(
            AA,
            left[site],
            mpo[site],
            mpo[site + 1],
            right[site + 2],
        )
        @test size(HAA) == size(AA)
        @test dot(vec(AA), vec(HAA)) ≈ energy_mps atol = 2.0e-10

        product = product_mps(N, 2, 1)
        product_energy = expectation(product, mpo)
        @test product_energy ≈ parameters.J * (N - 1) + parameters.hz * N atol = TEST_ATOL
    end

    @testset "N=4 DMRG agrees with exact diagonalization" begin
        N = 4
        parameters = (J = -1.0, hx = -0.71, hy = 0.0, hz = 0.17)
        mpo = ising_mpo(N; parameters..., interaction = :z)
        dense_hamiltonian = mpo_matrix(mpo)
        exact_energy = minimum(eigvals(Hermitian(dense_hamiltonian)))

        psi = random_mps(
            N,
            2;
            maxdim = 2,
            rng = MersenneTwister(0xd2),
            T = ComplexF64,
        )
        normalize!(psi)
        energies = Float64[real(expectation(psi, mpo))]
        for _ in 1:4
            diagnostic = dmrg2_sweep!(psi, mpo; maxdim = 8, cutoff = 0.0)
            current_energy = real(expectation(psi, mpo))
            push!(energies, current_energy)
            @test diagnostic.energy ≈ current_energy atol = 2.0e-8 rtol = 2.0e-8
            @test !isempty(diagnostic.local_energies)
            @test length(diagnostic.local_energies) == length(diagnostic.residuals)
            @test all(isfinite, diagnostic.local_energies)
            @test all(isfinite, diagnostic.residuals)
            @test diagnostic.discarded_weight >= -eps(Float64)
            @test diagnostic.converged isa Bool
        end
        @test nonincreasing(energies)
        @test energies[end] ≈ exact_energy atol = 1.0e-6 rtol = 1.0e-7
        @test overlap(psi, psi) ≈ 1 atol = 2.0e-9

        energy_before_one_site = real(expectation(psi, mpo))
        one_site = dmrg1_sweep!(psi, mpo)
        energy_after_one_site = real(expectation(psi, mpo))
        @test energy_after_one_site <= energy_before_one_site + 1.0e-8
        @test one_site.energy ≈ energy_after_one_site atol = 2.0e-8 rtol = 2.0e-8
        @test !isempty(one_site.local_energies)
        @test length(one_site.local_energies) == length(one_site.residuals)
        @test all(isfinite, one_site.residuals)
        @test one_site.converged isa Bool
        @test energy_after_one_site ≈ exact_energy atol = 1.0e-6 rtol = 1.0e-7

        psi_driver = random_mps(
            N,
            2;
            maxdim = 2,
            rng = MersenneTwister(0xd0),
            T = ComplexF64,
        )
        normalize!(psi_driver)
        driver_initial_energy = real(expectation(psi_driver, mpo))
        result = dmrg!(
            psi_driver,
            mpo;
            nsweeps = 6,
            algorithm = :two_site,
            maxdim = 8,
            cutoff = 0.0,
            eig_tol = 1.0e-11,
            energy_tol = 1.0e-9,
        )
        @test result.sweeps == length(result.history)
        @test 1 <= result.sweeps <= 6
        @test result.converged isa Bool
        @test result.energy ≈ real(expectation(psi_driver, mpo)) atol = 2.0e-8
        @test result.energy ≈ exact_energy atol = 1.0e-6 rtol = 1.0e-7
        history_energies = vcat(
            driver_initial_energy,
            [real(step.energy) for step in result.history],
        )
        @test nonincreasing(history_energies)
    end
end
