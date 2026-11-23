# Finite TDVP dense-reference and invariant tests.

function phase_aligned_error(actual, reference)
    overlap_value = dot(actual, reference)
    phase = iszero(overlap_value) ? one(overlap_value) :
            overlap_value / abs(overlap_value)
    return norm(phase .* actual .- reference) / norm(reference)
end

@testset "Finite TDVP" begin
    @testset "Bond effective action" begin
        rng = MersenneTwister(0xb0ad)
        FL = randn(rng, ComplexF64, 2, 3, 2)
        FR = randn(rng, ComplexF64, 4, 3, 4)
        C = randn(rng, ComplexF64, 2, 4)
        result = apply_keff(C, FL, FR)
        reference = zeros(ComplexF64, 2, 4)
        for alout in 1:2, arout in 1:4
            for alin in 1:2, arin in 1:4, w in 1:3
                reference[alout, arout] +=
                    FL[alin, w, alout] * C[alin, arin] *
                    FR[arin, w, arout]
            end
        end
        @test result ≈ reference atol = TEST_ATOL
    end

    @testset "Zero Hamiltonian is the identity up to gauge" begin
        N = 4
        zero_mpo = ising_mpo(
            N; J=0.0, hx=0.0, hy=0.0, hz=0.0, interaction=:z,
        )

        psi1 = random_mps(
            N, 2; maxdim=3, rng=MersenneTwister(0x101), T=ComplexF64,
        )
        before1 = dense_state(psi1)
        diagnostic1 = tdvp1_step!(psi1, zero_mpo, 0.13)
        after1 = dense_state(psi1)
        @test phase_aligned_error(after1, before1) <= 5.0e-11
        @test diagnostic1.krylov_converged
        @test diagnostic1.norm_after ≈ 1 atol = 5.0e-11
        @test diagnostic1.energy_after ≈ 0 atol = 5.0e-12

        psi2 = random_mps(
            N, 2; maxdim=3, rng=MersenneTwister(0x202), T=ComplexF64,
        )
        before2 = dense_state(psi2)
        diagnostic2 = tdvp2_step!(psi2, zero_mpo, 0.13; maxdim=4)
        after2 = dense_state(psi2)
        @test phase_aligned_error(after2, before2) <= 5.0e-11
        @test diagnostic2.krylov_converged
        @test diagnostic2.norm_after ≈ 1 atol = 5.0e-11
        @test diagnostic2.discarded_weight <= 5.0e-14
    end

    @testset "Real-time evolution agrees with dense exponentiation" begin
        N = 3
        dt = 2.0e-3
        parameters = (J=-0.83, hx=0.37, hy=0.12, hz=-0.21)
        mpo = ising_mpo(N; parameters..., interaction=:z)
        dense_hamiltonian = mpo_matrix(mpo)

        initial = random_mps(
            N, 2; maxdim=2, rng=MersenneTwister(0x303), T=ComplexF64,
        )
        initial_state = dense_state(initial)
        exact_state = exp(-im * dt * dense_hamiltonian) * initial_state

        psi1 = deepcopy(initial)
        diagnostic1 = tdvp1_step!(
            psi1, mpo, dt; tol=1.0e-13, maxiter=100, krylovdim=20,
        )
        @test phase_aligned_error(dense_state(psi1), exact_state) <= 2.0e-7
        @test diagnostic1.krylov_converged
        @test abs(diagnostic1.energy_after - diagnostic1.energy_before) <= 2.0e-7
        @test diagnostic1.norm_after ≈ 1 atol = 2.0e-10

        psi2 = deepcopy(initial)
        diagnostic2 = tdvp2_step!(
            psi2, mpo, dt;
            maxdim=4, cutoff=0.0, tol=1.0e-13,
            maxiter=100, krylovdim=20,
        )
        @test phase_aligned_error(dense_state(psi2), exact_state) <= 2.0e-7
        @test diagnostic2.krylov_converged
        @test abs(diagnostic2.energy_after - diagnostic2.energy_before) <= 2.0e-7
        @test diagnostic2.norm_after ≈ 1 atol = 2.0e-10
    end

    @testset "Symmetry, imaginary time, and truncation" begin
        N = 4
        parameters = (J=-1.0, hx=-0.61, hy=0.0, hz=0.18)
        mpo = ising_mpo(N; parameters..., interaction=:z)

        reversible = random_mps(
            N, 2; maxdim=4, rng=MersenneTwister(0x404), T=ComplexF64,
        )
        initial_state = dense_state(reversible)
        tdvp1_step!(reversible, mpo, 0.01; tol=1.0e-13)
        tdvp1_step!(reversible, mpo, -0.01; tol=1.0e-13)
        @test phase_aligned_error(dense_state(reversible), initial_state) <= 2.0e-7

        imaginary = random_mps(
            N, 2; maxdim=4, rng=MersenneTwister(0x505), T=ComplexF64,
        )
        energy_before = real(expectation(imaginary, mpo))
        diagnostic_imaginary = tdvp1_step!(
            imaginary, mpo, 0.01; time=:imaginary, tol=1.0e-13,
        )
        energy_after = real(expectation(imaginary, mpo))
        @test energy_after <= energy_before + 1.0e-10
        @test diagnostic_imaginary.energy_after ≈ energy_after atol = 1.0e-10

        truncated = product_mps(N, 2, 1; T=ComplexF64)
        diagnostic_truncated = tdvp2_step!(
            truncated, mpo, 0.03; maxdim=2, cutoff=0.0, tol=1.0e-13,
        )
        @test all(size(truncated[n], 2) <= 2 for n in 1:(N - 1))
        @test diagnostic_truncated.max_bond <= 2
        @test diagnostic_truncated.discarded_weight >= -eps(Float64)
        @test diagnostic_truncated.krylov_converged
        @test diagnostic_truncated.norm_after ≈ 1 atol = 2.0e-10
    end

    @testset "Real-time type safety" begin
        real_state = product_mps(3, 2, 1; T=Float64)
        mpo = ising_mpo(3; J=-1.0, hx=-0.4, interaction=:z)
        @test_throws ArgumentError tdvp1_step!(real_state, mpo, 0.1)
    end
end
