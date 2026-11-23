# Uniform-MPS adaptive bond-expansion tests.

@testset "Uniform-MPS bond expansion" begin
    @testset "Tangent complements and projected contraction" begin
        rng = MersenneTwister(0xb0ad)
        state = random_infinite_mps(3, 2; rng=rng, T=ComplexF64)
        AL, AR, C = state.AL, state.AR, state.C
        chi, _, d = size(AL)
        tangent_dimension = (d - 1) * chi
        VL = left_tangent_basis(AL)
        VR = right_tangent_basis(AR)
        @test size(VL) == (chi, tangent_dimension, d)
        @test size(VR) == (tangent_dimension, chi, d)

        ML = reshape(permutedims(AL, (1, 3, 2)), d * chi, chi)
        NL = reshape(permutedims(VL, (1, 3, 2)), d * chi,
                     tangent_dimension)
        MR = reshape(permutedims(AR, (1, 3, 2)), chi, d * chi)
        NR = reshape(permutedims(conj.(VR), (3, 2, 1)), d * chi,
                     tangent_dimension)
        @test ML' * NL ≈ zeros(ComplexF64, chi, tangent_dimension) atol=2e-12
        @test NL' * NL ≈ Matrix{ComplexF64}(I, tangent_dimension,
                                            tangent_dimension) atol=2e-12
        @test MR * NR ≈ zeros(ComplexF64, chi, tangent_dimension) atol=2e-12
        @test NR' * NR ≈ Matrix{ComplexF64}(I, tangent_dimension,
                                            tangent_dimension) atol=2e-12

        h = randn(rng, ComplexF64, d, d, d, d)
        Y = bond_expansion_coefficients(AL, AR, C, h; VL=VL, VR=VR)
        theta = two_site_center(AL, C, AR)
        reference = zeros(ComplexF64, tangent_dimension, tangent_dimension)
        for a in 1:tangent_dimension, b in 1:tangent_dimension,
            l in 1:chi, r in 1:chi, so in 1:d, to in 1:d,
            si in 1:d, ti in 1:d
            reference[a, b] += conj(VL[l, a, so]) * h[so, to, si, ti] *
                               theta[l, r, si, ti] * conj(VR[b, r, to])
        end
        @test Y ≈ reference atol=3e-12

        # A large identity action can leave an absolute projection roundoff
        # much larger than the user `atol`; the action-scaled machine floor
        # must still recognize it as a null tangent residual.
        scaled_identity = zeros(ComplexF64, d, d, d, d)
        for s in 1:d, t in 1:d
            scaled_identity[s, t, s, t] = 1.0e12
        end
        identity_expansion = expand_bond_dimension(
            AL, AR, C, scaled_identity; max_bond=2 * chi,
        )
        @test !identity_expansion.expanded
        @test identity_expansion.numerical_rank == 0
    end

    @testset "Analytic growth and exact embedding" begin
        AL = zeros(ComplexF64, 1, 1, 2)
        AL[1, 1, 1] = 1
        AR = copy(AL)
        C = ones(ComplexF64, 1, 1)
        hxx = ising_two_site(J=1.0, interaction=:x)

        expansion = expand_bond_dimension(AL, AR, C, hxx; max_bond=2)
        @test expansion.expanded
        @test expansion.old_bond == 1
        @test expansion.new_bond == 2
        @test expansion.added_bond == 1
        @test expansion.singular_values ≈ [1.0] atol=2e-12
        @test expansion.retained_singular_values ≈ [1.0] atol=2e-12

        residuals = infinite_residuals(
            expansion.AL, expansion.AR, expansion.C,
        )
        @test residuals.left_canonical <= 2e-12
        @test residuals.right_canonical <= 2e-12
        @test residuals.center_consistency <= 2e-12
        @test uniform_energy(expansion.AL, expansion.AR, expansion.C, hxx) ≈
              uniform_energy(AL, AR, C, hxx) atol=2e-12

        old_theta = two_site_center(AL, C, AR)
        new_theta = two_site_center(expansion.AL, expansion.C, expansion.AR)
        @test new_theta[1:1, 1:1, :, :] ≈ old_theta atol=2e-12
        @test norm(new_theta[2:2, :, :, :]) <= 2e-12
        @test norm(new_theta[:, 2:2, :, :]) <= 2e-12

        identity_h = zeros(ComplexF64, 2, 2, 2, 2)
        for s in 1:2, t in 1:2
            identity_h[s, t, s, t] = 1
        end
        unchanged = expand_bond_dimension(AL, AR, C, identity_h; max_bond=2)
        @test !unchanged.expanded
        @test unchanged.added_bond == 0
        @test unchanged.new_bond == 1

        capped = expand_bond_dimension(AL, AR, C, hxx; max_bond=1)
        @test !capped.expanded
        @test capped.new_bond == 1
        noncanonical = 2 .* AL
        @test_throws ArgumentError expand_bond_dimension(
            noncanonical, AR, C, hxx; max_bond=2,
        )
        @test_throws ArgumentError expand_bond_dimension(
            expansion.AL, expansion.AR, expansion.C, hxx; max_bond=1,
        )
    end

    @testset "Adaptive VUMPS and TDVP integration" begin
        h = ising_two_site(J=-1.0, hx=-0.7, interaction=:z)
        fixed = vumps(
            h, 1;
            rng=MersenneTwister(0xad01), maxiter=8, tol=1e-9,
            eig_tol=1e-11, env_tol=1e-11, transfer_tol=1e-11,
        )
        adaptive = vumps(
            h, 1;
            rng=MersenneTwister(0xad01), max_bond=2, maxiter=20,
            tol=1e-7, eig_tol=1e-11, env_tol=1e-11,
            transfer_tol=1e-11,
        )
        @test fixed.bond_dimension == 1
        @test adaptive.bond_dimension == 2
        @test any(event -> event.added_bond == 1, adaptive.growth_history)
        @test adaptive.energy < fixed.energy - 1e-4
        @test adaptive.converged

        evolved = infinite_tdvp_step(
            adaptive.AL, adaptive.AR, adaptive.C, h, 1e-3;
            max_bond=3, tol=1e-11, env_tol=1e-11,
            transfer_tol=1e-11,
        )
        @test evolved.expansion.expanded
        @test evolved.bond_dimension == 3
        @test evolved.residuals.left_canonical <= 2e-10
        @test evolved.residuals.right_canonical <= 2e-10
    end
end
