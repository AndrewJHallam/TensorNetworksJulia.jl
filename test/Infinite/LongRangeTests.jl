# Exponential-sum long-range VUMPS and uniform-TDVP tests.

using Test
using LinearAlgebra
using Random

function _longrange_dense_map(action, prototype)
    dimension = length(prototype)
    matrix = zeros(ComplexF64, dimension, dimension)
    for column in 1:dimension
        basis = zeros(ComplexF64, dimension)
        basis[column] = 1
        matrix[:, column] .= vec(action(reshape(basis, size(prototype))))
    end
    return matrix
end

function _longrange_one_site_density(AL, C)
    AC = center_tensor(AL, C)
    d = size(AC, 3)
    density = zeros(ComplexF64, d, d)
    for sout in 1:d, sin in 1:d
        density[sout, sin] = dot(vec(AC[:, :, sout]), vec(AC[:, :, sin]))
    end
    return density / tr(density)
end

@testset "Exponential-sum long-range infinite algorithms" begin
    I2 = Matrix{ComplexF64}(I, 2, 2)
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]

    @testset "Hamiltonian validation" begin
        model = ExponentialSumHamiltonian(
            0.2 .* X,
            [X, Z],
            [Z, X],
            [0.7 -0.1; 0.3 0.4],
            [0.2 -0.3; 0.5 0.1],
        )
        @test size(model.onsite) == (2, 2)
        @test size(model.coeffs) == (2, 2)
        @test eltype(model) == ComplexF64
        @test model.left_ops[1] !== X

        @test_throws DimensionMismatch ExponentialSumHamiltonian(
            zeros(2, 3), [X], [Z], ones(1, 1), zeros(1, 1),
        )
        @test_throws DimensionMismatch ExponentialSumHamiltonian(
            I2, [X], [Z, X], ones(1, 1), zeros(1, 1),
        )
        @test_throws DimensionMismatch ExponentialSumHamiltonian(
            I2, [X], [Z], ones(1, 2), zeros(1, 1),
        )
        @test_throws DimensionMismatch ExponentialSumHamiltonian(
            I2, [zeros(3, 3)], [Z], ones(1, 1), zeros(1, 1),
        )
        @test_throws ArgumentError ExponentialSumHamiltonian(
            I2, [X], [Z], ones(1, 1), ones(1, 1),
        )
        @test_throws ArgumentError ExponentialSumHamiltonian(
            I2, [X], [Z], ones(1, 1), fill(NaN, 1, 1),
        )
    end

    @testset "Geometric tails and analytic product-state energy" begin
        AL = zeros(ComplexF64, 1, 1, 2)
        AL[1, 1, 1] = 1
        AR = copy(AL)
        C = ones(ComplexF64, 1, 1)
        coefficients = reshape([-0.7, 0.25], 1, 2)
        decays = reshape([0.4, -0.2], 1, 2)
        model = ExponentialSumHamiltonian(
            -0.3 .* Z, [Z], [Z], coefficients, decays,
        )
        context = longrange_context(AL, AR, C, model)
        expected_tail = 1 ./ (1 .- vec(decays))
        expected_energy = -0.3 + dot(vec(coefficients), expected_tail)

        @test context.OL[1, 1][1] ≈ expected_tail[1] atol=2.0e-12
        @test context.OL[1, 2][1] ≈ expected_tail[2] atol=2.0e-12
        @test context.OR[1, 1][1] ≈ expected_tail[1] atol=2.0e-12
        @test context.energy ≈ expected_energy atol=2.0e-12
        @test longrange_energy(AL, AR, C, model) ≈ expected_energy atol=2.0e-12
        @test context.energy_left ≈ context.energy_right atol=2.0e-12
        @test context.tails_converged
        @test context.environments_converged

        mpo_model = triangular_mpo(model)
        @test size(mpo_model.W) == (2, 2, 4, 4)
        @test mpo_model.lambda ≈ [1, 0.4, -0.2, 1]
        @test uniform_mpo_energy(AL, AR, C, mpo_model) ≈
              expected_energy atol=2.0e-12
    end

    @testset "Nearest-neighbour limit is the direct oracle" begin
        hx, hz, coupling = -0.31, 0.17, -0.82
        model = ExponentialSumHamiltonian(
            hx .* X .+ hz .* Z,
            [Z],
            [Z],
            reshape([coupling], 1, 1),
            zeros(1, 1),
        )
        nearest = ising_two_site(
            J=coupling, hx=hx, hz=hz, interaction=:z,
        )
        state = random_infinite_mps(
            2, 2; rng=MersenneTwister(0x1a2b), T=ComplexF64,
        )
        @test longrange_energy(state.AL, state.AR, state.C, model) ≈
              uniform_energy(state.AL, state.AR, state.C, nearest) atol=2.0e-11

        longrange = longrange_context(
            state.AL, state.AR, state.C, model;
            env_tol=1.0e-13, transfer_tol=1.0e-13,
        )
        nearest_context = TensorNetworksJulia._uniform_context(
            state.AL, state.AR, state.C, nearest;
            env_tol=1.0e-13, transfer_tol=1.0e-13,
        )
        AC = center_tensor(state.AL, state.C)
        HAC_longrange = _longrange_dense_map(
            tensor -> apply_hac_longrange(tensor, model, longrange), AC,
        )
        HAC_nearest = _longrange_dense_map(
            tensor -> apply_hac(
                tensor,
                state.AL,
                state.AR,
                nearest,
                nearest_context.HL,
                nearest_context.HR,
            ),
            AC,
        )
        HC_longrange = _longrange_dense_map(
            tensor -> apply_hc_longrange(tensor, model, longrange), state.C,
        )
        HC_nearest = _longrange_dense_map(
            tensor -> apply_hc(
                tensor,
                state.AL,
                state.AR,
                nearest,
                nearest_context.HL,
                nearest_context.HR,
            ),
            state.C,
        )
        delta_AC = HAC_longrange - HAC_nearest
        delta_C = HC_longrange - HC_nearest
        shift_AC = tr(delta_AC) / size(delta_AC, 1)
        shift_C = tr(delta_C) / size(delta_C, 1)
        @test delta_AC ≈ shift_AC .* I atol=2.0e-10
        @test delta_C ≈ shift_C .* I atol=2.0e-10
        @test shift_AC ≈ shift_C atol=2.0e-10
    end

    @testset "Asymmetric endpoint closure and Hermiticity" begin
        # X on the left endpoint and Z on the right endpoint makes the
        # accidental O_L/O_R half-chain closure swap visible.  For a random
        # complex canonical state, that swap destroys Hermiticity.
        model = ExponentialSumHamiltonian(
            0.17 .* X .- 0.23 .* Y .+ 0.11 .* Z,
            [X],
            [Z],
            reshape([0.73], 1, 1),
            reshape([0.37], 1, 1),
        )
        state = random_infinite_mps(
            2, 2; rng=MersenneTwister(0x2b3c), T=ComplexF64,
        )
        context = longrange_context(
            state.AL, state.AR, state.C, model;
            env_tol=1.0e-13, transfer_tol=1.0e-13,
        )
        AC = center_tensor(state.AL, state.C)
        HAC = _longrange_dense_map(
            tensor -> apply_hac_longrange(tensor, model, context), AC,
        )
        HC = _longrange_dense_map(
            tensor -> apply_hc_longrange(tensor, model, context), state.C,
        )

        @test HAC ≈ HAC' atol=2.0e-10
        @test HC ≈ HC' atol=2.0e-10
        @test context.energy_left ≈ context.energy_right atol=2.0e-10
        @test context.energy_mismatch <= 2.0e-10
        @test context.environment_residual <= 2.0e-10
    end

    @testset "Fixed-chi VUMPS and TDVP" begin
        model = ExponentialSumHamiltonian(
            -Z,
            [Z],
            [Z],
            reshape([-0.3], 1, 1),
            reshape([0.4], 1, 1),
        )
        ground = longrange_vumps(
            model,
            1;
            rng=MersenneTwister(0x3c4d),
            maxiter=5,
            tol=1.0e-11,
        )
        @test ground.converged
        @test ground.bond_dimension == 1
        @test ground.energy ≈ -1.5 atol=2.0e-11
        @test ground.residual <= 2.0e-11

        density_before = _longrange_one_site_density(ground.AL, ground.C)
        evolved = longrange_tdvp_step(
            ground.AL, ground.AR, ground.C, model, 0.17,
        )
        density_after = _longrange_one_site_density(evolved.AL, evolved.C)
        @test evolved.bond_dimension == 1
        @test density_after ≈ density_before atol=2.0e-11
        @test evolved.energy_before ≈ -1.5 atol=2.0e-11
        @test evolved.energy_after ≈ -1.5 atol=2.0e-11
        @test evolved.residual <= 2.0e-11
        @test evolved.krylov_converged
    end

    @testset "MPO oracle and adaptive long-range growth" begin
        comparison_model = ExponentialSumHamiltonian(
            0.13 .* X .- 0.09 .* Y .+ 0.21 .* Z,
            [X, Z],
            [Z, X],
            [0.37 -0.14; 0.22 0.31],
            [0.23 -0.17; 0.41 0.08],
        )
        state = random_infinite_mps(
            2, 2; rng=MersenneTwister(0x4d5e), T=ComplexF64,
        )
        mpo_model = triangular_mpo(comparison_model)
        @test real(longrange_energy(
            state.AL, state.AR, state.C, comparison_model,
        )) ≈ uniform_mpo_energy(
            state.AL, state.AR, state.C, mpo_model,
        ) atol=3.0e-9

        nearest_model = ExponentialSumHamiltonian(
            -0.7 .* X,
            [Z],
            [Z],
            reshape([-1.0], 1, 1),
            zeros(1, 1),
        )
        adaptive = vumps(
            nearest_model, 1;
            rng=MersenneTwister(0x5e6f), max_bond=2, maxiter=20,
            tol=1.0e-7, eig_tol=1.0e-11, env_tol=1.0e-11,
            transfer_tol=1.0e-11,
        )
        @test adaptive.converged
        @test adaptive.bond_dimension == 2
        @test any(event -> event.expanded, adaptive.growth_history)
        @test adaptive.energy ≈ -1.1268212 atol=3.0e-6

        expansion = expand_bond_dimension(
            adaptive.AL, adaptive.AR, adaptive.C, nearest_model;
            max_bond=3,
        )
        @test expansion.expanded
        @test expansion.new_bond == 3

        evolved = infinite_tdvp_step(
            adaptive.AL,
            adaptive.AR,
            adaptive.C,
            nearest_model,
            1.0e-3;
            max_bond=3,
            env_tol=1.0e-11,
            transfer_tol=1.0e-11,
        )
        @test evolved.expansion.expanded
        @test evolved.bond_dimension == 3
        @test abs(evolved.energy_after - evolved.energy_before) <= 3.0e-8
    end
end
