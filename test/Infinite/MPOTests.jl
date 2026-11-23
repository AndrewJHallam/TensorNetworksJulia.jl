# Infinite-MPO VUMPS and uniform-TDVP tests.

using Test
using LinearAlgebra
using Random
using TensorNetworksJulia

const UNIFORM_MPO_TN = TensorNetworksJulia

function uniform_mpo_dense_map(action, prototype)
    n = length(prototype)
    matrix = zeros(ComplexF64, n, n)
    for column in 1:n
        basis = zeros(ComplexF64, n)
        basis[column] = 1
        matrix[:, column] .= vec(action(reshape(basis, size(prototype))))
    end
    return matrix
end

function uniform_mpo_one_site_density(AL, C)
    AC = center_tensor(AL, C)
    d = size(AC, 3)
    density = zeros(ComplexF64, d, d)
    for sout in 1:d, sin in 1:d
        density[sout, sin] = dot(vec(AC[:, :, sout]), vec(AC[:, :, sin]))
    end
    return density / tr(density)
end

function onsite_triangular_mpo(operator)
    d = size(operator, 1)
    T = promote_type(eltype(operator), Float64)
    identity_d = Matrix{T}(I, d, d)
    W = zeros(T, d, d, 2, 2)
    W[:, :, 1, 1] .= identity_d
    W[:, :, 2, 1] .= operator
    W[:, :, 2, 2] .= identity_d
    return UNIFORM_MPO_TN.TriangularMPO(W)
end

@testset "Uniform triangular MPO" begin
    I2 = Matrix{ComplexF64}(I, 2, 2)
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]

    lambda = 0.25
    coupling = -2.0
    field = 0.7
    W = zeros(ComplexF64, 2, 2, 3, 3)
    W[:, :, 1, 1] .= I2
    W[:, :, 2, 1] .= Y
    W[:, :, 2, 2] .= lambda .* I2
    W[:, :, 3, 1] .= field .* Y
    W[:, :, 3, 2] .= coupling .* Y
    W[:, :, 3, 3] .= I2

    @testset "Validated representation" begin
        H = UNIFORM_MPO_TN.TriangularMPO(W)
        @test H.lambda ≈ [1, lambda, 1]
        @test size(H) == (2, 2, 3, 3)
        @test UNIFORM_MPO_TN.TriangularMPO(W, [1, lambda, 1]).lambda ≈
              H.lambda

        @test_throws MethodError UNIFORM_MPO_TN.TriangularMPO(zeros(2, 2, 2))
        @test_throws DimensionMismatch UNIFORM_MPO_TN.TriangularMPO(
            zeros(2, 3, 3, 3),
        )
        @test_throws DimensionMismatch UNIFORM_MPO_TN.TriangularMPO(
            zeros(2, 2, 2, 3),
        )
        @test_throws ArgumentError UNIFORM_MPO_TN.TriangularMPO(
            reshape(I2, 2, 2, 1, 1),
        )

        upper = copy(W)
        upper[:, :, 1, 2] .= X
        @test_throws ArgumentError UNIFORM_MPO_TN.TriangularMPO(upper)

        noisy = copy(W)
        noisy[1, 1, 1, 2] = 1.0e-14
        @test UNIFORM_MPO_TN.TriangularMPO(noisy) isa
              UNIFORM_MPO_TN.TriangularMPO
        noisy[1, 1, 1, 2] = 1.0e-6
        @test_throws ArgumentError UNIFORM_MPO_TN.TriangularMPO(noisy)

        nonscalar = copy(W)
        nonscalar[:, :, 2, 2] .= Z
        @test_throws ArgumentError UNIFORM_MPO_TN.TriangularMPO(nonscalar)

        bad_endpoint = copy(W)
        bad_endpoint[:, :, 1, 1] .= 2 .* I2
        @test_throws ArgumentError UNIFORM_MPO_TN.TriangularMPO(bad_endpoint)
        @test_throws ArgumentError UNIFORM_MPO_TN.TriangularMPO(
            W, [1, 0.5, 1],
        )
    end

    @testset "Exact exponential channel and finite contraction" begin
        # This is the +1 eigenstate of Y.  It makes every channel scalar while
        # retaining a genuinely complex physical operator.
        vector = ComplexF64[1, im] / sqrt(2)
        A = reshape(vector, 1, 1, 2)
        C = ones(ComplexF64, 1, 1)
        H = UNIFORM_MPO_TN.TriangularMPO(W)
        context = UNIFORM_MPO_TN.uniform_mpo_environments(A, A, C, H)
        expected_energy = field + coupling / (1 - lambda)

        @test context.transfer_converged
        @test context.environments_converged
        @test context.environment_residual <= 1.0e-12
        @test context.energy_mismatch <= 1.0e-12
        @test context.energy ≈ expected_energy atol=1.0e-12
        @test context.FL[1, :, 1] ≈ [0, coupling / (1 - lambda), 1]
        @test context.FR[1, :, 1] ≈ [1, 1 / (1 - lambda), 0]

        left_source = UNIFORM_MPO_TN._mpo_left_block(
            view(context.FL, :, 3, :), A, view(W, :, :, 3, 2),
        )
        left_image = UNIFORM_MPO_TN._mpo_left_block(
            view(context.FL, :, 2, :), A, lambda .* I2,
        )
        @test Matrix(view(context.FL, :, 2, :)) - left_image ≈
              left_source atol=1.0e-12

        right_source = UNIFORM_MPO_TN._mpo_right_block(
            A, view(W, :, :, 2, 1), view(context.FR, :, 1, :),
        )
        right_image = UNIFORM_MPO_TN._mpo_right_block(
            A, lambda .* I2, view(context.FR, :, 2, :),
        )
        @test Matrix(view(context.FR, :, 2, :)) - right_image ≈
              right_source atol=1.0e-12

        left_boundary = Array(W[:, :, 3:3, :])
        right_boundary = Array(W[:, :, :, 1:1])
        dense = mpo_matrix([left_boundary, W, right_boundary])
        expected_dense = field .* (
            kron(Y, I2, I2) + kron(I2, Y, I2) + kron(I2, I2, Y)
        ) + coupling .* (
            kron(Y, Y, I2) + kron(I2, Y, Y) + lambda .* kron(Y, I2, Y)
        )
        @test dense ≈ expected_dense atol=1.0e-12

        # Exercise the genuine χ² Krylov solve for an internal |lambda| < 1
        # channel, rather than only the scalar χ=1 shortcut above.
        random_state = random_infinite_mps(
            2, 2; rng=MersenneTwister(0xa11), T=ComplexF64,
        )
        random_context = UNIFORM_MPO_TN.uniform_mpo_environments(
            random_state.AL, random_state.AR, random_state.C, H,
        )
        random_left_source = UNIFORM_MPO_TN._mpo_left_block(
            view(random_context.FL, :, 3, :),
            random_state.AL,
            view(W, :, :, 3, 2),
        )
        random_left_transfer = UNIFORM_MPO_TN._mpo_left_block(
            view(random_context.FL, :, 2, :), random_state.AL, I2,
        )
        random_right_source = UNIFORM_MPO_TN._mpo_right_block(
            random_state.AR,
            view(W, :, :, 2, 1),
            view(random_context.FR, :, 1, :),
        )
        random_right_transfer = UNIFORM_MPO_TN._mpo_right_block(
            random_state.AR, I2, view(random_context.FR, :, 2, :),
        )
        @test random_context.environments_converged
        @test random_context.environment_residual <= 2.0e-10
        @test Matrix(view(random_context.FL, :, 2, :)) -
              lambda .* random_left_transfer ≈ random_left_source atol=2.0e-10
        @test Matrix(view(random_context.FR, :, 2, :)) -
              lambda .* random_right_transfer ≈ random_right_source atol=2.0e-10
    end

    @testset "Short-range MPO agrees with two-site Hamiltonian" begin
        parameters = (
            J=-0.73,
            hx=0.31,
            hy=-0.17,
            hz=0.23,
            shift=0.08,
        )
        Wshort = ising_mpo(3; parameters..., interaction=:z)[2]
        H = UNIFORM_MPO_TN.TriangularMPO(Wshort)
        h = ising_two_site(; parameters..., interaction=:z)
        state = random_infinite_mps(
            2, 2; rng=MersenneTwister(0xb0b), T=ComplexF64,
        )
        mpo_context = UNIFORM_MPO_TN.uniform_mpo_environments(
            state.AL, state.AR, state.C, H;
            transfer_tol=1.0e-12, env_tol=1.0e-12,
        )
        local_context = UNIFORM_MPO_TN._uniform_context(
            state.AL, state.AR, state.C, h;
            transfer_tol=1.0e-12, env_tol=1.0e-12,
        )
        @test mpo_context.energy ≈
              real(uniform_energy(state.AL, state.AR, state.C, h)) atol=2.0e-9

        AC = center_tensor(state.AL, state.C)
        mpo_AC = uniform_mpo_dense_map(
            T -> apply_heff1(T, mpo_context.FL, Wshort, mpo_context.FR), AC,
        )
        local_AC = uniform_mpo_dense_map(
            T -> apply_hac(
                T, state.AL, state.AR, h, local_context.HL, local_context.HR,
            ),
            AC,
        )
        mpo_C = uniform_mpo_dense_map(
            T -> apply_keff(T, mpo_context.FL, mpo_context.FR), state.C,
        )
        local_C = uniform_mpo_dense_map(
            T -> apply_hc(
                T, state.AL, state.AR, h, local_context.HL, local_context.HR,
            ),
            state.C,
        )

        @test mpo_AC ≈ mpo_AC' atol=2.0e-9
        @test mpo_C ≈ mpo_C' atol=2.0e-9
        delta_AC_matrix = local_AC - mpo_AC
        delta_C_matrix = local_C - mpo_C
        delta_AC = tr(delta_AC_matrix) / size(delta_AC_matrix, 1)
        delta_C = tr(delta_C_matrix) / size(delta_C_matrix, 1)
        @test delta_AC_matrix ≈
              delta_AC .* Matrix{ComplexF64}(I, size(delta_AC_matrix)...) atol=2.0e-8
        @test delta_C_matrix ≈
              delta_C .* Matrix{ComplexF64}(I, size(delta_C_matrix)...) atol=2.0e-8
        @test delta_AC ≈ delta_C atol=2.0e-8
    end

    @testset "MPO VUMPS and TDVP" begin
        product_H = onsite_triangular_mpo(-Z)
        ground = UNIFORM_MPO_TN.vumps(
            product_H, 1;
            rng=MersenneTwister(0xc0c),
            maxiter=4,
            tol=1.0e-11,
        )
        @test ground.converged
        @test ground.energy ≈ -1 atol=2.0e-11
        @test ground.residual <= 2.0e-11
        @test ground.environments_converged
        @test ground.transfer_converged

        Y_H = onsite_triangular_mpo(Y)
        initial_A = zeros(ComplexF64, 1, 1, 2)
        initial_A[1, 1, 1] = 1
        initial_C = ones(ComplexF64, 1, 1)
        dt = 0.37
        evolved = UNIFORM_MPO_TN.infinite_tdvp_step(
            initial_A, initial_A, initial_C, Y_H, dt,
        )
        target = ComplexF64[cos(dt), sin(dt)]
        target_density = target * target'
        @test uniform_mpo_one_site_density(evolved.AL, evolved.C) ≈
              target_density atol=2.0e-11
        @test evolved.energy_before ≈ 0 atol=2.0e-11
        @test evolved.energy_after ≈ 0 atol=2.0e-11
        @test evolved.krylov_converged

        restored = UNIFORM_MPO_TN.infinite_tdvp_step(
            evolved.AL, evolved.AR, evolved.C, Y_H, -dt;
            FL0=evolved.FL, FR0=evolved.FR, L0=evolved.L, R0=evolved.R,
        )
        initial_density = ComplexF64[1 0; 0 0]
        @test uniform_mpo_one_site_density(restored.AL, restored.C) ≈
              initial_density atol=3.0e-11
        @test restored.krylov_converged
    end

    @testset "Adaptive MPO bond growth" begin
        product_H = onsite_triangular_mpo(-Z)
        product = UNIFORM_MPO_TN.vumps(
            product_H, 1; rng=MersenneTwister(0xd0d), maxiter=4,
        )
        no_growth = expand_bond_dimension(
            product.AL, product.AR, product.C, product_H; max_bond=2,
        )
        @test !no_growth.expanded
        @test no_growth.new_bond == 1

        Wshort = ising_mpo(
            3; J=-1.0, hx=-0.7, interaction=:z,
        )[2]
        H = UNIFORM_MPO_TN.TriangularMPO(Wshort)
        adaptive = UNIFORM_MPO_TN.vumps(
            H, 1;
            rng=MersenneTwister(0xe0e), max_bond=2, maxiter=20,
            tol=1.0e-7, eig_tol=1.0e-11, env_tol=1.0e-11,
            transfer_tol=1.0e-11,
        )
        @test adaptive.converged
        @test adaptive.bond_dimension == 2
        @test any(event -> event.expanded, adaptive.growth_history)
        @test adaptive.energy ≈ -1.1268212 atol=3.0e-6

        expansion = expand_bond_dimension(
            adaptive.AL, adaptive.AR, adaptive.C, H; max_bond=3,
        )
        @test expansion.expanded
        @test expansion.new_bond == 3
        @test expansion.environment_energy ≈ adaptive.energy atol=2.0e-8

        evolved = UNIFORM_MPO_TN.infinite_tdvp_step(
            adaptive.AL, adaptive.AR, adaptive.C, H, 1.0e-3;
            max_bond=3, env_tol=1.0e-11, transfer_tol=1.0e-11,
        )
        @test evolved.expansion.expanded
        @test evolved.bond_dimension == 3
        @test abs(evolved.energy_after - evolved.energy_before) <= 3.0e-8
    end
end
