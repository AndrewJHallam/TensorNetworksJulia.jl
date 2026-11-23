# Uniform-MPS state, nearest-neighbour VUMPS, and TDVP tests.

function dense_linear_map(action, prototype)
    dimension = length(prototype)
    matrix = zeros(ComplexF64, dimension, dimension)
    for column in 1:dimension
        basis = zeros(ComplexF64, dimension)
        basis[column] = 1
        matrix[:, column] .= vec(action(reshape(basis, size(prototype))))
    end
    return matrix
end

function one_site_density(AL, C)
    AC = center_tensor(AL, C)
    d = size(AC, 3)
    density = zeros(ComplexF64, d, d)
    for sout in 1:d, sin in 1:d
        density[sout, sin] = dot(vec(AC[:, :, sout]), vec(AC[:, :, sin]))
    end
    return density / tr(density)
end

@testset "Infinite MPS, VUMPS, and uniform TDVP" begin
    @testset "Two-site Hamiltonian convention" begin
        parameters = (
            J=-0.73,
            hx=0.31,
            hy=-0.17,
            hz=0.23,
            shift=0.08,
        )
        h = ising_two_site(; parameters..., interaction=:z)
        matrix = two_site_matrix(h)
        I2 = Matrix{ComplexF64}(I, 2, 2)
        X = ComplexF64[0 1; 1 0]
        Y = ComplexF64[0 -im; im 0]
        Z = ComplexF64[1 0; 0 -1]
        onsite = parameters.hx .* X .+ parameters.hy .* Y .+
                 parameters.hz .* Z .- parameters.shift .* I2
        reference = parameters.J .* kron(Z, Z) .+
                    0.5 .* (kron(onsite, I2) .+ kron(I2, onsite))
        @test matrix ≈ reference atol=TEST_ATOL
        @test matrix ≈ matrix' atol=TEST_ATOL
    end

    @testset "Mixed canonical form and transfer fixed points" begin
        state = random_infinite_mps(
            3, 2; rng=MersenneTwister(0x606), T=ComplexF64,
        )
        residuals = infinite_residuals(state.AL, state.AR, state.C)
        @test residuals.left_canonical <= 2.0e-10
        @test residuals.right_canonical <= 2.0e-10
        @test residuals.center_consistency <= 2.0e-10
        @test state.transfer_residual <= 2.0e-10

        fixed = transfer_fixed_points(state.AL, state.AR)
        @test fixed.converged
        @test fixed.residual_L <= 2.0e-10
        @test fixed.residual_R <= 2.0e-10
        @test tr(fixed.L) ≈ 1 atol=TEST_ATOL
        @test tr(fixed.R) ≈ 1 atol=TEST_ATOL
        @test transfer_left(state.AR, fixed.L) ≈ fixed.L atol=2.0e-10
        @test transfer_right(state.AL, fixed.R) ≈ fixed.R atol=2.0e-10
    end

    @testset "Effective Hamiltonians are Hermitian" begin
        h = ising_two_site(J=-1.0, hx=-0.7, hz=0.11, interaction=:z)
        state = random_infinite_mps(
            2, 2; rng=MersenneTwister(0x707), T=ComplexF64,
        )
        step = vumps_step(
            state.AL, state.AR, state.C, h;
            eig_tol=1.0e-11, env_tol=1.0e-11, transfer_tol=1.0e-11,
        )
        AC = center_tensor(state.AL, state.C)
        HAC = dense_linear_map(
            X -> apply_hac(X, state.AL, state.AR, h, step.HL, step.HR),
            AC,
        )
        HC = dense_linear_map(
            X -> apply_hc(X, state.AL, state.AR, h, step.HL, step.HR),
            state.C,
        )
        @test HAC ≈ HAC' atol=2.0e-10
        @test HC ≈ HC' atol=2.0e-10
        @test step.eig_converged
        @test step.environments_converged
        @test step.transfer_converged
    end

    @testset "VUMPS product state and interacting Ising chain" begin
        product_h = ising_two_site(J=0.0, hz=-1.0, interaction=:z)
        product = vumps(
            product_h,
            1;
            rng=MersenneTwister(0x808),
            maxiter=5,
            tol=1.0e-11,
        )
        @test product.converged
        @test product.energy ≈ -1 atol=2.0e-11
        @test product.residual <= 2.0e-11
        @test product.residuals.left_canonical <= 2.0e-11
        @test product.residuals.right_canonical <= 2.0e-11

        g = 0.7
        interacting_h = ising_two_site(
            J=-1.0, hx=-g, interaction=:z,
        )
        interacting = vumps(
            interacting_h,
            2;
            rng=MersenneTwister(9),
            maxiter=20,
            tol=1.0e-7,
            eig_tol=1.0e-11,
            env_tol=1.0e-11,
            transfer_tol=1.0e-11,
        )
        # Exact thermodynamic-limit energy density for
        # -sum Z Z - g sum X, evaluated by midpoint quadrature.
        quadrature_points = 20_000
        exact_energy = -sum(
            sqrt(1 + g^2 - 2g * cos(2pi * (j - 0.5) / quadrature_points))
            for j in 1:quadrature_points
        ) / quadrature_points
        @test interacting.converged
        @test interacting.residual <= 1.0e-7
        @test abs(interacting.energy - exact_energy) <= 2.0e-5
        @test last(interacting.history).residual < first(interacting.history).residual
        @test interacting.residuals.left_canonical <= 2.0e-10
        @test interacting.residuals.right_canonical <= 2.0e-10

        evolved = infinite_tdvp_step(
            interacting.AL,
            interacting.AR,
            interacting.C,
            interacting_h,
            0.01;
            tol=1.0e-12,
            env_tol=1.0e-11,
            transfer_tol=1.0e-11,
        )
        @test evolved.krylov_converged
        @test abs(evolved.energy_after - evolved.energy_before) <= 2.0e-8
        @test evolved.residual <= 2.0e-7
        @test evolved.residuals.left_canonical <= 2.0e-10
        @test evolved.residuals.right_canonical <= 2.0e-10
    end

    @testset "Uniform TDVP preserves an exact stationary product state" begin
        h = ising_two_site(J=0.0, hz=-1.0, interaction=:z)
        ground = vumps(h, 1; rng=MersenneTwister(0x909), maxiter=5)
        density_before = one_site_density(ground.AL, ground.C)
        evolved = vtdvp_step(ground.AL, ground.AR, ground.C, h, 0.17)
        density_after = one_site_density(evolved.AL, evolved.C)
        @test density_after ≈ density_before atol=2.0e-11
        @test evolved.energy_before ≈ -1 atol=2.0e-11
        @test evolved.energy_after ≈ -1 atol=2.0e-11
        @test evolved.residual <= 2.0e-11
        @test evolved.krylov_converged

        random_state = random_infinite_mps(
            2, 2; rng=MersenneTwister(0xa0a), T=ComplexF64,
        )
        zero_h = zeros(ComplexF64, 2, 2, 2, 2)
        random_density = one_site_density(random_state.AL, random_state.C)
        forward = infinite_tdvp_step(
            random_state.AL, random_state.AR, random_state.C, zero_h, 0.13,
        )
        backward = infinite_tdvp_step(
            forward.AL, forward.AR, forward.C, zero_h, -0.13,
        )
        @test one_site_density(forward.AL, forward.C) ≈
              random_density atol=2.0e-11
        @test one_site_density(backward.AL, backward.C) ≈
              random_density atol=2.0e-11
        @test forward.energy_after ≈ 0 atol=TEST_ATOL
        @test forward.krylov_converged && backward.krylov_converged
    end
end
