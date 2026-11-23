# Finite model MPO dense-reference tests.

function _finite_model_embed(N, d, site_operators...)
    T = promote_type(
        ComplexF64,
        (eltype(operator) for (_, operator) in site_operators)...,
    )
    operators = [Matrix{T}(I, d, d) for _ in 1:N]
    for (site, operator) in site_operators
        operators[site] = Matrix{T}(operator)
    end
    return reduce(kron, operators)
end

function _finite_model_exact_xyz(
    N;
    Jx,
    Jy,
    Jz,
    hx,
    hy,
    hz,
    shift,
)
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    H = zeros(ComplexF64, 2^N, 2^N)
    for site in 1:N
        onsite = hx[site] .* X .+ hy[site] .* Y .+ hz[site] .* Z .-
                 shift[site] .* I2
        H .+= _finite_model_embed(N, 2, (site, onsite))
    end
    for bond in 1:(N - 1)
        H .+= Jx[bond] .* _finite_model_embed(
            N, 2, (bond, X), (bond + 1, X),
        )
        H .+= Jy[bond] .* _finite_model_embed(
            N, 2, (bond, Y), (bond + 1, Y),
        )
        H .+= Jz[bond] .* _finite_model_embed(
            N, 2, (bond, Z), (bond + 1, Z),
        )
    end
    return H
end

function _finite_model_exact_pxp(N, detuning, interaction)
    X = ComplexF64[0 1; 1 0]
    P = ComplexF64[0 0; 0 1]
    Q = ComplexF64[1 0; 0 0]
    H = zeros(ComplexF64, 2^N, 2^N)
    if N == 1
        H .+= X
    else
        H .+= _finite_model_embed(N, 2, (1, X), (2, P))
        for site in 2:(N - 1)
            H .+= _finite_model_embed(
                N, 2, (site - 1, P), (site, X), (site + 1, P),
            )
        end
        H .+= _finite_model_embed(N, 2, (N - 1, P), (N, X))
    end
    for site in 1:N
        H .+= detuning[site] .* _finite_model_embed(N, 2, (site, Q))
    end
    for bond in 1:(N - 1)
        H .+= interaction[bond] .* _finite_model_embed(
            N, 2, (bond, Q), (bond + 1, Q),
        )
    end
    return H
end

@testset "Finite MPO model constructors" begin
    @testset "Generic nearest-neighbour automaton" begin
        N = 4
        d = 3
        onsite = [
            ComplexF64[
                0.1site 0.02im 0.03
                -0.02im -0.07site 0.01
                0.03 0.01 0.04site
            ] for site in 1:N
        ]
        left_ops = [
            ComplexF64[0 1 0; 0 0 2; 0 0 0],
            ComplexF64[1 0 0; 0 -1 0; 0 0 0.5],
        ]
        right_ops = [
            ComplexF64[0 0 0; 1 0 0; 0 2 0],
            ComplexF64[0 1 0; 1 0 0; 0 0 -0.2],
        ]
        couplings = ComplexF64[
            0.31 -0.27im 0.19
            -0.11 0.23 0.07im
        ]
        mpo = nearest_neighbor_mpo(
            N, onsite, left_ops, right_ops; couplings=couplings,
        )
        exact = zeros(ComplexF64, d^N, d^N)
        for site in 1:N
            exact .+= _finite_model_embed(N, d, (site, onsite[site]))
        end
        for bond in 1:(N - 1), channel in eachindex(left_ops)
            exact .+= couplings[channel, bond] .* _finite_model_embed(
                N,
                d,
                (bond, left_ops[channel]),
                (bond + 1, right_ops[channel]),
            )
        end
        @test mpo_matrix(mpo) ≈ exact atol=2.0e-12
        @test size(first(mpo)) == (d, d, 1, 4)
        @test size(last(mpo)) == (d, d, 4, 1)

        single_onsite = ComplexF64[0.2 0.1im; -0.3im -0.4]
        single = nearest_neighbor_mpo(
            1,
            single_onsite,
            [ComplexF64[0 1; 0 0]],
            [ComplexF64[0 0; 1 0]];
            couplings=2.0,
        )
        @test size(only(single)) == (2, 2, 1, 1)
        @test mpo_matrix(single) == single_onsite

        @test_throws DimensionMismatch nearest_neighbor_mpo(
            3, zeros(2, 2), [zeros(2, 2)], [zeros(3, 3)],
        )
        @test_throws DimensionMismatch nearest_neighbor_mpo(
            3,
            zeros(2, 2),
            [zeros(2, 2)],
            [zeros(2, 2)];
            couplings=zeros(1, 3),
        )
    end

    @testset "XYZ with site and bond coefficients" begin
        N = 4
        parameters = (
            Jx=[0.31, -0.22, 0.17],
            Jy=[-0.27, 0.08, 0.29],
            Jz=[0.19, 0.14, -0.11],
            hx=[-0.23, 0.05, 0.13, -0.09],
            hy=[0.17, -0.04, 0.07, 0.12],
            hz=[0.11, 0.21, -0.16, 0.03],
            shift=[0.07, -0.02, 0.01, 0.09],
        )
        dense = mpo_matrix(xyz_mpo(N; parameters...))
        exact = _finite_model_exact_xyz(N; parameters...)
        @test dense ≈ exact atol=2.0e-12
        @test dense ≈ dense' atol=2.0e-12

        scalar = xyz_mpo(
            2; Jx=0.4, Jy=-0.3, Jz=0.2,
            hx=0.1, hy=-0.05, hz=0.07, shift=0.03,
        )
        scalar_parameters = (
            Jx=fill(0.4, 1), Jy=fill(-0.3, 1), Jz=fill(0.2, 1),
            hx=fill(0.1, 2), hy=fill(-0.05, 2), hz=fill(0.07, 2),
            shift=fill(0.03, 2),
        )
        @test mpo_matrix(scalar) ≈
              _finite_model_exact_xyz(2; scalar_parameters...) atol=2.0e-12

        one_site = xyz_mpo(
            1; Jx=3.0, Jy=-2.0, Jz=1.0,
            hx=[0.21], hy=[-0.14], hz=[0.08], shift=[0.03],
        )
        one_parameters = (
            Jx=Float64[], Jy=Float64[], Jz=Float64[],
            hx=[0.21], hy=[-0.14], hz=[0.08], shift=[0.03],
        )
        @test size(only(one_site)) == (2, 2, 1, 1)
        @test mpo_matrix(one_site) ≈
              _finite_model_exact_xyz(1; one_parameters...) atol=2.0e-12
        @test_throws DimensionMismatch xyz_mpo(4; hx=zeros(3))
        @test_throws DimensionMismatch xyz_mpo(4; Jz=zeros(4))
    end

    @testset "Truncated Bose-Hubbard model" begin
        N = 3
        nmax = 2
        hopping = ComplexF64[0.37 + 0.09im, -0.21 + 0.04im]
        U = [1.2, 0.8, 1.4]
        potential = [0.17, -0.11, 0.23]
        chemical_potential = [0.04, 0.07, -0.02]
        mpo = bose_hubbard_mpo(
            N;
            hopping=hopping,
            U=U,
            potential=potential,
            chemical_potential=chemical_potential,
            nmax=nmax,
        )

        d = nmax + 1
        b = zeros(ComplexF64, d, d)
        for occupation in 1:nmax
            b[occupation, occupation + 1] = sqrt(occupation)
        end
        bdag = b'
        number = Matrix(Diagonal(ComplexF64.(0:nmax)))
        exact = zeros(ComplexF64, d^N, d^N)
        for site in 1:N
            onsite = U[site] / 2 .* number * (number - I) .+
                     (potential[site] - chemical_potential[site]) .* number
            exact .+= _finite_model_embed(N, d, (site, onsite))
        end
        for bond in 1:(N - 1)
            exact .-= hopping[bond] .* _finite_model_embed(
                N, d, (bond, bdag), (bond + 1, b),
            )
            exact .-= conj(hopping[bond]) .* _finite_model_embed(
                N, d, (bond, b), (bond + 1, bdag),
            )
        end
        dense = mpo_matrix(mpo)
        @test dense ≈ exact atol=3.0e-12
        @test dense ≈ dense' atol=3.0e-12

        total_number = sum(
            _finite_model_embed(N, d, (site, number)) for site in 1:N
        )
        @test norm(dense * total_number - total_number * dense) <= 2.0e-11

        single = bose_hubbard_mpo(
            1; hopping=9.0, U=[1.3], potential=[0.2],
            chemical_potential=[0.05], nmax=3,
        )
        expected_diagonal = ComplexF64[
            1.3 / 2 * occupation * (occupation - 1) + 0.15 * occupation
            for occupation in 0:3
        ]
        @test size(only(single)) == (4, 4, 1, 1)
        @test mpo_matrix(single) ≈ Matrix(Diagonal(expected_diagonal))
        @test_throws ArgumentError bose_hubbard_mpo(3; nmax=-1)
        @test_throws DimensionMismatch bose_hubbard_mpo(
            3; potential=zeros(2),
        )
    end

    @testset "Edge-aware PXP model" begin
        for N in (1, 2, 3, 5)
            detuning = [0.07 * site - 0.13 for site in 1:N]
            interaction = [0.09 - 0.04 * bond for bond in 1:(N - 1)]
            dense = mpo_matrix(pxp_mpo(
                N; detuning=detuning, interaction=interaction,
            ))
            exact = _finite_model_exact_pxp(N, detuning, interaction)
            @test dense ≈ exact atol=2.0e-12
            @test dense ≈ dense' atol=2.0e-12
        end

        positional = pxp_mpo(4, 0.23, -0.17)
        exact = _finite_model_exact_pxp(
            4, fill(0.23, 4), fill(-0.17, 3),
        )
        @test mpo_matrix(positional) ≈ exact atol=2.0e-12
        @test_throws DimensionMismatch pxp_mpo(4; detuning=zeros(3))
        @test_throws DimensionMismatch pxp_mpo(4; interaction=zeros(4))
    end
end
