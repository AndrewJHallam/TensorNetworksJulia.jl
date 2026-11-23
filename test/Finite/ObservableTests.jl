# Gauge-independent finite observable and reduced-state tests.

function _observable_dense_rdm(state, dimensions, sites; normalize=true)
    N = length(dimensions)
    retained = collect(sites)
    traced = setdiff(collect(1:N), retained)
    retained_dimension = prod(dimensions[retained])
    rho = zeros(promote_type(eltype(state), ComplexF64),
                retained_dimension, retained_dimension)

    function digits(linear_index)
        result = Vector{Int}(undef, N)
        remainder = linear_index - 1
        for site in N:-1:1
            result[site] = mod(remainder, dimensions[site]) + 1
            remainder = div(remainder, dimensions[site])
        end
        return result
    end

    function retained_index(configuration)
        index = 1
        for site in retained
            index = (index - 1) * dimensions[site] + configuration[site]
        end
        return index
    end

    configurations = [digits(index) for index in eachindex(state)]
    for ket in eachindex(state), bra in eachindex(state)
        all(configurations[ket][site] == configurations[bra][site]
            for site in traced) || continue
        row = retained_index(configurations[ket])
        column = retained_index(configurations[bra])
        rho[row, column] += state[ket] * conj(state[bra])
    end
    if normalize
        rho ./= real(dot(state, state))
    end
    return rho
end
function _observable_gauge_transform(mps, bond, gauge)
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

function _observable_ghz_mps(N)
    N >= 2 || throw(ArgumentError("GHZ test state needs at least two sites"))
    psi = Vector{Array{ComplexF64,3}}(undef, N)
    psi[1] = zeros(ComplexF64, 1, 2, 2)
    psi[1][1, 1, 1] = inv(sqrt(2))
    psi[1][1, 2, 2] = inv(sqrt(2))
    for site in 2:(N - 1)
        psi[site] = zeros(ComplexF64, 2, 2, 2)
        psi[site][1, 1, 1] = 1
        psi[site][2, 2, 2] = 1
    end
    psi[N] = zeros(ComplexF64, 2, 1, 2)
    psi[N][1, 1, 1] = 1
    psi[N][2, 1, 2] = 1
    return psi
end

@testset "Gauge-independent finite observables" begin
    identity2 = ComplexF64[1 0; 0 1]
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]

    @testset "Local observables and operator strings" begin
        rng = MersenneTwister(0x0b5e)
        N = 4
        psi = random_mps(N, 2; maxdim=3, rng=rng, T=ComplexF64)
        psi[1] .*= 1.7 - 0.4im
        state = dense_state(psi)
        norm_squared = real(dot(state, state))

        expected_x = ComplexF64[]
        for site in 1:N
            operators = [copy(identity2) for _ in 1:N]
            operators[site] = X
            push!(expected_x,
                  dot(state, embedded_operator(operators) * state) / norm_squared)
        end
        @test local_expectations(psi, X) ≈ expected_x atol=5e-11 rtol=5e-11
        for site in 1:N
            @test local_expectation(psi, X, site) ≈ expected_x[site] atol=5e-11
        end
        @test local_expectation(psi, X, 2; normalize=false) ≈
              expected_x[2] * norm_squared atol=5e-11

        string_operators = [Y, Z, Z, X]
        string_reference = dot(
            state, embedded_operator(string_operators) * state,
        ) / norm_squared
        @test two_point_correlation(psi, Y, 1, X, 4; string=Z) ≈
              string_reference atol=5e-11
        @test two_point_correlation(psi, X, 4, Y, 1; string=Z) ≈
              string_reference atol=5e-11

        same_site_operators = [identity2, Y * X, identity2, identity2]
        same_site_reference = dot(
            state, embedded_operator(same_site_operators) * state,
        ) / norm_squared
        @test two_point_correlation(psi, Y, 2, X, 2) ≈
              same_site_reference atol=5e-11

        bond = 2
        dimension = size(psi[bond], 2)
        Q = Matrix(qr(randn(rng, ComplexF64, dimension, dimension)).Q)
        gauge = Q * Diagonal(range(0.55, 1.8; length=dimension))
        gauged = _observable_gauge_transform(psi, bond, gauge)
        @test dense_state(gauged) ≈ state atol=2e-10 rtol=2e-10
        @test local_expectations(gauged, X) ≈ expected_x atol=2e-10 rtol=2e-10
        @test two_point_correlation(gauged, Y, 1, X, 4; string=Z) ≈
              string_reference atol=2e-10

        @test_throws BoundsError local_expectation(psi, X, 0)
        @test_throws BoundsError local_expectation(psi, X, N + 1)
        @test_throws DimensionMismatch local_expectation(psi, ones(3, 3), 1)
        @test_throws DimensionMismatch two_point_correlation(
            psi, X, 1, X, 4; string=ones(3, 3),
        )

        zero_state = product_mps(N, 2, 1; T=ComplexF64)
        fill!(zero_state[1], 0)
        @test_throws ArgumentError local_expectation(zero_state, X, 1)
        @test local_expectation(zero_state, X, 1; normalize=false) == 0
    end

    @testset "Contiguous reduced density matrices" begin
        rng = MersenneTwister(0xd3a5)
        N = 4
        psi = random_mps(N, 2; maxdim=4, rng=rng, T=ComplexF64)
        psi[1] .*= -0.3 + 2.1im
        state = dense_state(psi)

        for sites in (1:1, 2:3, 1:N)
            rho = reduced_density_matrix(psi, sites)
            reference = _observable_dense_rdm(state, fill(2, N), sites)
            @test rho ≈ reference atol=8e-11 rtol=8e-11
            @test rho ≈ rho' atol=5e-12
            @test real(tr(rho)) ≈ 1 atol=5e-11
            @test minimum(eigvals(Hermitian(rho))) >= -2e-11
        end

        rho2 = reduced_density_matrix(psi, 2)
        @test tr(rho2 * Y) ≈ local_expectation(psi, Y, 2) atol=5e-11
        raw_rho = reduced_density_matrix(psi, 2:3; normalize=false)
        raw_reference = _observable_dense_rdm(
            state, fill(2, N), 2:3; normalize=false,
        )
        @test raw_rho ≈ raw_reference atol=8e-11 rtol=8e-11

        dimension = size(psi[2], 2)
        Q = Matrix(qr(randn(rng, ComplexF64, dimension, dimension)).Q)
        gauge = Q * Diagonal(range(0.6, 1.7; length=dimension))
        gauged = _observable_gauge_transform(psi, 2, gauge)
        @test reduced_density_matrix(gauged, 2:3) ≈
              reduced_density_matrix(psi, 2:3) atol=2e-10 rtol=2e-10

        heterogeneous = product_mps([
            ComplexF64[1, im],
            ComplexF64[1, 2, -im],
            ComplexF64[1, -1],
        ])
        @test size(reduced_density_matrix(heterogeneous, 1:2)) == (6, 6)
        @test tr(reduced_density_matrix(heterogeneous, 1:2)) ≈ 1 atol=5e-12
        @test_throws DimensionMismatch local_expectations(heterogeneous, X)

        @test_throws ArgumentError reduced_density_matrix(psi, 2:1)
        @test_throws BoundsError reduced_density_matrix(psi, 0:1)
        @test_throws BoundsError reduced_density_matrix(psi, 3:5)

        zero_state = product_mps(N, 2, 1; T=ComplexF64)
        fill!(zero_state[1], 0)
        @test_throws ArgumentError reduced_density_matrix(zero_state, 2:3)
        @test iszero(norm(reduced_density_matrix(
            zero_state, 2:3; normalize=false,
        )))
    end

    @testset "Schmidt values and entanglement entropy" begin
        product = product_mps(5, 2, 1; T=ComplexF64)
        for bond in 1:4
            @test schmidt_values(product, bond) ≈ [1.0] atol=5e-13
            @test entanglement_entropy(product, bond) ≈ 0 atol=5e-13
        end

        bell = _observable_ghz_mps(2)
        @test schmidt_values(bell, 1) ≈ fill(inv(sqrt(2)), 2) atol=5e-13
        @test entanglement_entropy(bell, 1; base=2) ≈ 1 atol=5e-13

        ghz = _observable_ghz_mps(5)
        for bond in 1:4
            @test schmidt_values(ghz, bond) ≈ fill(inv(sqrt(2)), 2) atol=5e-13
            @test entanglement_entropy(ghz, bond; base=2) ≈ 1 atol=5e-13
        end
        gauge = ComplexF64[1.4 0.3im; -0.2 0.65]
        gauged_ghz = _observable_gauge_transform(ghz, 2, gauge)
        @test schmidt_values(gauged_ghz, 2) ≈
              schmidt_values(ghz, 2) atol=2e-12

        rng = MersenneTwister(0x5c4d)
        random_state = random_mps(4, 2; maxdim=3, rng=rng, T=ComplexF64)
        random_state[1] .*= 2.3im
        state_before = dense_state(random_state)
        for bond in 1:3
            actual = schmidt_values(random_state, bond)
            dense_values = svdvals(reshape(
                state_before, 2^(4 - bond), 2^bond,
            ))
            dense_values ./= norm(dense_values)
            @test actual ≈ dense_values[1:length(actual)] atol=8e-11 rtol=8e-11
            rho = reduced_density_matrix(random_state, 1:bond)
            probabilities = filter(>(1e-13), real.(eigvals(Hermitian(rho))))
            entropy_reference = -sum(probabilities .* log.(probabilities))
            @test entanglement_entropy(random_state, bond) ≈
                  entropy_reference atol=2e-10
        end
        @test dense_state(random_state) ≈ state_before atol=5e-12
        raw_values = schmidt_values(random_state, 2; normalize=false)
        @test norm(raw_values) ≈ norm(state_before) atol=5e-11

        @test_throws ArgumentError schmidt_values(product, 0)
        @test_throws ArgumentError schmidt_values(product, 5)
        @test_throws ArgumentError schmidt_values(product_mps(1, 2), 1)
        @test_throws ArgumentError entanglement_entropy(product, 1; base=1)
        @test_throws ArgumentError entanglement_entropy(product, 1; base=-2)

        zero_state = product_mps(3, 2, 1; T=ComplexF64)
        fill!(zero_state[1], 0)
        @test_throws ArgumentError schmidt_values(zero_state, 1)
    end
end
