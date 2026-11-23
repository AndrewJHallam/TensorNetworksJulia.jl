# Finite open-boundary long-range MPO tests.

@testset "Finite exponential-sum MPO" begin
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    onsite = 0.17X - 0.09Y + 0.23Z
    left_ops = [X + 0.2Y, Z]
    right_ops = [Z - 0.1X, Y]
    coefficients = ComplexF64[0.7 -0.21; 0.0 0.34]
    decays = ComplexF64[0.31 -0.18; 0.12 0.47]
    h = ExponentialSumHamiltonian(
        onsite, left_ops, right_ops, coefficients, decays,
    )

    @testset "single-site boundary" begin
        mpo = finite_mpo(h, 1)
        @test length(mpo) == 1
        @test size(only(mpo)) == (2, 2, 1, 1)
        @test mpo_matrix(mpo) ≈ onsite atol=TEST_ATOL
    end

    @testset "dense open-chain reference" begin
        N = 4
        mpo = finite_mpo(h, N)
        mpo_from_uniform = finite_mpo(triangular_mpo(h), N)
        dense = mpo_matrix(mpo)
        @test dense ≈ mpo_matrix(mpo_from_uniform) atol=TEST_ATOL

        reference = zeros(ComplexF64, 2^N, 2^N)
        for site in 1:N
            operators = [copy(I2) for _ in 1:N]
            operators[site] = onsite
            reference .+= embedded_operator(operators)
        end
        for first_site in 1:(N - 1), second_site in (first_site + 1):N
            separation = second_site - first_site
            for operator_index in eachindex(left_ops)
                coefficient = sum(
                    coefficients[operator_index, exponential_index] *
                    decays[operator_index, exponential_index]^(separation - 1)
                    for exponential_index in axes(coefficients, 2)
                )
                operators = [copy(I2) for _ in 1:N]
                operators[first_site] = left_ops[operator_index]
                operators[second_site] = right_ops[operator_index]
                reference .+= coefficient .* embedded_operator(operators)
            end
        end
        @test dense ≈ reference atol=2.0e-11 rtol=2.0e-11
    end

    @test_throws ArgumentError finite_mpo(h, 0)
end
