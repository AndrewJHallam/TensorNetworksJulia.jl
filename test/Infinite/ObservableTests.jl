@testset "Infinite observables" begin
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    v = ComplexF64[1, im] / sqrt(2)
    A = reshape(v, 1, 1, 2)
    C = ones(ComplexF64, 1, 1)
    @test local_expectation(A, C, Y) ≈ 1
    @test local_expectations(A, C, [X, Y, Z]) ≈ [0, 1, 0]
    @test reduced_density_matrix(A, C) ≈ v * v'
    @test reduced_density_matrix(A, C, 3) ≈ kron(v, v, v) * kron(v, v, v)'
    @test two_point_correlation(A, C, X, Z, 0) ≈ dot(v, X * Z * v)
    @test two_point_correlation(A, C, Y, Y, 17) ≈ 1
    @test abs(two_point_correlation(A, C, Y, Y, 17; connected=true)) < 1e-12
    @test entanglement_entropy(C) == 0
    @test correlation_length(A) == 0
    @test transfer_spectrum(A) ≈ [1]

    # A Pauli channel has known eigenvalues 1 and p-q (threefold).
    p = 0.7; q = (1-p)/3
    channel = cat(sqrt(p)*Matrix{ComplexF64}(I, 2, 2), sqrt(q)*X,
                  sqrt(q)*Y, sqrt(q)*Z; dims=3)
    centre = Matrix{ComplexF64}(I, 2, 2) / sqrt(2)
    @test transfer_spectrum(channel; nev=4) ≈ [1, p-q, p-q, p-q]
    @test correlation_length(channel) ≈ -1/log(p-q)
    @test schmidt_values(centre) ≈ fill(1/sqrt(2), 2)
    @test entanglement_entropy(centre; base=2) ≈ 1

    state = random_infinite_mps(3, 2; rng=MersenneTwister(832))
    rho = reduced_density_matrix(state.AL, state.C, 3)
    @test rho ≈ rho'
    @test tr(rho) ≈ 1
    @test minimum(eigvals(Hermitian(rho))) > -1e-12
    # Independent physical block amplitudes with the centre on the right.
    blocks = [state.AL[:, :, i] * state.AL[:, :, j] * state.AL[:, :, k] * state.C
              for i in 1:2 for j in 1:2 for k in 1:2]
    reference = [sum(blocks[i] .* conj.(blocks[j])) for i in 1:8, j in 1:8]
    @test rho ≈ reference / tr(reference)
    for O in (X, Y, X+im*Z), P in (Y, Z)
        @test local_expectation(state.AL, state.C, O) ≈
              tr(reduced_density_matrix(state.AL, state.C) * O)
        @test two_point_correlation(state.AL, state.C, O, P, 2) ≈
              tr(rho * kron(O, Matrix{ComplexF64}(I, 2, 2), P))
    end
    # Gauge rotations and centre rescaling leave physical measurements invariant.
    U = Matrix(qr(randn(MersenneTwister(57), ComplexF64, 3, 3)).Q)
    rotated = cat([U' * state.AL[:, :, s] * U for s in 1:2]...; dims=3)
    @test reduced_density_matrix(rotated, 3U'*state.C*U, 2) ≈
          reduced_density_matrix(state.AL, state.C, 2)
    @test correlation_length(rotated) ≈ correlation_length(state.AL)
    # Exercise Arnoldi above the dense threshold.
    larger = random_infinite_mps(5, 2; rng=MersenneTwister(91))
    spectrum = transfer_spectrum(larger.AL)
    @test spectrum[1] ≈ 1
    @test 0 < abs(spectrum[2]) < 1
    @test isfinite(correlation_length(larger.AL))
    @test_throws ArgumentError two_point_correlation(A, C, X, Y, -1)
    @test_throws DimensionMismatch local_expectation(A, C, ones(3, 3))
    @test_throws ArgumentError reduced_density_matrix(A, C, 50)
    @test_throws ArgumentError reduced_density_matrix(A, C, 0)
    @test_throws ArgumentError schmidt_values(zeros(2, 2))
    @test_throws ArgumentError entanglement_entropy(C; base=1)
    @test_throws ArgumentError transfer_spectrum(A; nev=0)
end
