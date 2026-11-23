# Finite TEBD gate, schedule, and truncation tests.

"""Independent dense conversion for `U[out1,out2,in1,in2]`."""
function _tebd_tensor_to_matrix(U)
    d1out, d2out, d1in, d2in = size(U)
    matrix = zeros(eltype(U), d1out * d2out, d1in * d2in)
    for out1 in 1:d1out, out2 in 1:d2out
        row = (out1 - 1) * d2out + out2
        for in1 in 1:d1in, in2 in 1:d2in
            column = (in1 - 1) * d2in + in2
            matrix[row, column] = U[out1, out2, in1, in2]
        end
    end
    return matrix
end

"""Independent dense-to-tensor conversion using site 1 as the slow index."""
function _tebd_matrix_to_tensor(matrix, d1, d2)
    size(matrix) == (d1 * d2, d1 * d2) || throw(DimensionMismatch())
    tensor = zeros(eltype(matrix), d1, d2, d1, d2)
    for out1 in 1:d1, out2 in 1:d2
        row = (out1 - 1) * d2 + out2
        for in1 in 1:d1, in2 in 1:d2
            column = (in1 - 1) * d2 + in2
            tensor[out1, out2, in1, in2] = matrix[row, column]
        end
    end
    return tensor
end

function _tebd_embed_one_site(operator, site, N, d)
    T = eltype(operator)
    left = Matrix{T}(I, d^(site - 1), d^(site - 1))
    right = Matrix{T}(I, d^(N - site), d^(N - site))
    return kron(kron(left, operator), right)
end

function _tebd_embed_two_site(operator, bond, N, d)
    T = eltype(operator)
    left = Matrix{T}(I, d^(bond - 1), d^(bond - 1))
    right = Matrix{T}(I, d^(N - bond - 1), d^(N - bond - 1))
    return kron(kron(left, operator), right)
end

function _tebd_random_hamiltonians(rng, N; d=2)
    hamiltonians = Vector{Array{ComplexF64,4}}(undef, N - 1)
    for bond in 1:(N - 1)
        raw = randn(rng, ComplexF64, d^2, d^2)
        matrix = raw + raw'
        matrix ./= norm(matrix)
        hamiltonians[bond] = _tebd_matrix_to_tensor(matrix, d, d)
    end
    return hamiltonians
end

function _tebd_layer_generators(hamiltonians, N, d)
    dimension = d^N
    odd = zeros(ComplexF64, dimension, dimension)
    even = zeros(ComplexF64, dimension, dimension)
    for bond in eachindex(hamiltonians)
        local_matrix = _tebd_tensor_to_matrix(hamiltonians[bond])
        embedded = _tebd_embed_two_site(local_matrix, bond, N, d)
        if isodd(bond)
            odd .+= embedded
        else
            even .+= embedded
        end
    end
    return odd, even
end

function _tebd_dense_schedule(odd, even, dt, order; time=:real)
    parameter = time === :real ? -im * dt : -dt
    dimension = size(odd, 1)

    first_order(coefficient) =
        exp((coefficient * parameter) .* even) *
        exp((coefficient * parameter) .* odd)
    second_order(coefficient) =
        exp((coefficient * parameter / 2) .* odd) *
        exp((coefficient * parameter) .* even) *
        exp((coefficient * parameter / 2) .* odd)

    if order == 1
        return first_order(1)
    elseif order == 2
        return second_order(1)
    elseif order == 4
        p = 1 / (4 - cbrt(4.0))
        result = Matrix{ComplexF64}(I, dimension, dimension)
        for coefficient in (p, p, 1 - 4p, p, p)
            result = second_order(coefficient) * result
        end
        return result
    end
    throw(ArgumentError("unsupported test order"))
end

@testset "Finite TEBD" begin
    @testset "Gate tensor orientation" begin
        rng = MersenneTwister(0x6a7e)
        d1, d2 = 2, 3
        raw = randn(rng, ComplexF64, d1 * d2, d1 * d2)
        dense_hamiltonian = raw + raw'
        dense_hamiltonian ./= norm(dense_hamiltonian)
        h = _tebd_matrix_to_tensor(dense_hamiltonian, d1, d2)
        dt = 0.173

        real_time_gate = two_site_gate(h, dt)
        @test size(real_time_gate) == (d1, d2, d1, d2)
        @test _tebd_tensor_to_matrix(real_time_gate) ≈
              exp(-im * dt * dense_hamiltonian) atol=2.0e-13 rtol=2.0e-13

        imaginary_time_gate = two_site_gate(h, dt; time=:imaginary)
        @test _tebd_tensor_to_matrix(imaginary_time_gate) ≈
              exp(-dt * dense_hamiltonian) atol=2.0e-13 rtol=2.0e-13
    end

    @testset "One- and two-site gate primitives" begin
        rng = MersenneTwister(0xa991)
        N = 3
        seed = random_mps(N, 2; maxdim=2, rng=rng, T=ComplexF64)

        one_site_matrix = randn(rng, ComplexF64, 2, 2)
        one_site = deepcopy(seed)
        one_before = dense_state(one_site)
        returned = apply_one_site_gate!(one_site, 2, one_site_matrix)
        @test returned === one_site
        @test dense_state(one_site) ≈
              _tebd_embed_one_site(one_site_matrix, 2, N, 2) * one_before atol=2.0e-12 rtol=2.0e-12

        two_site_matrix = randn(rng, ComplexF64, 4, 4)
        two_site_tensor = _tebd_matrix_to_tensor(two_site_matrix, 2, 2)
        for direction in (:left, :right)
            state = deepcopy(seed)
            before = dense_state(state)
            diagnostic = apply_two_site_gate!(
                state, 2, two_site_tensor;
                direction=direction, maxdim=16, cutoff=0.0,
            )
            expected = _tebd_embed_two_site(two_site_matrix, 2, N, 2) * before
            @test dense_state(state) ≈ expected atol=5.0e-12 rtol=5.0e-12
            @test diagnostic.discarded_weight ≈ 0 atol=5.0e-15
            @test diagnostic.kept_dim == size(state[2], 2)
            @test diagnostic.orthogonality_center ==
                  (direction === :left ? 2 : 3)
            if direction === :left
                @test right_canonical_gram(state[3]) ≈
                      Matrix{ComplexF64}(I, size(state[3], 1), size(state[3], 1)) atol=2.0e-12
            else
                @test left_canonical_gram(state[2]) ≈
                      Matrix{ComplexF64}(I, size(state[2], 2), size(state[2], 2)) atol=2.0e-12
            end
        end
    end

    @testset "Canonical layer and dense Trotter schedules" begin
        rng = MersenneTwister(0x1a9e)

        # A public layer applies exactly the selected disjoint dense gates.
        N = 5
        layer_hamiltonians = _tebd_random_hamiltonians(rng, N)
        odd_generator, _ = _tebd_layer_generators(layer_hamiltonians, N, 2)
        layer_dt = 0.041
        gates = [two_site_gate(h, layer_dt) for h in layer_hamiltonians]
        layer_state = random_mps(N, 2; maxdim=4, rng=rng, T=ComplexF64)
        layer_before = dense_state(layer_state)
        layer_diagnostic = tebd_layer!(
            layer_state, gates, 1:2:(N - 1);
            direction=:left, maxdim=32, cutoff=0.0,
        )
        @test dense_state(layer_state) ≈
              exp(-im * layer_dt * odd_generator) * layer_before atol=2.0e-11 rtol=2.0e-11
        @test length(layer_diagnostic.discarded_weights) == 2
        @test all(weight -> weight ≥ -eps(Float64),
                  layer_diagnostic.discarded_weights)

        # Check both boundary parities and all implemented product formulas.
        for N in (4, 5)
            hamiltonians = _tebd_random_hamiltonians(rng, N)
            odd, even = _tebd_layer_generators(hamiltonians, N, 2)
            initial = random_mps(N, 2; maxdim=4, rng=rng, T=ComplexF64)
            initial_dense = dense_state(initial)
            dt = 0.037
            odd_count = length(1:2:(N - 1))
            even_count = length(2:2:(N - 1))

            for order in (1, 2, 4)
                state = deepcopy(initial)
                diagnostic = tebd_step!(
                    state, hamiltonians, dt;
                    order=order, maxdim=64, cutoff=0.0,
                )
                oracle = _tebd_dense_schedule(odd, even, dt, order)
                @test dense_state(state) ≈ oracle * initial_dense atol=8.0e-11 rtol=8.0e-11
                expected_gate_count = order == 1 ? N - 1 :
                    order == 2 ? 2 * odd_count + even_count :
                    5 * (2 * odd_count + even_count)
                @test length(diagnostic.discarded_weights) == expected_gate_count
                @test length(diagnostic.kept_dimensions) == expected_gate_count
                @test diagnostic.order == order
                @test diagnostic.norm_before ≈ 1 atol=2.0e-12
                @test diagnostic.norm_after ≈ 1 atol=2.0e-10
                @test diagnostic.discarded_weight ≥ -eps(Float64)
                @test diagnostic.max_bond <= 2^(N ÷ 2)
            end
        end
    end

    @testset "Imaginary time" begin
        rng = MersenneTwister(0x1a61)
        raw = randn(rng, ComplexF64, 4, 4)
        dense_hamiltonian = raw + raw'
        dense_hamiltonian ./= norm(dense_hamiltonian)
        h = _tebd_matrix_to_tensor(dense_hamiltonian, 2, 2)
        state = random_mps(2, 2; maxdim=2, rng=rng, T=ComplexF64)
        before = dense_state(state)
        energy_before = real(dot(before, dense_hamiltonian * before))
        dt = 0.083

        diagnostic = tebd_step!(
            state, h, dt;
            order=2, time=:imaginary, maxdim=4, cutoff=0.0,
        )
        exact = exp(-dt * dense_hamiltonian) * before
        exact ./= norm(exact)
        after = dense_state(state)
        energy_after = real(dot(after, dense_hamiltonian * after))
        @test phase_aligned_error(after, exact) <= 2.0e-12
        @test energy_after <= energy_before + 2.0e-12
        @test diagnostic.norm_before ≈ 1 atol=2.0e-12
        @test diagnostic.norm_after ≈ 1 atol=2.0e-12

        # Real storage is valid when both the Hamiltonian and time parameter
        # remain real.
        X = Float64[0 1; 1 0]
        real_h = _tebd_matrix_to_tensor(kron(X, X), 2, 2)
        real_state = product_mps(2, 2, 1; T=Float64)
        tebd_step!(real_state, real_h, 0.02; time=:imaginary)
        @test all(eltype(A) == Float64 for A in real_state)
        @test overlap(real_state, real_state) ≈ 1 atol=2.0e-12
    end

    @testset "Truncation diagnostics and driver" begin
        Hgate = [1 1; 1 -1] / sqrt(2)
        identity2 = Matrix{Float64}(I, 2, 2)
        cnot = Float64[
            1 0 0 0
            0 1 0 0
            0 0 0 1
            0 0 1 0
        ]
        bell_matrix = cnot * kron(Hgate, identity2)
        bell_gate = _tebd_matrix_to_tensor(bell_matrix, 2, 2)
        bell_state = product_mps(2, 2, 1; T=Float64)
        diagnostic = apply_two_site_gate!(
            bell_state, 1, bell_gate;
            direction=:right, maxdim=1, cutoff=0.0,
        )
        @test diagnostic.kept_dim == 1
        @test diagnostic.discarded_weight ≈ 0.5 atol=2.0e-12
        @test overlap(bell_state, bell_state) ≈ 0.5 atol=2.0e-12

        rng = MersenneTwister(0x7a0c)
        N = 4
        hamiltonians = _tebd_random_hamiltonians(rng, N)
        truncated = random_mps(N, 2; maxdim=4, rng=rng, T=ComplexF64)
        step = tebd_step!(
            truncated, hamiltonians, 0.19;
            order=2, maxdim=1, cutoff=0.0,
        )
        @test step.max_bond == 1
        @test all(size(truncated[site], 2) == 1 for site in 1:(N - 1))
        @test step.discarded_weight > 0
        @test step.discarded_weight ≈ maximum(step.discarded_weights)
        @test step.total_discarded_weight ≈ sum(step.discarded_weights)
        @test all(==(1), step.kept_dimensions)

        initial = random_mps(N, 2; maxdim=2, rng=rng, T=ComplexF64)
        manual = deepcopy(initial)
        driven = deepcopy(initial)
        for _ in 1:2
            tebd_step!(manual, hamiltonians, 0.01; order=2, maxdim=8)
        end
        driver = tebd!(driven, hamiltonians, 0.01, 2; order=2, maxdim=8)
        @test dense_state(driven) ≈ dense_state(manual) atol=2.0e-11
        @test driver.steps == 2
        @test length(driver.history) == 2
        @test driver.discarded_weight ≈
              maximum(step.discarded_weight for step in driver.history)
        @test driver.total_discarded_weight ≈
              sum(step.total_discarded_weight for step in driver.history)
    end

    @testset "Validation and type safety" begin
        h = zeros(Float64, 2, 2, 2, 2)
        gate = zeros(ComplexF64, 2, 2, 2, 2)
        complex_state = product_mps(3, 2, 1; T=ComplexF64)

        @test_throws DimensionMismatch two_site_gate(zeros(2, 3, 3, 3), 0.1)
        @test_throws ArgumentError two_site_gate(h, 0.1; time=:unsupported)
        @test_throws BoundsError apply_one_site_gate!(complex_state, 0, ones(2, 2))
        @test_throws DimensionMismatch apply_one_site_gate!(
            complex_state, 1, ones(3, 3),
        )
        @test_throws BoundsError apply_two_site_gate!(complex_state, 3, gate)
        @test_throws DimensionMismatch apply_two_site_gate!(
            complex_state, 1, zeros(ComplexF64, 2, 3, 2, 3),
        )
        @test_throws ArgumentError apply_two_site_gate!(
            complex_state, 1, gate; direction=:middle,
        )
        @test_throws ArgumentError apply_two_site_gate!(
            complex_state, 1, gate; maxdim=0,
        )
        @test_throws ArgumentError apply_two_site_gate!(
            complex_state, 1, gate; cutoff=-1.0,
        )

        gates = fill(gate, 2)
        @test_throws ArgumentError tebd_layer!(
            complex_state, gates, [1, 2]; direction=:left,
        )
        @test_throws ArgumentError tebd_layer!(
            complex_state, gates, [1, 1]; direction=:left,
        )
        @test_throws DimensionMismatch tebd_layer!(
            complex_state, gates[1:1], [1]; direction=:left,
        )
        @test_throws ArgumentError tebd_layer!(
            complex_state, gates, [1]; direction=:diagonal,
        )

        @test_throws ArgumentError tebd_step!(
            deepcopy(complex_state), [h, h], 0.1; order=3,
        )
        @test_throws DimensionMismatch tebd_step!(
            deepcopy(complex_state), [h], 0.1,
        )
        @test_throws DimensionMismatch tebd_step!(
            deepcopy(complex_state), [h, zeros(2, 3, 2, 3)], 0.1,
        )
        @test_throws ArgumentError tebd!(
            deepcopy(complex_state), [h, h], 0.1, -1,
        )

        real_state = product_mps(3, 2, 1; T=Float64)
        @test_throws ArgumentError tebd_step!(real_state, [h, h], 0.1)
        @test_throws ArgumentError apply_one_site_gate!(
            real_state, 1, ComplexF64[0 1; 1 0],
        )

        single_site = product_mps(1, 2, 1; T=ComplexF64)
        @test_throws ArgumentError tebd_step!(single_site, h, 0.1)

        zero_steps = tebd!(
            deepcopy(complex_state), [h, h], 0.1, 0; order=2,
        )
        @test zero_steps.steps == 0
        @test isempty(zero_steps.history)
        @test zero_steps.discarded_weight == 0
    end
end
