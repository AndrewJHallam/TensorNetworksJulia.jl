@testset "Phase-space permutation and SVD" begin
    rng = MersenneTwister(182)
    for shape in ((8, 8), (8, 4)), ordering in (:xfast, :vfast, :snake, :morton, :quadtree),
        significance in (:fine_to_coarse, :coarse_to_fine)
        ordering == :quadtree && shape[1] != shape[2] && continue
        f = randn(rng, shape...)
        layout = phase_space_layout(shape...; ordering=ordering, significance=significance)
        @test sort(layout.permutation) == collect(1:length(f))
        @test unflatten_phase_space(flatten_phase_space(f, layout), layout) == f
        exact = phase_space_mps(f, layout)
        @test mps_phase_space(exact.state, layout) ≈ f atol=1e-12
        truncated = phase_space_mps(f, layout; maxdim=2)
        reconstructed = mps_phase_space(truncated.state, layout)
        @test all(d <= 2 for d in truncated.bond_dimensions)
        @test norm(f - reconstructed)^2 / norm(f)^2 ≈ truncated.discarded_weight atol=1e-12
        @test all(left_canonical_gram(A) ≈ I for A in truncated.state[1:end-1])
    end

    # Independent MATLAB bit formula, plus direct contraction at selected grid points.
    layout = phase_space_layout(8, 8)
    f = randn(rng, 8, 8)
    psi = phase_space_mps(f, layout).state
    for iv in 0:7, ix in 0:7
        q = sum((((ix >> b) & 1) + 2 * ((iv >> b) & 1)) * 4^b for b in 0:2)
        @test layout.permutation[q + 1] == ix + 1 + 8iv
        row = ones(1, 1)
        for b in 0:2
            s = 1 + ((ix >> b) & 1) + 2 * ((iv >> b) & 1)
            row = row * psi[b + 1][:, :, s]
        end
        @test only(row) ≈ f[ix + 1, iv + 1] atol=1e-12
    end
    @test phase_space_layout(4, 4; ordering=:xfast).permutation == collect(1:16)
    @test phase_space_layout(4, 4; ordering=:snake).permutation[5:8] == [8, 7, 6, 5]
    @test_throws ArgumentError phase_space_layout(6, 8)
    @test_throws ArgumentError phase_space_layout(8, 4)
    @test_throws ArgumentError phase_space_layout(8, 8; ordering=:bad)

    positive = rand(rng, 8, 8) .+ 0.1
    projected = project_phase_space(positive, layout; maxdim=2, target_sum=sum(positive))
    @test sum(projected.grid) ≈ sum(positive) atol=1e-12
    @test mps_phase_space(projected.state, layout) ≈ projected.grid atol=1e-12
    @test all(size(A, 2) <= 2 for A in projected.state)
    @test_throws ArgumentError project_phase_space(zeros(8, 8), layout; maxdim=2, target_sum=1)
    @test_throws ArgumentError phase_space_mps(fill(NaN, 8, 8), layout)
    @test_throws DimensionMismatch flatten_phase_space(zeros(4, 4), layout)
    @test_throws ArgumentError phase_space_mps(f, layout; maxdim=0)
    one = reshape([1., 2., 3., 4.], 2, 2)
    l1 = phase_space_layout(2, 2)
    @test mps_phase_space(phase_space_mps(one, l1; maxdim=1).state, l1) == one
end

@testset "Vlasov field, transport, and driving" begin
    grid = VlasovGrid(32, 32; Lx=2pi, vmax=6)
    f = [(1 + 0.1cos(2x)) / (length(grid.v) * grid.dv) for x in grid.x, v in grid.v]
    @test vlasov_field(f, grid) ≈ 0.05sin.(2grid.x) atol=1e-14
    @test maximum(abs, vlasov_field(fill(1., 32, 32), grid)) < 1e-14
    # Exact grid-cell translation independently checks the transport sign and wrap.
    pattern = reshape(collect(1.:128), 16, 8)
    shifted = TensorNetworksJulia._vlasov_advect(pattern, fill(2., 8), 1., 1., 1)
    @test shifted == circshift(pattern, (2, 0))
    shifted = TensorNetworksJulia._vlasov_advect(pattern, fill(-1., 16), 1., 1., 2)
    @test shifted == circshift(pattern, (0, -1))
    tiny = TensorNetworksJulia._vlasov_advect(pattern, fill(eps(), 16), 1., 1., 2)
    @test tiny ≈ pattern atol=1e-12

    equilibrium = [exp(-v^2/2) for x in grid.x, v in grid.v]
    @test evolve_vlasov(equilibrium, grid; dt=0.05, steps=5) ≈ equilibrium atol=1e-12
    driven = vlasov_step(equilibrium, grid, 0.1; drive=(x,t) -> 0.3)
    @test vlasov_diagnostics(driven, grid).momentum > 0
    @test sum(driven) ≈ sum(equilibrium) rtol=1e-13
    @test minimum(driven) >= -1e-8 # Fourier interpolation may overshoot zero.
    linear = vlasov_step(equilibrium, grid, 0.1; drive=(x,t)->0.3, advection=:linear)
    @test minimum(linear) >= 0
    @test sum(linear) ≈ sum(equilibrium) rtol=1e-13

    # Continuously increasing time: no reset between iterations or in the probe.
    times = Float64[]
    drive = (x,t) -> (push!(times, t); 0.1cos(x)*sin(t))
    r = vlasov_scarfinder(equilibrium, grid; maxdim=4, dt=0.05,
                          steps_per_iteration=2, iterations=2, time=1., drive=drive)
    @test sort(unique(times)) ≈ [1.025, 1.075, 1.125, 1.175, 1.225, 1.275]
    @test r.time ≈ 1.2
    @test length(r.history) == 2
    @test all(abs(h.mass - r.initial_diagnostics.mass) < 1e-10 for h in r.history)
    @test all(h.max_bond <= 4 for h in r.history)
    @test mps_phase_space(r.state, r.layout) ≈ r.grid atol=1e-12
    @test r.cycle_error ≈ norm(r.unprojected_cycle - r.grid) / norm(r.grid)
    @test vlasov_diagnostics([-1. 2.; 3. 4.], VlasovGrid(2, 2)).negative_mass > 0
    @test_throws ArgumentError VlasovGrid(8, 8; vmax=0)
    @test_throws ArgumentError vlasov_step(equilibrium, grid, -0.1)
    @test_throws ArgumentError vlasov_step(equilibrium, grid, 0.1; drive=(x,t)->NaN)
    @test_throws ArgumentError vlasov_step(equilibrium, grid, 0.1; advection=:unknown)
end

@testset "Fourier translation and independent DFT reference" begin
    rng = MersenneTwister(490)
    # Direct DFT implementation is independent of FFTW and the package field solve.
    function reference_shift(y, shift, period)
        n = length(y)
        h = n ÷ 2
        modes = iseven(n) ? [collect(0:h-1); 0; collect(1-h:-1)] :
                            [collect(0:h); collect(-h:-1)]
        forward = [sum(y[j+1] * exp(-2pi*im*m*j/n) for j in 0:n-1) for m in 0:n-1]
        shifted = forward .* exp.(-im .* (2pi/period .* modes) .* shift)
        return [real(sum(shifted[m+1] * exp(2pi*im*m*j/n) for m in 0:n-1)) / n for j in 0:n-1]
    end
    for n in (7, 8, 16)
        f = randn(rng, n, n)
        speeds = randn(rng, n)
        grid = VlasovGrid(n, n)
        xshift = TensorNetworksJulia._vlasov_fourier_advect(f, speeds, 0.17, grid.kx, 1)
        vshift = TensorNetworksJulia._vlasov_fourier_advect(f, speeds, 0.17, grid.kv, 2)
        @test xshift ≈ hcat([reference_shift(f[:,j], speeds[j]*0.17, 2pi) for j in 1:n]...) atol=1e-12
        @test vshift ≈ reduce(vcat, [reshape(reference_shift(f[i,:], speeds[i]*0.17, 12), 1, :) for i in 1:n]) atol=1e-12
        @test sum(xshift) ≈ sum(f) atol=1e-12
        @test norm(xshift) ≈ norm(f) atol=1e-12
        @test norm(vshift) ≈ norm(f) atol=1e-12
        back = TensorNetworksJulia._vlasov_fourier_advect(xshift, speeds, -0.17, grid.kx, 1)
        @test back ≈ f atol=1e-12
    end
    grid = VlasovGrid(32, 32)
    f = [(1 + 0.2cos(x))*exp(-v^2/2) for x in grid.x, v in grid.v]
    evolved = evolve_vlasov(f, grid; dt=0.02, steps=10)
    @test norm(evolved) ≈ norm(f) rtol=1e-13
    @test sum(evolved) ≈ sum(f) rtol=1e-13
    # A resolved sinusoid has the exact analytic translation sign.
    line = sin.(3grid.x)
    translated = TensorNetworksJulia._vlasov_fourier_advect(reshape(line, :, 1), [0.7], 0.2, grid.kx, 1)
    @test vec(translated) ≈ sin.(3 .* (grid.x .- 0.14)) atol=1e-12
    # Nyquist is intentionally held fixed, precisely as in the MATLAB helper.
    nyquist = reshape([(-1.)^i for i in 0:31], :, 1)
    @test TensorNetworksJulia._vlasov_fourier_advect(nyquist, [0.7], 0.2, grid.kx, 1) ≈ nyquist atol=1e-12
end
