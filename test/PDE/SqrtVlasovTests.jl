@testset "Square-root Vlasov representation" begin
    grid = VlasovGrid(16,16)
    layout = phase_space_layout(16,16)
    rng = MersenneTwister(825)
    # Deliberately signed g: positivity is a property of the squared density.
    g = randn(rng,16,16)
    p = project_vlasov_amplitude(g,layout;maxdim=2,target_sum_squares=sum(abs2,g))
    @test minimum(p.grid) >= 0
    @test p.grid == p.amplitude.^2
    @test sum(p.grid) ≈ sum(abs2,g) rtol=1e-13
    @test mps_phase_space(p.state,layout) ≈ p.amplitude atol=1e-12
    @test all(size(A,2)<=2 for A in p.state)
    @test_throws ArgumentError project_vlasov_amplitude(zeros(16,16),layout;maxdim=2,target_sum_squares=1)
    f0 = [(1+0.2cos(x))*exp(-v^2/2)/sqrt(2pi) for x in grid.x,v in grid.v]
    g0 = sqrt.(f0)
    evolved = evolve_vlasov_amplitude(g0,grid;dt=0.02,steps=10)
    @test sum(abs2,evolved) ≈ sum(f0) rtol=1e-13
    @test minimum(evolved.^2) >= 0
    # Signed amplitude transport is reversible for prescribed constant speed.
    kicked = TensorNetworksJulia._vlasov_fourier_advect(g,fill(0.3,16),0.1,grid.kv,2)
    @test sum(abs2,kicked) ≈ sum(abs2,g) rtol=1e-13
    back = TensorNetworksJulia._vlasov_fourier_advect(kicked,fill(0.3,16),-0.1,grid.kv,2)
    @test back ≈ g atol=1e-12
    # Amplitude gauge g -> -g gives the same field and density trajectory.
    @test vlasov_amplitude_step(-g0,grid,0.02) ≈ -vlasov_amplitude_step(g0,grid,0.02) atol=1e-12
    equilibrium = [exp(-v^2/4) for x in grid.x,v in grid.v]
    @test evolve_vlasov_amplitude(equilibrium,grid;dt=0.02,steps=3) ≈ equilibrium atol=1e-12
    times = Float64[]
    drive = (x,t)->(push!(times,t);0.1sin(x)*cos(t))
    result = vlasov_sqrt_scarfinder(f0,grid;maxdim=2,dt=0.02,
        steps_per_iteration=2,iterations=2,time=1.,drive=drive)
    @test sort(unique(times)) ≈ [1.01,1.03,1.05,1.07,1.09,1.11]
    @test result.time ≈ 1.08
    @test result.grid == result.amplitude.^2
    @test mps_phase_space(result.state,layout) ≈ result.amplitude atol=1e-12
    @test all(h.negative_mass==0 for h in result.history)
    @test all(isapprox(h.mass,grid.dx*grid.dv*sum(f0);rtol=1e-13) for h in result.history)
    @test result.cycle_error ≈ norm(result.unprojected_cycle-result.grid)/norm(result.grid)
    @test_throws ArgumentError vlasov_sqrt_scarfinder(-f0,grid;maxdim=2,dt=0.02,steps_per_iteration=2)
    @test_throws ArgumentError evolve_vlasov_amplitude(g0,grid;dt=0.02,steps=0,nyquist=:bad)
    # Dropping Nyquist removes its frozen checkerboard; mass loss is visible.
    checker = [(-1.)^i for i in 0:15,j in 1:16]
    removed = TensorNetworksJulia._vlasov_fourier_advect(checker,fill(0.3,16),0.1,grid.kx,1;nyquist=:drop)
    @test norm(removed) < 1e-12
    # Independent analytic free-streaming solution checks continuum equivalence.
    for n in (32,64)
        fine = VlasovGrid(n,n)
        a = [(1+0.1cos(x))*exp(-v^2/2)/sqrt(2pi) for x in fine.x,v in fine.v]
        amplitude_shift = TensorNetworksJulia._vlasov_fourier_advect(sqrt.(a),fine.v,0.13,fine.kx,1)
        exact = [(1+0.1cos(x-v*0.13))*exp(-v^2/2)/sqrt(2pi) for x in fine.x,v in fine.v]
        @test amplitude_shift.^2 ≈ exact atol=1e-12
    end
end
