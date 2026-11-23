@testset "Original MATLAB Vlasov fixtures" begin
    folder = joinpath(@__DIR__, "..", "fixtures", "vlasov_matlab")
    function read_fixture(name)
        rows = [parse.(Float64, split(line, ',')) for line in readlines(joinpath(folder, name))]
        return reduce(vcat, [reshape(row, 1, :) for row in rows])
    end
    f0 = read_fixture("initial.csv")
    expected = read_fixture("evolved.csv")
    grid = VlasovGrid(16, 16)
    layout = phase_space_layout(16, 16)
    evolved = evolve_vlasov(f0, grid; dt=0.02, steps=5)
    @test evolved ≈ expected atol=2e-13 rtol=2e-13
    p = project_phase_space(evolved, layout; maxdim=2)
    @test p.grid ≈ read_fixture("raw_projection.csv") atol=2e-12 rtol=2e-12
    # Original clipping, mass scaling, and momentum-shift correction.
    # Independently assembled here; the production filter does not do this.
    corrected = max.(p.grid, 0)
    corrected .*= sum(f0) / sum(corrected)
    target_mean = sum(f0 .* reshape(grid.v, 1, :)) / sum(f0)
    current_mean = sum(corrected .* reshape(grid.v, 1, :)) / sum(corrected)
    corrected = TensorNetworksJulia._vlasov_fourier_advect(corrected,
        fill(target_mean - current_mean, 16), 1., grid.kv, 2)
    corrected = max.(corrected, 0)
    corrected .*= sum(f0) / sum(corrected)
    @test corrected ≈ read_fixture("corrected_projection.csv") atol=2e-12 rtol=2e-12
end
