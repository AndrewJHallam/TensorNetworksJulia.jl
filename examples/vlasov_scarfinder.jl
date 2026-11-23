using TensorNetworksJulia
include("Output.jl")
using .ExampleOutput

"""Small two-stream experiment. Change ordering, rank, and drive here."""
function vlasov_example(; levels=6, maxdim=4, ordering=:quadtree,
                         significance=:fine_to_coarse,
                         dt=0.02, steps_per_iteration=50, iterations=50,
                         drive_amplitude=0.0, drive_period=1.0,
                         advection=:fourier,
                         output=joinpath(@__DIR__, "results", "vlasov", string(advection),
                                         string(ordering), "chi$(maxdim)"))
    grid = VlasovGrid(2^levels, 2^levels; Lx=2pi, vmax=6)
    layout = phase_space_layout(length(grid.x), length(grid.v);
                                ordering=ordering, significance=significance)
    width, speed, alpha = 0.4, 2.4, 0.2
    f0 = [(1 + alpha*cos(x)) *
          (exp(-(v - speed)^2 / (2width^2)) + exp(-(v + speed)^2 / (2width^2))) /
          (2sqrt(2pi)*width) for x in grid.x, v in grid.v]
    drive = (x,t) -> drive_amplitude * sin(x) * cos(2pi*t/drive_period)
    result = vlasov_scarfinder(f0, grid; layout=layout, maxdim=maxdim,
        dt=dt, steps_per_iteration=steps_per_iteration, iterations=iterations,
        drive=drive, advection=advection)
    if !isempty(result.history)
        write_rows(joinpath(output, "iterations.csv"), result.history)
    end
    # Save the final state and the unprojected probe endpoint for inspection.
    write_rows(joinpath(output, "final_grid.csv"),
        [(x=grid.x[i], v=grid.v[j], f=result.grid[i,j], probe=result.unprojected_cycle[i,j])
         for j in eachindex(grid.v) for i in eachindex(grid.x)])
    # Same integrator with no rank filter.
    control = evolve_vlasov(f0, grid; dt=dt,
                            steps=iterations*steps_per_iteration, drive=drive, advection=advection)
    write_rows(joinpath(output, "unfiltered_control.csv"),
        [(x=grid.x[i], v=grid.v[j], f=control[i,j])
         for j in eachindex(grid.v) for i in eachindex(grid.x)])
    println("Unprojected final-cycle error: ", result.cycle_error)
    println("Final diagnostics: ", vlasov_diagnostics(result.grid, grid))
    println("Vlasov results: ", output)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    vlasov_example()
end
