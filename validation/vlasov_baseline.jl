using LinearAlgebra, TensorNetworksJulia
include("../examples/Output.jl")
using .ExampleOutput

# Kept in validation only: reproduce MATLAB's extra post-SVD corrections.
# Clipping and the velocity translation may increase ranks; this grid is
# deliberately not advertised as a maxdim-MPS.
function matlab_corrected_projection(f, layout, maxdim, grid, target_mass, target_momentum)
    p = project_phase_space(f, layout; maxdim=maxdim)
    corrected = max.(p.grid, 0)
    corrected .*= target_mass / vlasov_diagnostics(corrected, grid).mass
    shift = target_momentum / target_mass -
            vlasov_diagnostics(corrected, grid).momentum / target_mass
    corrected = TensorNetworksJulia._vlasov_fourier_advect(
        corrected, fill(shift, length(grid.x)), 1., grid.kv, 2)
    corrected = max.(corrected, 0)
    corrected .*= target_mass / vlasov_diagnostics(corrected, grid).mass
    return corrected
end

function read_numeric_matrix(path; skiprows=0)
    rows = [parse.(Float64, split(line, ',')) for line in readlines(path)[skiprows+1:end] if !isempty(line)]
    return reduce(vcat, [reshape(row, 1, :) for row in rows])
end

"""Follow the final state for five cycles without projections or corrections."""
function unprojected_probe(f, grid; dt, steps_per_iteration, cycles=5, advection=:fourier)
    current = copy(f)
    initial = vlasov_diagnostics(f, grid)
    rows = NamedTuple[]
    for step in 0:(cycles * steps_per_iteration)
        if step > 0
            current = vlasov_step(current, grid, dt; time=(step - 1)*dt, advection=advection)
        end
        diagnostics = vlasov_diagnostics(current, grid)
        push!(rows, merge((step=step, time=step*dt,
            relative_change=norm(current - f)/norm(f),
            relative_mass_drift=(diagnostics.mass - initial.mass)/initial.mass,
            relative_l2_drift=norm(current)/norm(f) - 1), diagnostics))
    end
    return (grid=current, history=rows,
        cycle_error=rows[steps_per_iteration + 1].relative_change,
        maximum_motion=maximum(row.relative_change for row in rows[1:steps_per_iteration+1]),
        five_cycle_error=last(rows).relative_change)
end

"""Match MATLAB's two-stream baseline; compare raw MPS filters and legacy corrections."""
function vlasov_baseline(; levels=7, iterations=100, ranks=(4, 8, 16),
                         dt=0.01, steps_per_iteration=100, alpha=0.4, width=0.75, speed=2.4,
                         output=joinpath(@__DIR__, "results", "vlasov_baseline"))
    grid = VlasovGrid(2^levels, 2^levels)
    layout = phase_space_layout(2^levels, 2^levels)
    f0 = [(1 + alpha*cos(x)) * (exp(-0.5*((v-speed)/width)^2) +
          exp(-0.5*((v+speed)/width)^2)) / (2sqrt(2pi)*width) for x in grid.x, v in grid.v]
    initial = vlasov_diagnostics(f0, grid)
    summaries = NamedTuple[]
    states = Dict{String,Matrix{Float64}}()
    mkpath(output)
    # Every variant uses the same PDE grid, interval, and original distribution.
    for (label, rank, advection) in [("fourier_chi$(r)", r, :fourier) for r in ranks]
        result = vlasov_scarfinder(f0, grid; maxdim=rank, layout=layout, dt=dt,
            steps_per_iteration=steps_per_iteration, iterations=iterations, advection=advection)
        write_rows(joinpath(output, label, "iterations.csv"), result.history)
        states[label] = result.grid
    end
    # Direct comparison with the previous interpolation, at the same resolution.
    linear = vlasov_scarfinder(f0, grid; maxdim=first(ranks), dt=dt,
        steps_per_iteration=steps_per_iteration, iterations=iterations, advection=:linear)
    write_rows(joinpath(output, "linear_chi$(first(ranks))", "iterations.csv"), linear.history)
    states["linear_chi$(first(ranks))"] = linear.grid

    # Independent saved MATLAB initial state includes its initial corrections.
    saved_initial_path = joinpath(output, "matlab_saved_initial.csv")
    saved_final_path = joinpath(output, "matlab_saved_final.csv")
    have_saved = isfile(saved_initial_path) && isfile(saved_final_path)
    if have_saved
        params = parse.(Float64, split(readlines(joinpath(output,"matlab_saved_parameters.csv"))[2], ','))
        expected = [levels, 2pi, 6., dt, steps_per_iteration, iterations, first(ranks), alpha, 2, speed, width]
        all(isapprox.(params, expected)) || error("Saved MATLAB parameters do not match this baseline")
    end
    corrected = have_saved ? read_numeric_matrix(saved_initial_path) :
        matlab_corrected_projection(f0, layout, first(ranks), grid, initial.mass, initial.momentum)
    corrected_rows = NamedTuple[]
    for iteration in 1:iterations
        evolved = evolve_vlasov(corrected, grid; dt=dt, steps=steps_per_iteration)
        next = matlab_corrected_projection(evolved, layout, first(ranks), grid,
                                            initial.mass, initial.momentum)
        push!(corrected_rows, merge((iteration=iteration, time=iteration*dt*steps_per_iteration,
            step_error=norm(next - corrected)/norm(corrected)), vlasov_diagnostics(next, grid)))
        corrected = next
    end
    write_rows(joinpath(output, "matlab_corrected", "iterations.csv"), corrected_rows)
    states["matlab_corrected"] = corrected
    if have_saved
        saved_final = read_numeric_matrix(saved_final_path)
        comparison = norm(corrected - saved_final) / norm(saved_final)
        write_rows(joinpath(output, "matlab_full_run_parity.csv"), [(relative_final_error=comparison,)])
        println("Saved MATLAB full-run relative error: ", comparison)
    end
    states["unfiltered"] = evolve_vlasov(f0, grid; dt=dt, steps=iterations*steps_per_iteration)
    for label in ["fourier_chi$(r)" for r in ranks] ∪ ["linear_chi$(first(ranks))", "matlab_corrected", "unfiltered"]
        final = states[label]
        advection = startswith(label, "linear") ? :linear : :fourier
        probe = unprojected_probe(final, grid; dt=dt, steps_per_iteration=steps_per_iteration,
                                   advection=advection)
        halfdt_endpoint = evolve_vlasov(final, grid; dt=dt/2,
                                         steps=2steps_per_iteration, advection=advection)
        normal_endpoint = evolve_vlasov(final, grid; dt=dt,
                                         steps=steps_per_iteration, advection=advection)
        d = vlasov_diagnostics(final, grid)
        push!(summaries, (variant=label, cycle_error=probe.cycle_error,
            maximum_motion=probe.maximum_motion, five_cycle_error=probe.five_cycle_error,
            endpoint_dt_halving=norm(halfdt_endpoint - normal_endpoint)/norm(final),
            mass=d.mass, negative_mass_fraction=d.negative_mass/d.mass,
            minimum=d.minimum, field_l2=d.field_l2,
            relative_energy_change=(d.energy - initial.energy)/initial.energy,
            relative_l2_change=norm(final)/norm(f0) - 1,
            velocity_edge_max=d.velocity_edge_max))
        write_rows(joinpath(output, label, "unprojected_probe.csv"), probe.history)
        write_rows(joinpath(output, label, "final_grid.csv"),
            [(x=grid.x[i], v=grid.v[j], f=final[i,j]) for j in eachindex(grid.v) for i in eachindex(grid.x)])
        println(label, ": cycle_error=", probe.cycle_error,
                ", maximum_motion=", probe.maximum_motion,
                ", negative_mass_fraction=", d.negative_mass/d.mass)
    end
    write_rows(joinpath(output,"summary.csv"), summaries)
    return (summary=summaries, states=states)
end

if abspath(PROGRAM_FILE) == @__FILE__
    vlasov_baseline()
end
