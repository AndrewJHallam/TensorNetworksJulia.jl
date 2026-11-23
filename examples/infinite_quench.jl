using LinearAlgebra, Random, TensorNetworksJulia
include("Output.jl")
using .ExampleOutput

"""Adaptive VUMPS followed by a uniform-TDVP field quench."""
function infinite_quench(; max_bond=8, steps=10, dt=0.005,
                          output=joinpath(@__DIR__, "results", "infinite"))
    X = ComplexF64[0 1; 1 0]; Z = ComplexF64[1 0; 0 -1]
    h0 = ising_two_site(J=-1, hx=-0.7)
    ground = vumps(h0, 1; rng=MersenneTwister(2026), max_bond=max_bond,
                   maxiter=300, tol=1e-8)
    ground.converged || @warn "VUMPS exhausted iteration budget" residual=ground.residual
    write_rows(joinpath(output, "ground.csv"),
        [(iteration=i, energy=r.energy, residual=r.residual)
         for (i,r) in enumerate(ground.history)])
    h = ising_two_site(J=-1, hx=-1.1)
    AL, AR, C = ground.AL, ground.AR, ground.C
    initial_energy = real(uniform_energy(AL, AR, C, h))
    rows = NamedTuple[]
    for step in 0:steps
        if step > 0
            evolved = infinite_tdvp_step(AL, AR, C, h, dt)
            evolved.krylov_converged && evolved.environments_converged ||
                @warn "Inner TDVP solve failed" step
            AL, AR, C = evolved.AL, evolved.AR, evolved.C
        end
        residuals = infinite_residuals(AL, AR, C)
        push!(rows, (time=step*dt, energy=real(uniform_energy(AL,AR,C,h)),
            energy_drift=real(uniform_energy(AL,AR,C,h))-initial_energy,
            x=real(local_expectation(AL,C,X)), z=real(local_expectation(AL,C,Z)),
            entropy=entanglement_entropy(C), correlation_length=correlation_length(AL),
            zz_connected=real(two_point_correlation(AL,C,Z,Z,5; connected=true)),
            canonical_residual=max(residuals.left_canonical,residuals.right_canonical),
            centre_residual=residuals.center_consistency, bond_dimension=size(C,1)))
    end
    write_rows(joinpath(output, "tdvp.csv"), rows)
    println("Infinite TDVP: Δe=$(last(rows).energy_drift), centre residual=$(last(rows).centre_residual)")
    println("Infinite results: $output")
    return (ground=ground, AL=AL, AR=AR, C=C, history=rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    infinite_quench()
end
