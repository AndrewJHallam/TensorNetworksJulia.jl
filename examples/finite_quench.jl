using LinearAlgebra, Random, TensorNetworksJulia
include("Output.jl")
using .ExampleOutput

"""Ground state followed by the same quench with TDVP1, TDVP2, and TEBD2."""
function finite_quench(; N=12, maxdim=32, steps=10, dt=0.01,
                        output=joinpath(@__DIR__, "results", "finite"))
    X = ComplexF64[0 1; 1 0]; Z = ComplexF64[1 0; 0 -1]
    H0 = xyz_mpo(N; Jz=-1, hx=-0.7)
    psi = random_mps(N, 2; maxdim=min(maxdim, 8), rng=MersenneTwister(2026))
    ground = dmrg!(psi, H0; nsweeps=12, maxdim=maxdim, cutoff=1e-12)
    refinement = dmrg1_sweep!(psi, H0)
    write_rows(joinpath(output, "ground.csv"),
        [(sweep=r.sweep, energy=r.energy, energy_change=r.energy_change,
          local_converged=r.converged, max_residual=maximum(r.residuals))
         for r in ground.history])
    ground.converged || @warn "DMRG exhausted sweep budget" energy=ground.energy
    refinement.converged || @warn "One-site refinement did not converge"
    H = xyz_mpo(N; Jz=-1, hx=-1.1)
    bonds = nearest_neighbor_bond_hamiltonians(N, -1.1X, [Z], [Z]; couplings=-1)
    results = Dict()
    for algorithm in (:tdvp1, :tdvp2, :tebd2)
        state = deepcopy(psi)
        E0 = real(expectation(state, H))
        rows = NamedTuple[]
        for step in 0:steps
            if step > 0
                if algorithm == :tdvp1
                    tdvp1_step!(state, H, dt)
                elseif algorithm == :tdvp2
                    tdvp2_step!(state, H, dt; maxdim=maxdim, cutoff=1e-12)
                else
                    tebd_step!(state, bonds, dt; order=2, maxdim=maxdim, cutoff=1e-12)
                end
            end
            push!(rows, (time=step*dt, energy=real(expectation(state,H)),
                energy_drift=real(expectation(state,H))-E0,
                norm_squared=real(overlap(state,state)),
                centre_x=real(local_expectation(state,X,N÷2)),
                centre_z=real(local_expectation(state,Z,N÷2)),
                entropy=entanglement_entropy(state,N÷2),
                bond_dimension=maximum(size(A,2) for A in state)))
        end
        write_rows(joinpath(output, "$algorithm.csv"), rows)
        results[algorithm] = (state=state, history=rows)
        println("$algorithm: ΔE=$(last(rows).energy_drift), norm²=$(last(rows).norm_squared)")
    end
    println("Finite results: $output")
    return (ground=ground, state=psi, quenches=results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    finite_quench()
end
