using LinearAlgebra, Random, TensorNetworksJulia
include("Output.jl")
using .ExampleOutput

"""Two-site periodic Hamiltonian solved by exact physical-site blocking."""
function staggered_unit_cell(;max_bond=8,
    output=joinpath(@__DIR__,"results","unit_cell"))
    # Each bulk tensor carries the onsite field and outgoing bond coupling
    # for its position in the repeating cell.
    W1=ising_mpo(3;J=-0.8,hx=-1.1,hz=0.2,interaction=:z)[2]
    W2=ising_mpo(3;J=-1.2,hx=-0.7,hz=-0.2,interaction=:z)[2]
    H=block_mpo([W1,W2])
    ground=vumps(H,1;d=4,max_bond=max_bond,rng=MersenneTwister(128),
                 maxiter=300,tol=1e-8)
    ground.converged || @warn "Unit-cell VUMPS exhausted budget" residual=ground.residual
    Z=ComplexF64[1 0;0 -1]; I2=Matrix{ComplexF64}(I,2,2)
    rows=[(energy_per_cell=ground.energy,energy_per_site=ground.energy/2,
        z_first=real(local_expectation(ground.AL,ground.C,kron(Z,I2))),
        z_second=real(local_expectation(ground.AL,ground.C,kron(I2,Z))),
        residual=ground.residual,converged=ground.converged,
        bond_dimension=size(ground.C,1))]
    write_rows(joinpath(output,"ground.csv"),rows)
    step=infinite_tdvp_step(ground.AL,ground.AR,ground.C,H,0.01)
    println("Two-site cell: energy/site=$(ground.energy/2), converged=$(ground.converged)")
    return (ground=ground,step=step)
end

if abspath(PROGRAM_FILE)==@__FILE__
    staggered_unit_cell()
end
