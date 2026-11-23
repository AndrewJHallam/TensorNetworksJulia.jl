# Reproducible convergence checks. Run from the project: julia --project=. validation/accuracy.jl
using LinearAlgebra, Random, TensorNetworksJulia
include("../examples/Output.jl")
using .ExampleOutput

function phase_distance(a,b)
    a=a/norm(a); b=b/norm(b)
    overlap=dot(b,a)
    phase=iszero(overlap) ? one(overlap) : overlap/abs(overlap)
    return norm(a-phase*b)
end

function small_dense_state(psi)
    N=length(psi)
    values=zeros(ComplexF64,2^N)
    for index in 0:2^N-1
        bond=ones(ComplexF64,1,1)
        for n in 1:N
            s=((index >> (N-n)) & 1)+1
            bond=bond*psi[n][:,:,s]
        end
        values[index+1]=only(bond)
    end
    return values
end

# Analytic thermodynamic Ising integral, approximated with periodic trapezoids.
function ising_reference(g; points=65536)
    return -sum(sqrt(1+g^2-2g*cos(2pi*j/points)) for j in 0:points-1)/points
end

function run_accuracy(; output=joinpath(@__DIR__,"results"), full=false)
    rows=NamedTuple[]
    N=8; X=ComplexF64[0 1;1 0]; Z=ComplexF64[1 0;0 -1]
    H=xyz_mpo(N;Jz=-1,hx=-0.9)
    bonds=nearest_neighbor_bond_hamiltonians(N,-0.9X,[Z],[Z];couplings=-1)
    initial=random_mps(N,2;maxdim=16,rng=MersenneTwister(391))
    # Exact reference is confined to the small eight-site chain.
    total=0.16; reference=exp(-im*total*Hermitian(mpo_matrix(H)))*small_dense_state(initial)
    for algorithm in (:tdvp1,:tdvp2,:tebd2,:tebd4), dt in (0.04,0.02,0.01)
        psi=deepcopy(initial)
        E0=real(expectation(psi,H))
        for _ in 1:round(Int,total/dt)
            if algorithm==:tdvp1
                tdvp1_step!(psi,H,dt)
            elseif algorithm==:tdvp2
                tdvp2_step!(psi,H,dt;maxdim=16)
            else
                tebd_step!(psi,bonds,dt;order=algorithm==:tebd2 ? 2 : 4,maxdim=16)
            end
        end
        push!(rows,(algorithm=algorithm,dt=dt,state_error=phase_distance(reference,small_dense_state(psi)),
            energy_drift=abs(real(expectation(psi,H))-E0),norm_error=abs(real(overlap(psi,psi))-1)))
    end
    write_rows(joinpath(output,"timestep.csv"),rows)
    # Finite chain large enough to avoid a dense Hilbert-space calculation.
    N=full ? 64 : 32; caps=full ? (16,32,64) : (8,16,32)
    finite_rows=NamedTuple[]
    for cap in caps
        psi=random_mps(N,2;maxdim=8,rng=MersenneTwister(138))
        H=xyz_mpo(N;Jz=-1,hx=-1.05)
        ground=dmrg!(psi,H;nsweeps=12,maxdim=cap,cutoff=1e-12,energy_tol=1e-9)
        push!(finite_rows,(N=N,maxdim=cap,energy_per_site=ground.energy/N,
            converged=ground.converged,sweeps=ground.sweeps,
            actual_bond=maximum(size(A,2) for A in psi),
            discarded_weight=maximum(r.discarded_weight for r in ground.history),
            local_residual=maximum(last(ground.history).residuals),
            entropy=entanglement_entropy(psi,N÷2)))
    end
    write_rows(joinpath(output,"finite_bond.csv"),finite_rows)
    infinite_rows=NamedTuple[]; g=1.3; exact=ising_reference(g)
    for cap in (full ? (4,8,16,32) : (4,8,16))
        ground=vumps(ising_two_site(J=-1,hx=-g),1;rng=MersenneTwister(911),
                     max_bond=cap,maxiter=300,tol=1e-8)
        push!(infinite_rows,(max_bond=cap,energy=ground.energy,reference=exact,
            energy_error=abs(ground.energy-exact),converged=ground.converged,
            residual=ground.residual,iterations=ground.iterations,
            entropy=entanglement_entropy(ground.C),correlation_length=correlation_length(ground.AL)))
    end
    write_rows(joinpath(output,"infinite_bond.csv"),infinite_rows)
    # Validate integrators against dense evolution and explicit refinement.
    for algorithm in (:tdvp1,:tdvp2,:tebd2,:tebd4)
        series=filter(r->r.algorithm==algorithm,rows)
        all(r->r.norm_error<1e-8,series) || error("$algorithm lost norm")
        last(series).state_error < 1e-4 || error("$algorithm failed dense comparison")
        if algorithm==:tebd2
            last(series).state_error < first(series).state_error/8 || error("TEBD2 failed timestep refinement")
        elseif algorithm==:tebd4
            last(series).state_error < first(series).state_error/64 || error("TEBD4 failed timestep refinement")
        end
    end
    last(finite_rows).energy_per_site <= first(finite_rows).energy_per_site+1e-8 || error("finite bond refinement increased energy")
    all(r->r.converged,finite_rows) || error("finite DMRG budget exhausted; inspect finite_bond.csv")
    last(infinite_rows).energy_error < 1e-5 || error("VUMPS missed exact Ising energy")
    all(r->r.converged,infinite_rows) || error("VUMPS budget exhausted; inspect infinite_bond.csv")
    println("Accuracy checks passed. Results: $output")
    return (timesteps=rows,finite=finite_rows,infinite=infinite_rows)
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_accuracy(;full="--full" in ARGS)
end
