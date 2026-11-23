# Timings include state copying but exclude compilation and initial preparation.
using LinearAlgebra, Random, TensorNetworksJulia
include("../examples/Output.jl")
using .ExampleOutput

function measure_case(operation; samples=3)
    operation() # compile/warm before measuring
    runs=[@timed(operation()) for _ in 1:samples]
    times=sort([r.time for r in runs]); bytes=sort([r.bytes for r in runs])
    return (seconds=times[cld(samples,2)],bytes=bytes[cld(samples,2)],
            gc_seconds=sum(r.gctime for r in runs)/samples)
end

function run_benchmarks(;full=false,output=joinpath(@__DIR__,"results"))
    # Fixed thread count for repeatability; metadata records it explicitly.
    BLAS.set_num_threads(1)
    N=full ? 64 : 32; caps=full ? (16,32,64) : (16,32)
    X=ComplexF64[0 1;1 0]; Z=ComplexF64[1 0;0 -1]
    rows=NamedTuple[]
    for cap in caps
        H=xyz_mpo(N;Jz=-1,hx=-1.05)
        bonds=nearest_neighbor_bond_hamiltonians(N,-1.05X,[Z],[Z];couplings=-1)
        psi=random_mps(N,2;maxdim=cap,rng=MersenneTwister(591))
        operations=(
            dmrg1=()->dmrg1_sweep!(deepcopy(psi),H),
            dmrg2=()->dmrg2_sweep!(deepcopy(psi),H;maxdim=cap),
            tdvp1=()->tdvp1_step!(deepcopy(psi),H,0.01),
            tdvp2=()->tdvp2_step!(deepcopy(psi),H,0.01;maxdim=cap),
            tebd2=()->tebd_step!(deepcopy(psi),bonds,0.01;order=2,maxdim=cap),
            apply_mpo=()->apply_mpo(H,psi;maxdim=cap))
        for (name,operation) in pairs(operations)
            result=measure_case(operation)
            push!(rows,merge((algorithm=name,N=N,bond_dimension=cap),result))
            println("$name, N=$N, χ=$cap: $(result.seconds)s, $(result.bytes) bytes")
        end
    end
    for cap in (full ? (8,16,32) : (4,8))
        state=random_infinite_mps(cap,2;rng=MersenneTwister(671))
        h=ising_two_site(J=-1,hx=-1.3)
        mpo=TriangularMPO(ising_mpo(3;J=-1,hx=-1.3,interaction=:z)[2])
        lr=ExponentialSumHamiltonian(-1.3X,[Z],[Z],reshape([-1.0],1,1),reshape([0.4],1,1))
        for (label,hamiltonian) in ((:nearest,h),(:mpo,mpo),(:longrange,lr))
            for algorithm in (:vumps,:tdvp)
                operation=algorithm==:vumps ?
                    ()->vumps_step(state.AL,state.AR,state.C,hamiltonian) :
                    ()->infinite_tdvp_step(state.AL,state.AR,state.C,hamiltonian,0.005)
                result=measure_case(operation)
                name=Symbol("infinite_",label,"_",algorithm)
                push!(rows,merge((algorithm=name,N=0,bond_dimension=cap),result))
                println("$name, χ=$cap: $(result.seconds)s, $(result.bytes) bytes")
            end
        end
    end
    write_rows(joinpath(output,"benchmarks.csv"),rows)
    open(joinpath(output,"machine.txt"),"w") do io
        println(io,"Julia: ",VERSION,"\nCPU: ",Sys.CPU_NAME,"\nJulia threads: ",Threads.nthreads(),
            "\nBLAS threads: ",BLAS.get_num_threads(),"\nBLAS: ",BLAS.get_config(),
            "\nMode: ",full ? "full" : "standard",
            "\nThree samples per case, median elapsed seconds and allocated bytes.",
            "\nIncludes copies of mutated states; excludes compilation/preparation.",
            "\nTimings are measurements, not numerical-convergence certifications.")
    end
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_benchmarks(;full="--full" in ARGS)
end
