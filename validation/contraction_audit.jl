# Reproduce the old orders alongside current kernels, without changing source.
using LinearAlgebra, Random, TensorOperations, TensorNetworksJulia
include("../examples/Output.jl")
using .ExampleOutput

function old_mpo_right(A,O,X)
    @tensor Y[l,lb] := A[l,r,si]*O[so,si]*conj(A[lb,rb,so])*X[r,rb]
    return Y
end
function old_right_source(A,h)
    @tensor Y[l,lb] := A[l,m,si]*A[m,r,ti]*h[so,to,si,ti]*
                       conj(A[lb,mb,so])*conj(A[mb,r,to])
    return Y
end
function old_block_density(A,L,R)
    @tensor rho[p,q] := L[l,lb]*A[l,r,p]*conj(A[lb,rb,q])*R[r,rb]
    return rho
end
function new_block_density(A,L,R)
    @tensor rho[p,q] := (L[l,lb]*A[l,r,p]*R[r,rb])*conj(A[lb,rb,q])
    return rho
end

function run_contraction_audit(;output=joinpath(@__DIR__,"results"))
    BLAS.set_num_threads(1)
    TN=TensorNetworksJulia; rng=MersenneTwister(630)
    rows=NamedTuple[]
    for chi in (8,16,32)
        A=randn(rng,ComplexF64,chi,chi,2)
        AA=randn(rng,ComplexF64,chi,chi,2,2)
        C=randn(rng,ComplexF64,chi,chi)
        O=ComplexF64[0 -im;im 0]
        X=randn(rng,ComplexF64,chi,chi)
        h=ising_two_site(J=-1,hx=-0.7,hy=0.1)
        L=randn(rng,ComplexF64,chi,5,chi); R=randn(rng,ComplexF64,chi,5,chi)
        W=xyz_mpo(3;Jz=-1,hx=-1.05)[2]
        denseW=randn(rng,ComplexF64,2,2,5,5)
        site=TN._one_site_action(A,L,W,R)
        dense_site=TN._one_site_action(A,L,denseW,R)
        pair=TN._two_site_action(AA,L,W,W,R)
        dense_pair=TN._two_site_action(AA,L,denseW,denseW,R)
        bond=TN._bond_action(C,L,R)
        block=randn(rng,ComplexF64,chi,chi,16)
        cases=(
            (:infinite_mpo_right,()->old_mpo_right(A,O,X),()->TN._mpo_right_block(A,O,X)),
            (:infinite_right_source,()->old_right_source(A,h),()->TN._right_hamiltonian_source(A,h)),
            (:block_density,()->old_block_density(block,X,X),()->new_block_density(block,X,X)),
            (:one_site_matvec,()->apply_heff1(A,L,W,R),()->site(A)),
            (:dense_one_site_matvec,()->apply_heff1(A,L,denseW,R),()->dense_site(A)),
            (:two_site_matvec,()->apply_heff2(AA,L,W,W,R),()->pair(AA)),
            (:dense_two_site_matvec,()->apply_heff2(AA,L,denseW,denseW,R),()->dense_pair(AA)),
            (:bond_matvec,()->apply_keff(C,L,R),()->bond(C)))
        for (kernel,before,after) in cases
            reference=before(); result=after()
            error=norm(result-reference)/max(norm(reference),eps(Float64))
            @assert error<1e-12
            before(); after()
            oldbytes=@allocated before(); newbytes=@allocated after()
            oldtimes=sort([(@timed before()).time for _ in 1:3])
            newtimes=sort([(@timed after()).time for _ in 1:3])
            push!(rows,(kernel=kernel,bond_dimension=chi,before_bytes=oldbytes,
                        after_bytes=newbytes,before_seconds=oldtimes[2],
                        after_seconds=newtimes[2],relative_error=error))
            println("$kernel χ=$chi: $oldbytes → $newbytes bytes, $(oldtimes[2]) → $(newtimes[2]) s, error=$error")
        end
    end
    write_rows(joinpath(output,"contraction_audit.csv"),rows)
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_contraction_audit()
end
