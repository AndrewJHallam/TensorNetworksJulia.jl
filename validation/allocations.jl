# Investigate contraction intermediates separately from complete sweeps.
using LinearAlgebra, Random, TensorOperations, TensorNetworksJulia
include("../examples/Output.jl")
using .ExampleOutput

function unoptimized_right_environment(A,W,FR)
    @tensor out[l,wl,lb] := A[l,r,si] * W[so,si,wl,wr] *
                           conj(A[lb,rb,so]) * FR[r,wr,rb]
    return out
end

function staged_heff1(A,FL,W,FR)
    @tensor left[wl,lo,ri,si] := FL[li,wl,lo] * A[li,ri,si]
    @tensor centre[lo,ri,so,wr] := left[wl,lo,ri,si] * W[so,si,wl,wr]
    @tensor out[lo,ro,so] := centre[lo,ri,so,wr] * FR[ri,wr,ro]
    return out
end

function staged_heff2(A,FL,W1,W2,FR)
    @tensor left[w0,lo,ri,s1i,s2i] := FL[li,w0,lo] * A[li,ri,s1i,s2i]
    @tensor centre[lo,ri,s1o,w1,s2i] := left[w0,lo,ri,s1i,s2i] * W1[s1o,s1i,w0,w1]
    @tensor right[lo,ri,s1o,s2o,w2] := centre[lo,ri,s1o,w1,s2i] * W2[s2o,s2i,w1,w2]
    @tensor out[lo,ro,s1o,s2o] := right[lo,ri,s1o,s2o,w2] * FR[ri,w2,ro]
    return out
end

function staged_right_environment(A,W,FR)
    @tensor tail[l,si,wr,rb] := A[l,r,si] * FR[r,wr,rb]
    @tensor weighted[l,rb,so,wl] := tail[l,si,wr,rb] * W[so,si,wl,wr]
    @tensor out[l,wl,lb] := weighted[l,rb,so,wl] * conj(A[lb,rb,so])
    return out
end

function measure_allocations()
    BLAS.set_num_threads(1)
    rng=MersenneTwister(192)
    rows=NamedTuple[]
    for chi in (16,32)
        A=randn(rng,ComplexF64,chi,chi,2)
        AA=randn(rng,ComplexF64,chi,chi,2,2)
        FL=randn(rng,ComplexF64,chi,5,chi)
        FR=randn(rng,ComplexF64,chi,5,chi)
        W=xyz_mpo(3;Jz=-1,hx=-1.05)[2]
        for (label,original,staged) in (
            (:one_site,()->apply_heff1(A,FL,W,FR),()->staged_heff1(A,FL,W,FR)),
            (:two_site,()->apply_heff2(AA,FL,W,W,FR),()->staged_heff2(AA,FL,W,W,FR)),
            (:right_environment,()->unoptimized_right_environment(A,W,FR),()->right_environment(A,W,FR)))
            reference=original(); alternate=staged()
            @assert isapprox(reference,alternate;rtol=1e-12)
            # Warm both functions, then measure one action, without a sweep or copies.
            original(); staged()
            oldbytes=@allocated original()
            stagedbytes=@allocated staged()
            push!(rows,(bond_dimension=chi,kernel=label,baseline_bytes=oldbytes,staged_bytes=stagedbytes))
            println("χ=$chi $label: baseline=$oldbytes bytes, staged=$stagedbytes bytes per call")
        end
    end
    write_rows(joinpath(@__DIR__,"results","allocations.csv"),rows)
end

if abspath(PROGRAM_FILE)==@__FILE__
    measure_allocations()
end
