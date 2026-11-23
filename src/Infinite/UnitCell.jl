# Exact small-unit-cell support by grouping physical sites into a super-site.

"""
    block_mps(tensors; max_physicaldim=256)

Contract a periodic cell's MPS tensors into one uniform tensor. Adjacent
virtual dimensions must match, and the final bond must close onto the first.
The composite physical index uses the usual `kron` ordering (last site fast).
Pass the result to `canonicalize_uniform`. Storage grows exponentially with
the cell length; this adapter is intended for small cells.
"""
function block_mps(tensors::AbstractVector; max_physicaldim::Integer=256)
    isempty(tensors) && throw(ArgumentError("a unit cell cannot be empty"))
    max_physicaldim >= 1 || throw(ArgumentError("max_physicaldim must be positive"))
    physical = 1
    for (i,A) in enumerate(tensors)
        _check_rank(A, 3, "cell[$i]")
        all(>(0),size(A)) || throw(DimensionMismatch("empty tensor axis"))
        physical <= max_physicaldim ÷ size(A,3) || throw(ArgumentError("cell exceeds max_physicaldim"))
        physical *= size(A,3)
        next = tensors[mod1(i+1,length(tensors))]
        size(A,2) == size(next,1) || throw(DimensionMismatch("unit-cell bond $i does not close"))
    end
    block = copy(first(tensors))
    for A in tensors[2:end]
        @tensor merged[l,r,p,s] := block[l,m,p] * A[m,r,s]
        block = reshape(permutedims(merged,(1,2,4,3)),size(block,1),size(A,2),size(block,3)*size(A,3))
    end
    return block
end

"""
    block_mpo(cell; max_physicaldim=256)

Group a periodic vector of `TriangularMPO`s (or bulk rank-four tensors) into
one Hamiltonian MPO for a super-site. Each site must share the same auxiliary
channels and their meaning. Site-dependent fields and couplings are allowed.
The resulting MPO works with `vumps` and `infinite_tdvp_step`, including bond
growth. Reported energies are per cell: divide by the cell length for energy
per original site. Evolution time is unchanged.

This exact blocking costs a physical dimension `prod(dᵢ)`. It supports small
periodic cells, but does not replace a scalable native multi-site algorithm.
"""
function block_mpo(cell::AbstractVector; max_physicaldim::Integer=256)
    isempty(cell) && throw(ArgumentError("a unit cell cannot be empty"))
    max_physicaldim >= 1 || throw(ArgumentError("max_physicaldim must be positive"))
    models = [H isa TriangularMPO ? H : TriangularMPO(H) for H in cell]
    physical = 1
    auxiliary = size(first(models).W,3)
    for H in models
        size(H.W,3) == auxiliary || throw(DimensionMismatch("MPO auxiliary dimensions differ"))
        physical <= max_physicaldim ÷ size(H.W,1) || throw(ArgumentError("cell exceeds max_physicaldim"))
        physical *= size(H.W,1)
    end
    block = copy(first(models).W)
    for H in models[2:end]
        W = H.W
        @tensor merged[po,so,pi,si,a,b] := block[po,pi,a,m] * W[so,si,m,b]
        d = size(block,1)*size(W,1)
        block = reshape(permutedims(merged,(2,1,4,3,5,6)),d,d,auxiliary,auxiliary)
    end
    return TriangularMPO(block)
end
