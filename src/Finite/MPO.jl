# Open-boundary finite-MPO validation, environments, and dense conversion.

function _check_mpo(mpo::AbstractVector; name::AbstractString="mpo")
    isempty(mpo) && throw(ArgumentError("$name must contain at least one site"))

    for n in eachindex(mpo)
        W = mpo[n]
        _check_rank(W, 4, "$name[$n]")
        all(>(0), size(W)) || throw(DimensionMismatch(
            "$name[$n] has an empty axis: $(size(W))"))
    end

    size(first(mpo), 3) == 1 || throw(DimensionMismatch(
        "$name must have an open left MPO boundary of dimension 1"))
    size(last(mpo), 4) == 1 || throw(DimensionMismatch(
        "$name must have an open right MPO boundary of dimension 1"))

    for n in 1:(length(mpo) - 1)
        size(mpo[n], 4) == size(mpo[n + 1], 3) ||
            throw(DimensionMismatch(
                "$name bond $n has dimensions $(size(mpo[n], 4)) and " *
                "$(size(mpo[n + 1], 3))"))
    end
    return nothing
end

function _check_mps_mpo(mps::AbstractVector, mpo::AbstractVector)
    _check_mps(mps)
    _check_mpo(mpo)
    length(mps) == length(mpo) || throw(DimensionMismatch(
        "MPS and MPO lengths differ: $(length(mps)) and $(length(mpo))"))

    for n in eachindex(mps)
        d = size(mps[n], 3)
        size(mpo[n], 1) == d || throw(DimensionMismatch(
            "MPO output dimension at site $n is $(size(mpo[n], 1)); expected $d"))
        size(mpo[n], 2) == d || throw(DimensionMismatch(
            "MPO input dimension at site $n is $(size(mpo[n], 2)); expected $d"))
    end
    return nothing
end

"""
    left_environment(FL, A, W)

Absorb one MPS/MPO site into a left Hamiltonian environment.  `FL` and the
result use the axis order `(ket_bond, mpo_bond, bra_bond)`.
"""
function left_environment(FL::AbstractArray, A::AbstractArray, W::AbstractArray)
    _check_rank(FL, 3, "FL")
    _check_rank(A, 3, "A")
    _check_rank(W, 4, "W")
    size(FL, 1) == size(A, 1) || throw(DimensionMismatch(
        "FL ket bond and A left bond differ"))
    size(FL, 3) == size(A, 1) || throw(DimensionMismatch(
        "FL bra bond and A left bond differ"))
    size(FL, 2) == size(W, 3) || throw(DimensionMismatch(
        "FL MPO bond and W left bond differ"))
    size(W, 1) == size(A, 3) || throw(DimensionMismatch(
        "W output physical dimension and A physical dimension differ"))
    size(W, 2) == size(A, 3) || throw(DimensionMismatch(
        "W input physical dimension and A physical dimension differ"))

    @tensor FLnew[r, wr, rb] := FL[l, wl, lb] *
                                A[l, r, sin] *
                                W[sout, sin, wl, wr] *
                                conj(A[lb, rb, sout])
    return FLnew
end

"""Absorb one MPS/MPO site into a right Hamiltonian environment."""
function right_environment(A::AbstractArray, W::AbstractArray, FR::AbstractArray)
    _check_rank(A, 3, "A")
    _check_rank(W, 4, "W")
    _check_rank(FR, 3, "FR")
    size(FR, 1) == size(A, 2) || throw(DimensionMismatch(
        "FR ket bond and A right bond differ"))
    size(FR, 3) == size(A, 2) || throw(DimensionMismatch(
        "FR bra bond and A right bond differ"))
    size(FR, 2) == size(W, 4) || throw(DimensionMismatch(
        "FR MPO bond and W right bond differ"))
    size(W, 1) == size(A, 3) || throw(DimensionMismatch(
        "W output physical dimension and A physical dimension differ"))
    size(W, 2) == size(A, 3) || throw(DimensionMismatch(
        "W input physical dimension and A physical dimension differ"))

    # Absorb FR before joining ket and bra. Joining A and conj(A) first
    # constructs a chi^4 MPO transfer tensor instead of a chi^2 workspace.
    @tensor FRnew[l, wl, lb] := (A[l, r, sin] * FR[r, wr, rb]) *
                                W[sout, sin, wl, wr] *
                                conj(A[lb, rb, sout])
    return FRnew
end

"""
    left_environments(mps, mpo)

Return `N+1` left environments.  Entry `L[n]` contains sites `1:n-1`, so
the effective problem at site `n` uses `L[n]`.
"""
function left_environments(mps::AbstractVector, mpo::AbstractVector)
    _check_mps_mpo(mps, mpo)
    N = length(mps)
    T = _promoted_eltype(mps, mpo)
    environments = Vector{Array{T,3}}(undef, N + 1)
    environments[1] = ones(T, 1, 1, 1)
    for n in 1:N
        environments[n + 1] = left_environment(environments[n], mps[n], mpo[n])
    end
    return environments
end

"""
    right_environments(mps, mpo)

Return `N+1` right environments.  Entry `R[n]` contains sites `n:N`, so
the effective problem at site `n` uses `R[n+1]`.
"""
function right_environments(mps::AbstractVector, mpo::AbstractVector)
    _check_mps_mpo(mps, mpo)
    N = length(mps)
    T = _promoted_eltype(mps, mpo)
    environments = Vector{Array{T,3}}(undef, N + 1)
    environments[N + 1] = ones(T, 1, 1, 1)
    for n in N:-1:1
        environments[n] = right_environment(mps[n], mpo[n], environments[n + 1])
    end
    return environments
end

"""Return the scalar expectation value `⟨mps|mpo|mps⟩`."""
function expectation(mps::AbstractVector, mpo::AbstractVector)
    environments = left_environments(mps, mpo)
    final_environment = environments[end]
    size(final_environment) == (1, 1, 1) || throw(DimensionMismatch(
        "expectation did not close to a scalar; final size is " *
        "$(size(final_environment))"))
    return final_environment[1, 1, 1]
end

"""
    mpo_matrix(mpo)

Contract an open-boundary MPO to a dense matrix.  Site 1 is the most
significant Kronecker factor.  This routine is intended for small-system
verification, not large calculations.
"""
function mpo_matrix(mpo::AbstractVector)
    _check_mpo(mpo)
    T = _promoted_eltype(mpo)
    first_site = mpo[1]
    blocks = Vector{Matrix{T}}(undef, size(first_site, 4))
    for right_bond in axes(first_site, 4)
        blocks[right_bond] = Matrix{T}(view(first_site, :, :, 1, right_bond))
    end

    for n in 2:length(mpo)
        W = mpo[n]
        length(blocks) == size(W, 3) || throw(DimensionMismatch(
            "MPO contraction bond mismatch before site $n"))
        old_rows, old_columns = size(first(blocks))
        new_blocks = [zeros(T, old_rows * size(W, 1),
                            old_columns * size(W, 2))
                      for _ in axes(W, 4)]

        for left_bond in axes(W, 3), right_bond in axes(W, 4)
            local_operator = Matrix{T}(view(W, :, :, left_bond, right_bond))
            new_blocks[right_bond] .+= kron(blocks[left_bond], local_operator)
        end
        blocks = new_blocks
    end

    length(blocks) == 1 || throw(DimensionMismatch(
        "MPO contraction did not terminate at a scalar right bond"))
    return only(blocks)
end
