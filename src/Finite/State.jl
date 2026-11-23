# Open-boundary finite-MPS construction, overlap, and normalization.

import LinearAlgebra: normalize!

function _check_mps(mps::AbstractVector; name::AbstractString="mps")
    isempty(mps) && throw(ArgumentError("$name must contain at least one site"))

    for n in eachindex(mps)
        A = mps[n]
        _check_rank(A, 3, "$name[$n]")
        all(>(0), size(A)) || throw(DimensionMismatch(
            "$name[$n] has an empty axis: $(size(A))"))
    end

    size(first(mps), 1) == 1 || throw(DimensionMismatch(
        "$name must have an open left boundary of dimension 1"))
    size(last(mps), 2) == 1 || throw(DimensionMismatch(
        "$name must have an open right boundary of dimension 1"))

    for n in 1:(length(mps) - 1)
        size(mps[n], 2) == size(mps[n + 1], 1) ||
            throw(DimensionMismatch(
                "$name bond $n has dimensions $(size(mps[n], 2)) and " *
                "$(size(mps[n + 1], 1))"))
    end
    return nothing
end

"""
    product_mps(states, d; T=Float64)

Construct an open-boundary product MPS from one-based local basis labels.
Every site tensor has shape `(1, 1, d)`.
"""
function product_mps(states::AbstractVector{<:Integer}, d::Integer;
                     T::Type{<:Number}=Float64)
    isempty(states) && throw(ArgumentError("states must not be empty"))
    d >= 1 || throw(ArgumentError("local dimension d must be positive"))
    _is_floating_scalar_type(T) || throw(ArgumentError(
        "T must be a real or complex floating-point type"))
    all(s -> 1 <= s <= d, states) || throw(ArgumentError(
        "basis labels must lie in 1:$d"))

    mps = Vector{Array{T,3}}(undef, length(states))
    for n in eachindex(states)
        A = zeros(T, 1, 1, d)
        A[1, 1, states[n]] = one(T)
        mps[n] = A
    end
    return mps
end

function product_mps(states::AbstractVector{<:Integer};
                     d::Integer=isempty(states) ? 0 : maximum(states),
                     T::Type{<:Number}=Float64)
    return product_mps(states, d; T=T)
end

"""
    product_mps(local_states; normalize=true)

Construct a product MPS from explicit local state vectors.  Local dimensions
may vary from site to site.  By default each local vector is normalized.
"""
function product_mps(local_states::AbstractVector{<:AbstractVector};
                     normalize::Bool=true)
    isempty(local_states) && throw(ArgumentError("local_states must not be empty"))
    all(v -> !isempty(v), local_states) || throw(ArgumentError(
        "local state vectors must not be empty"))
    T = float(_promoted_eltype(local_states))
    mps = Vector{Array{T,3}}(undef, length(local_states))

    for n in eachindex(local_states)
        vector = T.(local_states[n])
        if normalize
            vector_norm = norm(vector)
            iszero(vector_norm) && throw(ArgumentError(
                "local state $n has zero norm"))
            vector ./= vector_norm
        end
        mps[n] = Array(reshape(vector, 1, 1, length(vector)))
    end
    return mps
end

"""Construct `N` copies of the same one-based local basis state."""
function product_mps(N::Integer, d::Integer, state::Integer=1;
                     T::Type{<:Number}=Float64)
    N >= 1 || throw(ArgumentError("N must be positive"))
    return product_mps(fill(state, N), d; T=T)
end

"""
    random_mps(N, d; maxdim=1, rng=Random.default_rng(), T=ComplexF64)

Construct a normalized random open-boundary MPS.  The bond after site `n` is
bounded by `min(maxdim, d^n, d^(N-n))`.  The returned state is left canonical.
"""
function random_mps(N::Integer, d::Integer;
                    maxdim::Integer=1,
                    rng::AbstractRNG=Random.default_rng(),
                    T::Type{<:Number}=ComplexF64)
    N >= 1 || throw(ArgumentError("N must be positive"))
    d >= 1 || throw(ArgumentError("local dimension d must be positive"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    _is_floating_scalar_type(T) || throw(ArgumentError(
        "T must be a real or complex floating-point type"))

    bonds = Vector{Int}(undef, N + 1)
    bonds[1] = 1
    bonds[end] = 1
    for n in 1:(N - 1)
        left_capacity = _bounded_power(d, n, maxdim)
        right_capacity = _bounded_power(d, N - n, maxdim)
        bonds[n + 1] = min(maxdim, left_capacity, right_capacity)
    end

    mps = Vector{Array{T,3}}(undef, N)
    for n in 1:N
        mps[n] = _random_array(rng, T, bonds[n], bonds[n + 1], d)
    end
    return normalize!(mps)
end

function random_mps(N::Integer, d::Integer, maxdim::Integer; kwargs...)
    return random_mps(N, d; maxdim=maxdim, kwargs...)
end

"""Return the Hilbert-space overlap `⟨left|right⟩`."""
function overlap(left::AbstractVector, right::AbstractVector)
    _check_mps(left; name="left")
    _check_mps(right; name="right")
    length(left) == length(right) || throw(DimensionMismatch(
        "MPS lengths differ: $(length(left)) and $(length(right))"))

    T = _promoted_eltype(left, right)
    environment = ones(T, 1, 1)
    for n in eachindex(left)
        A = left[n]
        B = right[n]
        size(A, 3) == size(B, 3) || throw(DimensionMismatch(
            "physical dimensions differ at site $n"))
        size(environment, 1) == size(A, 1) || throw(DimensionMismatch(
            "left overlap bond mismatch at site $n"))
        size(environment, 2) == size(B, 1) || throw(DimensionMismatch(
            "right overlap bond mismatch at site $n"))
        @tensor next_environment[r, rb] := environment[l, lb] *
                                              conj(A[l, r, s]) *
                                              B[lb, rb, s]
        environment = next_environment
    end
    size(environment) == (1, 1) || throw(DimensionMismatch(
        "overlap did not close to a scalar; final size is $(size(environment))"))
    return environment[1, 1]
end

"""
    normalize!(mps)

Bring an open-boundary MPS into left-canonical form and normalize it.  Site
arrays are replaced in `mps`; the vector itself is returned.
"""
function normalize!(mps::AbstractVector{<:AbstractArray{<:Number,3}})
    _check_mps(mps)
    norm_squared = real(overlap(mps, mps))
    (isfinite(norm_squared) && norm_squared > 0) || throw(ArgumentError(
        "cannot normalize an MPS with non-positive or non-finite norm"))

    C = ones(eltype(first(mps)), 1, 1)
    for n in eachindex(mps)
        centre = _absorb_left(C, mps[n])
        C, canonical_site = left_orthogonalize(centre)
        mps[n] = canonical_site
    end
    size(C) == (1, 1) || throw(DimensionMismatch(
        "normalization did not terminate at a scalar boundary"))
    return mps
end
