#= Finite MPS compression and MPO application.
Finite-MPS compression and MPO application.

Compression first moves the orthogonality centre to the left boundary and
then performs a left-to-right sequence of Schmidt truncations.  Consequently
each local discarded weight is relative to the norm entering that cut, and
the cumulative discarded weight is `1 - prod(1 - discarded_weights)`.
=#

# ---------------------------------------------------------------------------
# Validation and small helpers
# ---------------------------------------------------------------------------

function _check_compression_options(maxdim::Integer, cutoff::Real)
    maxdim >= 1 || throw(ArgumentError("maxdim must be at least one"))
    (isfinite(cutoff) && cutoff >= 0) || throw(ArgumentError(
        "cutoff must be finite and non-negative",
    ))
    return nothing
end

function _compression_absorb_right(
    A::AbstractArray{<:Number,3},
    C::AbstractMatrix,
)
    size(A, 2) == size(C, 1) || throw(DimensionMismatch(
        "cannot absorb a $(size(C)) matrix into the right bond of $(size(A))",
    ))
    @tensor AC[l, rnew, s] := A[l, rold, s] * C[rold, rnew]
    return AC
end

function _compression_norm_squared(mps::AbstractVector)
    value = real(overlap(mps, mps))
    return Float64(value)
end

function _compression_bond_dimensions(mps::AbstractVector)
    return [size(mps[site], 2) for site in 1:(length(mps) - 1)]
end

function _check_mpo_application(mpo::AbstractVector, mps::AbstractVector)
    _check_mps(mps)
    _check_mpo(mpo)
    length(mpo) == length(mps) || throw(DimensionMismatch(
        "MPS and MPO lengths differ: $(length(mps)) and $(length(mpo))",
    ))
    for site in eachindex(mps)
        input_dimension = size(mpo[site], 2)
        physical_dimension = size(mps[site], 3)
        input_dimension == physical_dimension || throw(DimensionMismatch(
            "MPO input dimension at site $site is $input_dimension; " *
            "the MPS physical dimension is $physical_dimension",
        ))
    end
    return nothing
end

function _apply_mpo_site(
    W::AbstractArray{<:Number,4},
    A::AbstractArray{<:Number,3},
)
    size(W, 2) == size(A, 3) || throw(DimensionMismatch(
        "MPO input and MPS physical dimensions differ",
    ))
    @tensor joined[lpsi, rpsi, lw, rw, sout] :=
        A[lpsi, rpsi, sin] * W[sout, sin, lw, rw]
    joined = permutedims(joined, (1, 3, 2, 4, 5))
    return reshape(
        joined,
        size(A, 1) * size(W, 3),
        size(A, 2) * size(W, 4),
        size(W, 1),
    )
end

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

"""
    compress!(psi; maxdim, cutoff=0, normalize=false)

Compress an open-boundary MPS in place.  The input may be in an arbitrary
gauge.  A right-canonical sweep first moves the centre to site 1; a subsequent
left-to-right SVD sweep truncates every bond according to the hard `maxdim`
and relative discarded-weight `cutoff`.

Scale is preserved by default.  With `normalize=true`, the retained state is
normalized after truncation; a zero state then raises `ArgumentError` before
`psi` is mutated.

The returned named tuple contains `state` (identical to `psi`), per-bond and
cumulative discarded weights, maximum discarded weight, final bond
dimensions, and squared norms before truncation, after truncation, and after
the optional normalization.
"""
function compress!(
    psi::AbstractVector;
    maxdim::Integer,
    cutoff::Real=0,
    normalize::Bool=false,
)
    _check_mps(psi)
    _check_compression_options(maxdim, cutoff)

    norm_squared_before = _compression_norm_squared(psi)
    if normalize && !(isfinite(norm_squared_before) && norm_squared_before > 0)
        throw(ArgumentError("cannot normalize a zero or non-finite MPS"))
    end

    N = length(psi)
    T = _promoted_eltype(psi)

    # Move the centre to the left boundary without truncation.  This makes
    # every suffix to the right of the active bond an orthonormal basis.
    C = ones(T, 1, 1)
    for site in N:-1:2
        centre = _compression_absorb_right(psi[site], C)
        C, AR = right_orthogonalize(centre)
        psi[site] = AR
    end
    psi[1] = _compression_absorb_right(psi[1], C)

    discarded_weights = Float64[]
    sizehint!(discarded_weights, max(N - 1, 0))

    # Since the prefix is left isometric and the untouched suffix is right
    # isometric, each SVD is a genuine Schmidt truncation at that cut.
    for site in 1:(N - 1)
        centre = contract_two_sites(psi[site], psi[site + 1])
        AL, AC, discarded = split_left(
            centre; maxdim=maxdim, cutoff=cutoff,
        )
        psi[site] = AL
        psi[site + 1] = AC
        push!(discarded_weights, Float64(discarded))
    end

    norm_squared_after_truncation = _compression_norm_squared(psi)
    cumulative_discarded = isempty(discarded_weights) ? 0.0 :
        1 - prod(1 - weight for weight in discarded_weights)
    cumulative_discarded = clamp(cumulative_discarded, 0.0, 1.0)
    maximum_discarded = isempty(discarded_weights) ? 0.0 :
        maximum(discarded_weights)

    if normalize
        normalize!(psi)
    end
    norm_squared_after = _compression_norm_squared(psi)

    return (
        state=psi,
        discarded_weight=cumulative_discarded,
        max_discarded_weight=maximum_discarded,
        discarded_weights=discarded_weights,
        bond_dimensions=_compression_bond_dimensions(psi),
        norm_squared_before=norm_squared_before,
        norm_squared_after_truncation=norm_squared_after_truncation,
        norm_squared_after=norm_squared_after,
        normalized=normalize,
    )
end

# ---------------------------------------------------------------------------
# MPO application
# ---------------------------------------------------------------------------

"""
    apply_mpo(mpo, psi; maxdim=typemax(Int), cutoff=0, normalize=false)

Apply an open-boundary MPO to an MPS without mutating either input.  The exact
local contraction first produces bonds of dimension
`mps_bond * mpo_bond`; `compress!` then controls the retained dimensions.
Only each MPO input dimension must match the corresponding MPS physical
dimension, so rectangular local maps and site-dependent physical dimensions
are supported.

The result is a named tuple whose `state` field is the output MPS.  It also
contains all `compress!` diagnostics, the input norm, and the enlarged bond
dimensions before compression.  Scale is preserved unless `normalize=true`.
"""
function apply_mpo(
    mpo::AbstractVector,
    psi::AbstractVector;
    maxdim::Integer=typemax(Int),
    cutoff::Real=0,
    normalize::Bool=false,
)
    _check_mpo_application(mpo, psi)
    _check_compression_options(maxdim, cutoff)

    T = promote_type(_promoted_eltype(mpo), _promoted_eltype(psi))
    output = Vector{Array{T,3}}(undef, length(psi))
    for site in eachindex(psi)
        output[site] = _apply_mpo_site(mpo[site], psi[site])
    end

    enlarged_bond_dimensions = _compression_bond_dimensions(output)
    input_norm_squared = _compression_norm_squared(psi)
    diagnostics = compress!(
        output; maxdim=maxdim, cutoff=cutoff, normalize=normalize,
    )
    return merge(diagnostics, (
        input_norm_squared=input_norm_squared,
        enlarged_bond_dimensions=enlarged_bond_dimensions,
    ))
end
