# The layout is a permutation plus local dimensions. No physical approximation
# occurs until the SVD in phase_space_mps.
struct PhaseSpaceLayout
    shape::Tuple{Int,Int}
    permutation::Vector{Int} # ordered vector entry -> Julia vec(grid) entry
    dimensions::Vector{Int}
end

"""
    phase_space_layout(nx, nv; ordering=:quadtree, significance=:fine_to_coarse)

Dyadic phase-space tensorization. Site 1 is the FASTEST tensor index (Julia
column-major convention). `:xfast`, `:vfast`, `:snake`, and `:morton` have
binary sites. `:quadtree` has dimension-4 sites and requires nx == nv.
Morton interleaves x and v bits; quadtree groups each pair into s = xbit +
2vbit + 1. The default significance reproduces the MATLAB INSE mapping.
Coarse-to-fine reverses the *tensor sites*, leaving each site's labels intact.
"""
function phase_space_layout(nx::Integer, nv::Integer;
                            ordering::Symbol=:quadtree,
                            significance::Symbol=:fine_to_coarse)
    nx >= 2 && nv >= 2 && ispow2(nx) && ispow2(nv) ||
        throw(ArgumentError("nx and nv must be powers of two, at least 2"))
    ordering in (:xfast, :vfast, :snake, :morton, :quadtree) ||
        throw(ArgumentError("unknown phase-space ordering: $ordering"))
    significance in (:fine_to_coarse, :coarse_to_fine) ||
        throw(ArgumentError("unknown bit significance: $significance"))
    ordering == :quadtree && nx != nv &&
        throw(ArgumentError("quadtree requires nx == nv"))
    bx, bv = trailing_zeros(nx), trailing_zeros(nv)
    dims = ordering == :quadtree ? fill(4, bx) : fill(2, bx + bv)
    permutation = Vector{Int}(undef, nx * nv)
    for iv in 0:(nv - 1), ix in 0:(nx - 1)
        q = if ordering == :xfast
            ix + nx * iv
        elseif ordering == :vfast
            iv + nv * ix
        elseif ordering == :snake
            (iseven(iv) ? ix : nx - 1 - ix) + nx * iv
        else
            # Morton code, including rectangular dyadic grids.
            code, position = 0, 0
            for bit in 0:(max(bx, bv) - 1)
                if bit < bx
                    code |= ((ix >> bit) & 1) << position
                    position += 1
                end
                if bit < bv
                    code |= ((iv >> bit) & 1) << position
                    position += 1
                end
            end
            code
        end
        if significance == :coarse_to_fine
            # Reverse mixed-radix digits, not individual bits within d=4 sites.
            reversed = 0
            for d in dims
                reversed = reversed * d + mod(q, d)
                q = div(q, d)
            end
            q = reversed
        end
        permutation[q + 1] = ix + 1 + nx * iv
    end
    return PhaseSpaceLayout((Int(nx), Int(nv)), permutation, dims)
end

function flatten_phase_space(f::AbstractMatrix, layout::PhaseSpaceLayout)
    size(f) == layout.shape || throw(DimensionMismatch("grid and layout differ"))
    return vec(f)[layout.permutation]
end

function unflatten_phase_space(y::AbstractVector, layout::PhaseSpaceLayout)
    length(y) == prod(layout.shape) || throw(DimensionMismatch("vector and layout differ"))
    f = Matrix{eltype(y)}(undef, layout.shape...)
    vec(f)[layout.permutation] = y
    return f
end

"""
    phase_space_mps(f, layout; maxdim=typemax(Int), cutoff=0)

Successive SVD (TT-SVD) of the ordered values of f itself, preserving scale.
At each cut keep at most maxdim singular values; cutoff is the relative
squared tail allowance at that cut. Returns state and discarded-weight
diagnostics. This is a sequential approximation, not a globally optimal fit.
Tensors have the package convention (left, right, physical).
"""
function phase_space_mps(f::AbstractMatrix{<:Number}, layout::PhaseSpaceLayout;
                         maxdim::Integer=typemax(Int), cutoff::Real=0)
    _check_compression_options(maxdim, cutoff)
    all(isfinite, f) || throw(ArgumentError("distribution must be finite"))
    carry = float.(flatten_phase_space(f, layout))
    T = eltype(carry)
    state = Vector{Array{T,3}}(undef, length(layout.dimensions))
    weights = Float64[]
    left = 1
    for site in 1:(length(state) - 1)
        d = layout.dimensions[site]
        matrix = reshape(carry, left * d, :)
        F = svd(matrix; full=false)
        rank, discarded = _svd_rank(F.S, maxdim, cutoff)
        state[site] = permutedims(reshape(F.U[:, 1:rank], left, d, rank), (1, 3, 2))
        carry = Diagonal(F.S[1:rank]) * F.Vt[1:rank, :]
        push!(weights, Float64(discarded))
        left = rank
    end
    state[end] = permutedims(reshape(carry, left, layout.dimensions[end], 1), (1, 3, 2))
    return (state=state, discarded_weights=weights,
            discarded_weight=clamp(1 - prod(1 .- weights), 0.0, 1.0),
            bond_dimensions=[size(A, 2) for A in state[1:end-1]])
end

"""Reconstruct the dense phase-space grid; site 1 varies fastest."""
function mps_phase_space(state::AbstractVector, layout::PhaseSpaceLayout)
    _check_mps(state)
    length(state) == length(layout.dimensions) ||
        throw(DimensionMismatch("MPS and layout lengths differ"))
    all(size(state[s], 3) == layout.dimensions[s] for s in eachindex(state)) ||
        throw(DimensionMismatch("MPS and layout physical dimensions differ"))
    T = _promoted_eltype(state)
    carry = ones(T, 1, 1)
    for A in state
        rows = size(carry, 1)
        next = Matrix{T}(undef, rows * size(A, 3), size(A, 2))
        for s in axes(A, 3)
            next[((s - 1) * rows + 1):(s * rows), :] = carry * A[:, :, s]
        end
        carry = next
    end
    return unflatten_phase_space(vec(carry), layout)
end

"""
    project_phase_space(f, layout; maxdim, cutoff=0, target_sum=nothing)

Truncate f directly. Optional scalar mass restoration scales one MPS core,
so it preserves the rank bound. No positivity clipping, momentum shifting,
or L2 normalization is performed. `relative_error` includes mass restoration;
`discarded_weight` measures the SVD loss before that rescaling.
"""
function project_phase_space(f::AbstractMatrix{<:Real}, layout::PhaseSpaceLayout;
                             maxdim::Integer, cutoff::Real=0, target_sum=nothing)
    result = phase_space_mps(f, layout; maxdim=maxdim, cutoff=cutoff)
    grid = mps_phase_space(result.state, layout)
    scale = 1.0
    if target_sum !== nothing
        isfinite(target_sum) && target_sum > 0 ||
            throw(ArgumentError("target_sum must be finite and positive"))
        retained_sum = sum(grid)
        isfinite(retained_sum) && retained_sum > 0 ||
            throw(ArgumentError("truncated distribution has non-positive mass"))
        scale = target_sum / retained_sum
        result.state[end] .*= scale
        grid .*= scale
    end
    relative_error = norm(grid - f) / max(norm(f), eps(Float64))
    return merge(result, (grid=grid, mass_scale=scale, relative_error=relative_error))
end
