# Measurements for a one-site mixed-canonical infinite MPS.

function _uniform_density(AL, C)
    chi, d = _check_uniform_tensor(AL, "AL")
    size(C) == (chi, chi) || throw(DimensionMismatch("C and AL dimensions differ"))
    weight = real(sum(abs2, C))
    isfinite(weight) && weight > 0 || throw(ArgumentError("C must have finite nonzero norm"))
    return C * C' / weight, d
end

function _uniform_operator_transfer(AL, R, O)
    @tensor Y[l, lb] := AL[l, r, si] * R[r, rb] *
                       conj(AL[lb, rb, so]) * O[so, si]
    return Y
end

"""
    local_expectation(AL, C, O)

Measure `O[out,in]` in a one-site infinite MPS. `AL` must be left canonical
and `C` the consistent bond centre; its overall norm cancels. The state is
not mutated. Complex values are retained, including for non-Hermitian operators.
"""
function local_expectation(AL::AbstractArray{<:Number,3}, C::AbstractMatrix,
                           O::AbstractMatrix)
    R, d = _uniform_density(AL, C)
    size(O) == (d, d) || throw(DimensionMismatch("O must have size ($d, $d)"))
    return tr(_uniform_operator_transfer(AL, R, O))
end

"""Measure each operator in `operators` in a mixed-canonical infinite MPS."""
local_expectations(AL::AbstractArray{<:Number,3}, C::AbstractMatrix,
                   operators::AbstractVector) =
    [local_expectation(AL, C, O) for O in operators]

"""
    two_point_correlation(AL, C, O, P, distance; connected=false)

Return `<O₀ Pᵣ>` for integer `distance=r≥0`. At distance zero the ordered
product `O*P` acts on one site. With `connected=true`, subtract `<O><P>`.
Cost grows linearly with distance, without constructing a physical block.
"""
function two_point_correlation(AL::AbstractArray{<:Number,3}, C::AbstractMatrix,
                               O::AbstractMatrix, P::AbstractMatrix,
                               distance::Integer; connected::Bool=false)
    distance >= 0 || throw(ArgumentError("distance must be nonnegative"))
    R, d = _uniform_density(AL, C)
    size(O) == size(P) == (d, d) || throw(DimensionMismatch("operators and physical dimension differ"))
    if distance == 0
        value = local_expectation(AL, C, O * P)
    else
        tail = _uniform_operator_transfer(AL, R, P)
        for _ in 1:(distance - 1)
            tail = transfer_right(AL, tail)
        end
        value = tr(_uniform_operator_transfer(AL, tail, O))
    end
    return connected ? value - local_expectation(AL, C, O) *
                       local_expectation(AL, C, P) : value
end

"""
    reduced_density_matrix(AL, C, nsites=1; max_blockdim=4096)

Return a contiguous infinite-chain block density matrix in the usual `kron`
basis (last physical site varies fastest). Its storage grows as `d^(2nsites)`;
`max_blockdim` guards accidental exponential allocations.
"""
function reduced_density_matrix(AL::AbstractArray{<:Number,3}, C::AbstractMatrix,
                                nsites::Integer=1; max_blockdim::Integer=4096)
    nsites >= 1 || throw(ArgumentError("nsites must be positive"))
    max_blockdim >= 1 || throw(ArgumentError("max_blockdim must be positive"))
    R, d = _uniform_density(AL, C)
    # Avoid computing an overflowing d^nsites.
    blockdim = 1
    for _ in 1:nsites
        blockdim <= max_blockdim ÷ d || throw(ArgumentError("block exceeds max_blockdim"))
        blockdim *= d
    end
    chi = size(AL, 1)
    T = promote_type(eltype(AL), eltype(C))
    blocks = [Matrix{T}(I, chi, chi)]
    for _ in 1:nsites
        blocks = [B * view(AL, :, :, s) for B in blocks for s in 1:d]
    end
    rho = Matrix{T}(undef, blockdim, blockdim)
    for u in 1:blockdim, v in 1:blockdim
        rho[u, v] = tr(blocks[u] * R * blocks[v]')
    end
    return rho / real(tr(rho))
end

"""Normalized infinite-MPS Schmidt values: singular values of the centre `C`."""
function schmidt_values(C::AbstractMatrix; normalize::Bool=true)
    values = svdvals(C)
    weight = norm(values)
    isfinite(weight) && weight > 0 || throw(ArgumentError("C must have finite nonzero norm"))
    normalize && (values ./= weight)
    return values
end

"""Infinite-MPS bond entropy in nats, or bits with `base=2`."""
function entanglement_entropy(C::AbstractMatrix; base::Real=ℯ)
    isfinite(base) && base > 0 && base != 1 || throw(ArgumentError("invalid logarithm base"))
    probabilities = abs2.(schmidt_values(C))
    return -sum(p * log(p) for p in probabilities if p > 0) / log(base)
end

"""
    transfer_spectrum(AL; nev=2, tol=1e-12, maxiter=500, krylovdim=30)

Largest-magnitude transfer eigenvalues, normalized by the leading value.
Small bond spaces (χ²≤16) use a dense reference map; larger spaces use
matrix-free Arnoldi. For repeated eigenvalues Arnoldi may return fewer than
`nev` distinct eigenvectors; requesting a correlation length then errors
rather than reporting a spurious value.
"""
function transfer_spectrum(AL::AbstractArray{<:Number,3}; nev::Integer=2,
                           tol::Real=1e-12, maxiter::Integer=500,
                           krylovdim::Integer=30)
    chi, _ = _check_uniform_tensor(AL)
    nev >= 1 && maxiter >= 1 && krylovdim >= 2 && tol > 0 ||
        throw(ArgumentError("invalid eigensolver controls"))
    dimension = chi^2
    count = min(nev, dimension)
    if dimension <= 16
        matrix = zeros(ComplexF64, dimension, dimension)
        for column in 1:dimension
            basis = zeros(ComplexF64, chi, chi)
            basis[column] = 1
            matrix[:, column] .= vec(transfer_right(AL, basis))
        end
        values = eigvals(matrix)
    else
        seed = randn(MersenneTwister(0x7472616e), ComplexF64, chi, chi)
        values, _, info = eigsolve(X -> transfer_right(AL, X), seed, count, :LM;
            tol=tol, maxiter=maxiter, krylovdim=min(dimension, max(krylovdim, count+2)))
        info.converged >= count || error("transfer spectrum did not converge ($count eigenvalues requested)")
    end
    values = values[sortperm(abs.(values); rev=true)][1:count]
    abs(values[1]) > eps(Float64) || throw(ArgumentError("zero transfer spectral radius"))
    return values ./ values[1]
end

"""
    correlation_length(AL; kwargs...)

Longest transfer correlation length in lattice sites, `-1/log(abs(λ₂/λ₁))`.
Returns zero for a bond-dimension-one product state or a zero subleading
eigenvalue, and infinity for a degenerate leading magnitude. This length
need not be visible in every chosen observable.
"""
function correlation_length(AL::AbstractArray{<:Number,3}; kwargs...)
    chi, _ = _check_uniform_tensor(AL)
    chi == 1 && return 0.0
    values = transfer_spectrum(AL; nev=2, kwargs...)
    ratio = abs(values[2])
    ratio <= eps(Float64) && return 0.0
    ratio >= 1 - 100eps(Float64) && return Inf
    return -1 / log(ratio)
end
