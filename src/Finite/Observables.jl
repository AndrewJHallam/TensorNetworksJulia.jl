#= Finite MPS observables and reduced-state diagnostics.
Gauge-independent observables for open-boundary finite MPS.

The routines in this file deliberately do not assume that the input is in a
particular canonical form.  This differs from several of the compact MATLAB
measurement helpers, whose contractions are correct only when the omitted
left or right norm environment is an identity.
=#

# ---------------------------------------------------------------------------
# Norm environments and validation
# ---------------------------------------------------------------------------

function _finite_norm_left_environment(
    environment::AbstractMatrix,
    A::AbstractArray{<:Number,3},
)
    size(environment, 1) == size(A, 1) || throw(DimensionMismatch(
        "left norm environment and MPS ket bond differ",
    ))
    size(environment, 2) == size(A, 1) || throw(DimensionMismatch(
        "left norm environment and MPS bra bond differ",
    ))
    @tensor next_environment[r, rb] := environment[l, lb] *
        A[l, r, s] * conj(A[lb, rb, s])
    return next_environment
end

function _finite_norm_right_environment(
    A::AbstractArray{<:Number,3},
    environment::AbstractMatrix,
)
    size(environment, 1) == size(A, 2) || throw(DimensionMismatch(
        "right norm environment and MPS ket bond differ",
    ))
    size(environment, 2) == size(A, 2) || throw(DimensionMismatch(
        "right norm environment and MPS bra bond differ",
    ))
    @tensor next_environment[l, lb] := (A[l, r, s] * environment[r, rb]) *
        conj(A[lb, rb, s])
    return next_environment
end

function _finite_norm_environments(mps::AbstractVector)
    _check_mps(mps)
    N = length(mps)
    T = _promoted_eltype(mps)

    left = Vector{Matrix{T}}(undef, N + 1)
    right = Vector{Matrix{T}}(undef, N + 1)
    left[1] = ones(T, 1, 1)
    right[N + 1] = ones(T, 1, 1)

    for site in 1:N
        left[site + 1] = _finite_norm_left_environment(left[site], mps[site])
    end
    for site in N:-1:1
        right[site] = _finite_norm_right_environment(mps[site], right[site + 1])
    end
    return left, right
end

function _check_finite_site(mps::AbstractVector, site::Integer)
    1 <= site <= length(mps) || throw(BoundsError(mps, site))
    return Int(site)
end

function _check_local_operator(
    operator::AbstractMatrix,
    physical_dimension::Integer,
    name::AbstractString,
)
    size(operator) == (physical_dimension, physical_dimension) ||
        throw(DimensionMismatch(
            "$name has size $(size(operator)); expected " *
            "($physical_dimension, $physical_dimension)",
        ))
    return nothing
end

function _observable_denominator(norm_value::Number, normalize::Bool)
    normalize || return one(real(norm_value))
    norm_squared = real(norm_value)
    (isfinite(norm_squared) && norm_squared > 0) || throw(ArgumentError(
        "cannot normalize an observable of a zero or non-finite MPS",
    ))
    return norm_squared
end

function _local_expectation_from_environments(
    A::AbstractArray{<:Number,3},
    operator::AbstractMatrix,
    left::AbstractMatrix,
    right::AbstractMatrix,
)
    @tensor value = left[l, lb] * A[l, r, sin] *
        operator[sout, sin] * conj(A[lb, rb, sout]) * right[r, rb]
    return value
end

# ---------------------------------------------------------------------------
# Local observables and correlations
# ---------------------------------------------------------------------------

"""
    local_expectation(mps, operator, site; normalize=true)

Return the one-site expectation value of `operator` at `site`.  The input may
be in an arbitrary MPS gauge.  With the default `normalize=true`, the result
is divided by `overlap(mps, mps)`; setting it to `false` returns the raw matrix
element.
"""
function local_expectation(
    mps::AbstractVector,
    operator::AbstractMatrix,
    site::Integer;
    normalize::Bool=true,
)
    _check_mps(mps)
    site = _check_finite_site(mps, site)
    _check_local_operator(operator, size(mps[site], 3), "operator")
    left, right = _finite_norm_environments(mps)
    denominator = _observable_denominator(left[end][1, 1], normalize)
    value = _local_expectation_from_environments(
        mps[site], operator, left[site], right[site + 1],
    )
    return value / denominator
end

"""
    local_expectations(mps, operator; normalize=true)

Measure the same one-site `operator` at every site, reusing a single set of
left and right norm environments.  Physical dimensions are allowed to vary,
provided `operator` has the correct dimension at every site.
"""
function local_expectations(
    mps::AbstractVector,
    operator::AbstractMatrix;
    normalize::Bool=true,
)
    _check_mps(mps)
    for site in eachindex(mps)
        _check_local_operator(operator, size(mps[site], 3), "operator")
    end

    left, right = _finite_norm_environments(mps)
    denominator = _observable_denominator(left[end][1, 1], normalize)
    return [
        _local_expectation_from_environments(
            mps[site], operator, left[site], right[site + 1],
        ) / denominator for site in eachindex(mps)
    ]
end

function _finite_identity_transfer(
    environment::AbstractMatrix,
    A::AbstractArray{<:Number,3},
)
    @tensor next_environment[r, rb] := environment[l, lb] *
        A[l, r, s] * conj(A[lb, rb, s])
    return next_environment
end

function _finite_operator_transfer(
    environment::AbstractMatrix,
    A::AbstractArray{<:Number,3},
    operator::AbstractMatrix,
)
    @tensor next_environment[r, rb] := environment[l, lb] *
        A[l, r, sin] * operator[sout, sin] * conj(A[lb, rb, sout])
    return next_environment
end

"""
    two_point_correlation(mps, A, i, B, j; string=nothing, normalize=true)

Return `⟨Aᵢ Bⱼ⟩`.  When `string` is a matrix, a copy of it is inserted at
every site strictly between `i` and `j`, which directly supports, for example,
Jordan–Wigner strings.  For `i == j`, the measured operator is the matrix
product `A * B`.  The contraction is gauge independent.
"""
function two_point_correlation(
    mps::AbstractVector,
    Aoperator::AbstractMatrix,
    i::Integer,
    Boperator::AbstractMatrix,
    j::Integer;
    string::Union{Nothing,AbstractMatrix}=nothing,
    normalize::Bool=true,
)
    _check_mps(mps)
    i = _check_finite_site(mps, i)
    j = _check_finite_site(mps, j)
    _check_local_operator(Aoperator, size(mps[i], 3), "Aoperator")
    _check_local_operator(Boperator, size(mps[j], 3), "Boperator")

    first_site, last_site = minmax(i, j)
    if string !== nothing
        for site in (first_site + 1):(last_site - 1)
            _check_local_operator(string, size(mps[site], 3), "string")
        end
    end

    T = promote_type(
        _promoted_eltype(mps),
        eltype(Aoperator),
        eltype(Boperator),
        string === nothing ? Bool : eltype(string),
    )
    environment = ones(T, 1, 1)

    for site in eachindex(mps)
        operator = if i == j && site == i
            Aoperator * Boperator
        elseif site == i
            Aoperator
        elseif site == j
            Boperator
        elseif string !== nothing && first_site < site < last_site
            string
        else
            nothing
        end

        environment = if operator === nothing
            _finite_identity_transfer(environment, mps[site])
        else
            _finite_operator_transfer(environment, mps[site], operator)
        end
    end

    size(environment) == (1, 1) || throw(DimensionMismatch(
        "correlation contraction did not close to a scalar",
    ))
    norm_value = overlap(mps, mps)
    denominator = _observable_denominator(norm_value, normalize)
    return environment[1, 1] / denominator
end

# ---------------------------------------------------------------------------
# Reduced states
# ---------------------------------------------------------------------------

"""
    reduced_density_matrix(mps, sites::AbstractUnitRange; normalize=true)

Return the reduced density matrix of a nonempty contiguous range of sites.
The first site in `sites` is the most significant tensor-product factor, as
in `mpo_matrix`.  The routine contracts both norm boundaries and is therefore
independent of the MPS gauge.  Its cost is necessarily exponential in the
number of retained sites.
"""
function reduced_density_matrix(
    mps::AbstractVector,
    sites::AbstractUnitRange{<:Integer};
    normalize::Bool=true,
)
    _check_mps(mps)
    isempty(sites) && throw(ArgumentError("sites must be a nonempty range"))
    first_site = Int(first(sites))
    last_site = Int(last(sites))
    1 <= first_site <= last_site <= length(mps) || throw(BoundsError(mps, sites))

    left, right = _finite_norm_environments(mps)
    theta = copy(mps[first_site])
    physical_dimension = size(theta, 3)

    for site in (first_site + 1):last_site
        A = mps[site]
        @tensor joined[l, r, p, q] := theta[l, middle, p] * A[middle, r, q]

        # Julia's first array index varies fastest.  Reversing the two
        # physical axes before reshaping makes the newly appended site the
        # least-significant tensor-product factor.
        joined = permutedims(joined, (1, 2, 4, 3))
        physical_dimension *= size(A, 3)
        theta = reshape(joined, size(joined, 1), size(joined, 2),
                        physical_dimension)
    end

    left_boundary = left[first_site]
    right_boundary = right[last_site + 1]
    # Absorb both boundaries while only one physical block index is open.
    @tensor rho[p, q] := (left_boundary[l, lb] * theta[l, r, p] *
        right_boundary[r, rb]) * conj(theta[lb, rb, q])

    denominator = _observable_denominator(left[end][1, 1], normalize)
    rho ./= denominator
    # Remove only the anti-Hermitian roundoff introduced by contraction
    # order.  This does not alter a mathematically valid reduced state.
    return Matrix((rho + adjoint(rho)) / 2)
end

function reduced_density_matrix(
    mps::AbstractVector,
    site::Integer;
    kwargs...,
)
    _check_mps(mps)
    site = _check_finite_site(mps, site)
    return reduced_density_matrix(mps, site:site; kwargs...)
end

# ---------------------------------------------------------------------------
# Schmidt data and entropy
# ---------------------------------------------------------------------------

function _finite_absorb_right(
    A::AbstractArray{<:Number,3},
    C::AbstractMatrix,
)
    size(A, 2) == size(C, 1) || throw(DimensionMismatch(
        "cannot absorb a $(size(C)) bond matrix into an MPS tensor $(size(A))",
    ))
    @tensor AC[l, rnew, s] := A[l, rold, s] * C[rold, rnew]
    return AC
end

"""
    schmidt_values(mps, bond; normalize=true)

Return the Schmidt singular values across the bond between sites `bond` and
`bond + 1`, in descending order.  A copy is brought to mixed-canonical form,
so the input is not mutated and need not already be canonical.  By default
the values have unit Euclidean norm.
"""
function schmidt_values(
    mps::AbstractVector,
    bond::Integer;
    normalize::Bool=true,
)
    _check_mps(mps)
    N = length(mps)
    1 <= bond < N || throw(ArgumentError(
        "bond must identify an interior cut in 1:$(max(N - 1, 0))",
    ))
    bond = Int(bond)
    work = [copy(A) for A in mps]
    T = _promoted_eltype(work)

    # Make the suffix right-isometric and leave its gauge on site `bond`.
    Cright = ones(T, 1, 1)
    for site in N:-1:(bond + 1)
        centre = _finite_absorb_right(work[site], Cright)
        Cright, AR = right_orthogonalize(centre)
        work[site] = AR
    end
    work[bond] = _finite_absorb_right(work[bond], Cright)

    # Make the prefix left-isometric.  The remaining matrix is the Schmidt
    # centre between two orthonormal block bases.
    Cleft = ones(T, 1, 1)
    for site in 1:bond
        centre = _absorb_left(Cleft, work[site])
        Cleft, AL = left_orthogonalize(centre)
        work[site] = AL
    end

    values = svdvals(Cleft)
    if normalize
        state_norm = norm(values)
        (isfinite(state_norm) && state_norm > 0) || throw(ArgumentError(
            "cannot normalize Schmidt values of a zero or non-finite MPS",
        ))
        values ./= state_norm
    end
    return values
end

"""
    entanglement_entropy(mps, bond; base=ℯ)

Return the von Neumann entropy across an interior MPS bond.  `base=ℯ` gives
nats and `base=2` gives bits.  Exact zero Schmidt probabilities are omitted
from the logarithm.
"""
function entanglement_entropy(
    mps::AbstractVector,
    bond::Integer;
    base::Real=ℯ,
)
    (isfinite(base) && base > 0 && base != 1) || throw(ArgumentError(
        "entropy base must be positive, finite, and different from one",
    ))
    probabilities = abs2.(schmidt_values(mps, bond; normalize=true))
    entropy = -sum(probability * log(probability) for probability in probabilities
                   if probability > 0)
    return entropy / log(base)
end
