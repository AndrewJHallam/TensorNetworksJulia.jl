# Shared scalar, array, and truncation utilities.

function _check_rank(A::AbstractArray, rank::Integer, name::AbstractString)
    ndims(A) == rank || throw(DimensionMismatch(
        "$name must be a rank-$rank array; got size $(size(A))"))
    return nothing
end

function _promoted_eltype(arrays)
    iterator = iterate(arrays)
    iterator === nothing && return Float64
    first_array, state = iterator
    T = eltype(first_array)
    while true
        iterator = iterate(arrays, state)
        iterator === nothing && return T
        array, state = iterator
        T = promote_type(T, eltype(array))
    end
end

function _promoted_eltype(left, right)
    return promote_type(_promoted_eltype(left), _promoted_eltype(right))
end

_real_type(::Type{Complex{T}}) where {T<:Real} = T
_real_type(::Type{T}) where {T<:Real} = T

_is_floating_scalar_type(::Type{T}) where {T} =
    T <: AbstractFloat || (T <: Complex && _real_type(T) <: AbstractFloat)

function _random_array(rng::AbstractRNG, ::Type{T}, dims::Vararg{Int}) where {T<:Real}
    return randn(rng, T, dims...)
end

function _random_array(rng::AbstractRNG, ::Type{Complex{T}}, dims::Vararg{Int}) where {T<:Real}
    return complex.(randn(rng, T, dims...), randn(rng, T, dims...))
end

"""Return `min(base^power, cap)` without overflowing an integer."""
function _bounded_power(base::Integer, power::Integer, cap::Integer)
    power >= 0 || throw(ArgumentError("power must be non-negative"))
    cap >= 1 || throw(ArgumentError("cap must be positive"))
    value = 1
    for _ in 1:power
        value > cap ÷ base && return cap
        value *= base
    end
    return min(value, cap)
end

function _svd_rank(singular_values::AbstractVector, maxdim::Integer, cutoff::Real)
    maxdim >= 1 || throw(ArgumentError("maxdim must be at least 1"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative"))
    isempty(singular_values) && throw(ArgumentError("cannot truncate an empty SVD"))

    rank = min(maxdim, length(singular_values))
    total_weight = sum(abs2, singular_values)
    iszero(total_weight) && return (1, zero(real(total_weight)))

    discarded = sum(abs2, @view(singular_values[(rank + 1):end]))
    if cutoff > 0
        while rank > 1
            candidate = discarded + abs2(singular_values[rank])
            candidate / total_weight <= cutoff || break
            discarded = candidate
            rank -= 1
        end
    end
    return rank, real(discarded / total_weight)
end
