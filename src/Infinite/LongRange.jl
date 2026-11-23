# VUMPS and uniform TDVP for exponential-sum long-range interactions.

"""
    ExponentialSumHamiltonian(onsite, left_ops, right_ops, coeffs, decays)

Represent the translation-invariant infinite-chain Hamiltonian

`sum_i onsite_i + sum_{i,a,k,r>=1} coeffs[a,k] * decays[a,k]^(r-1) *
 left_ops[a]_i * right_ops[a]_(i+r)`.

`onsite` and every endpoint operator are `d × d` matrices.  `coeffs` and
`decays` have shape `(number of operator pairs, number of exponentials)`.
The strict `abs(decay) < 1` check makes every geometric environment equation
well defined for a canonical transfer matrix.
"""
struct ExponentialSumHamiltonian{T<:Number}
    onsite::Matrix{T}
    left_ops::Vector{Matrix{T}}
    right_ops::Vector{Matrix{T}}
    coeffs::Matrix{T}
    decays::Matrix{T}
end

_longrange_isfinite(x::Number) = isfinite(real(x)) && isfinite(imag(x))

function ExponentialSumHamiltonian(
    onsite::AbstractMatrix{<:Number},
    left_ops::AbstractVector,
    right_ops::AbstractVector,
    coeffs::AbstractMatrix{<:Number},
    decays::AbstractMatrix{<:Number},
)
    size(onsite, 1) == size(onsite, 2) || throw(DimensionMismatch(
        "onsite must be square; got $(size(onsite))",
    ))
    d = size(onsite, 1)
    d > 0 || throw(ArgumentError("the physical dimension must be positive"))
    length(left_ops) == length(right_ops) || throw(DimensionMismatch(
        "left_ops and right_ops must have the same length",
    ))
    number_of_operators = length(left_ops)
    size(coeffs) == size(decays) || throw(DimensionMismatch(
        "coeffs and decays must have the same shape",
    ))
    size(coeffs, 1) == number_of_operators || throw(DimensionMismatch(
        "the first coeffs/decays dimension must equal the number of operator pairs",
    ))
    number_of_operators == 0 || size(coeffs, 2) > 0 || throw(ArgumentError(
        "at least one exponential is required when endpoint operators are present",
    ))

    for (name, operators) in (("left_ops", left_ops),
                              ("right_ops", right_ops))
        for (index, operator) in pairs(operators)
            operator isa AbstractMatrix || throw(ArgumentError(
                "$name[$index] must be a matrix",
            ))
            eltype(operator) <: Number || throw(ArgumentError(
                "$name[$index] must contain numbers",
            ))
            size(operator) == (d, d) || throw(DimensionMismatch(
                "$name[$index] must have size ($d, $d); got $(size(operator))",
            ))
        end
    end
    all(_longrange_isfinite, onsite) || throw(ArgumentError(
        "onsite contains a non-finite value",
    ))
    all(_longrange_isfinite, coeffs) || throw(ArgumentError(
        "coeffs contains a non-finite value",
    ))
    all(_longrange_isfinite, decays) || throw(ArgumentError(
        "decays contains a non-finite value",
    ))
    all(operator -> all(_longrange_isfinite, operator), left_ops) ||
        throw(ArgumentError("left_ops contains a non-finite value"))
    all(operator -> all(_longrange_isfinite, operator), right_ops) ||
        throw(ArgumentError("right_ops contains a non-finite value"))
    all(decay -> abs(decay) < 1, decays) || throw(ArgumentError(
        "every exponential decay must satisfy abs(decay) < 1",
    ))

    scalar_types = Type[eltype(onsite), eltype(coeffs), eltype(decays)]
    append!(scalar_types, (eltype(operator) for operator in left_ops))
    append!(scalar_types, (eltype(operator) for operator in right_ops))
    T = promote_type(scalar_types...)
    return ExponentialSumHamiltonian{T}(
        Matrix{T}(onsite),
        [Matrix{T}(operator) for operator in left_ops],
        [Matrix{T}(operator) for operator in right_ops],
        Matrix{T}(coeffs),
        Matrix{T}(decays),
    )
end

Base.eltype(::Type{ExponentialSumHamiltonian{T}}) where {T} = T
Base.eltype(::ExponentialSumHamiltonian{T}) where {T} = T

"""
    triangular_mpo(h::ExponentialSumHamiltonian)

Convert an exponential-sum Hamiltonian into the equivalent lower-triangular
uniform MPO.  Each nonzero `(operator, exponential)` pair receives one
propagating auxiliary channel.  This supplies an independent implementation
of the same model and lets long-range calculations reuse the generic MPO bond
expansion.
"""
function triangular_mpo(h::ExponentialSumHamiltonian)
    terms = Tuple{Int,Int}[]
    for operator_index in axes(h.coeffs, 1),
        exponential_index in axes(h.coeffs, 2)
        iszero(h.coeffs[operator_index, exponential_index]) ||
            push!(terms, (operator_index, exponential_index))
    end

    d = size(h.onsite, 1)
    auxiliary_dimension = length(terms) + 2
    W = zeros(eltype(h), d, d, auxiliary_dimension, auxiliary_dimension)
    identity_physical = Matrix{eltype(h)}(I, d, d)
    W[:, :, 1, 1] .= identity_physical
    W[:, :, auxiliary_dimension, auxiliary_dimension] .= identity_physical
    W[:, :, auxiliary_dimension, 1] .= h.onsite
    for (term_index, (operator_index, exponential_index)) in pairs(terms)
        channel = term_index + 1
        coefficient = h.coeffs[operator_index, exponential_index]
        decay = h.decays[operator_index, exponential_index]
        W[:, :, channel, 1] .= h.right_ops[operator_index]
        W[:, :, channel, channel] .= decay .* identity_physical
        W[:, :, auxiliary_dimension, channel] .=
            coefficient .* h.left_ops[operator_index]
    end
    return TriangularMPO(W)
end

"""
    finite_mpo(h::ExponentialSumHamiltonian, N)

Construct the open-boundary finite MPO for the same exponential-sum model.
At separation `r >= 1`, each endpoint pair has coefficient
`coeffs[a,k] * decays[a,k]^(r-1)`, matching the infinite representation.
"""
finite_mpo(h::ExponentialSumHamiltonian, N::Integer) =
    finite_mpo(triangular_mpo(h), N)

function bond_expansion_coefficients(
    AL,
    AR,
    C,
    h::ExponentialSumHamiltonian;
    kwargs...,
)
    return bond_expansion_coefficients(
        AL, AR, C, triangular_mpo(h); kwargs...,
    )
end

function expand_bond_dimension(
    AL,
    AR,
    C,
    h::ExponentialSumHamiltonian;
    kwargs...,
)
    return expand_bond_dimension(AL, AR, C, triangular_mpo(h); kwargs...)
end

function _check_longrange_state(AL, AR, C, h::ExponentialSumHamiltonian)
    chi, d = _check_mixed_canonical(AL, AR, C)
    size(h.onsite) == (d, d) || throw(DimensionMismatch(
        "Hamiltonian physical dimension $(size(h.onsite, 1)) does not match MPS physical dimension $d",
    ))
    return chi, d
end

# The nearest-neighbour backend uses the transpose of the conventional
# left density matrix.  These two maps deliberately follow that storage
# convention while keeping physical operators in Julia's (output,input)
# order.
function _longrange_transfer_left(A, X, operator)
    @tensor Y[r, rb] := X[l, lb] * A[l, r, si] *
                        operator[so, si] * conj(A[lb, rb, so])
    return Y
end

function _longrange_transfer_right(A, X, operator)
    @tensor Y[l, lb] := A[l, r, si] * X[r, rb] *
                        operator[so, si] * conj(A[lb, rb, so])
    return Y
end

function _check_longrange_tail_seed(seed, name, dimensions, chi)
    seed === nothing && return nothing
    size(seed) == dimensions || throw(DimensionMismatch(
        "$name must have size $dimensions; got $(size(seed))",
    ))
    for index in CartesianIndices(seed)
        value = seed[index]
        value isa AbstractMatrix || throw(ArgumentError(
            "$name[$index] must be a matrix",
        ))
        size(value) == (chi, chi) || throw(DimensionMismatch(
            "$name[$index] must have size ($chi, $chi); got $(size(value))",
        ))
    end
    return nothing
end

function _longrange_matrix_seed(seed, name, chi, T)
    seed === nothing && return zeros(T, chi, chi)
    seed isa AbstractMatrix || throw(ArgumentError("$name must be a matrix"))
    size(seed) == (chi, chi) || throw(DimensionMismatch(
        "$name must have size ($chi, $chi); got $(size(seed))",
    ))
    return Matrix{T}(seed)
end

function _longrange_tail_context(
    AL,
    AR,
    C,
    h::ExponentialSumHamiltonian;
    L0=nothing,
    R0=nothing,
    OL0=nothing,
    OR0=nothing,
    transfer_tol::Real=1e-12,
    transfer_maxiter::Integer=500,
    transfer_krylovdim::Integer=30,
    env_tol::Real=1e-12,
    env_maxiter::Integer=200,
    env_krylovdim::Integer=30,
)
    chi, d = _check_longrange_state(AL, AR, C, h)
    number_of_operators, number_of_exponentials = size(h.coeffs)
    tail_dimensions = (number_of_operators, number_of_exponentials)
    _check_longrange_tail_seed(OL0, "OL0", tail_dimensions, chi)
    _check_longrange_tail_seed(OR0, "OR0", tail_dimensions, chi)

    fixed = transfer_fixed_points(
        AL,
        AR;
        L0=L0,
        R0=R0,
        tol=transfer_tol,
        maxiter=transfer_maxiter,
        krylovdim=transfer_krylovdim,
    )
    T = promote_type(eltype(AL), eltype(AR), eltype(C), eltype(h))
    identity_matrix = Matrix{T}(I, chi, chi)
    left_density_tensor = transpose(fixed.L)
    OL = Matrix{Matrix{T}}(undef, tail_dimensions...)
    OR = Matrix{Matrix{T}}(undef, tail_dimensions...)
    OL_sum = [zeros(T, chi, chi) for _ in 1:number_of_operators]
    OR_sum = [zeros(T, chi, chi) for _ in 1:number_of_operators]

    tail_converged = true
    tail_residual = 0.0
    tail_numops = 0
    tail_numiter = 0
    for operator_index in 1:number_of_operators
        source_left = _longrange_transfer_left(
            AL, identity_matrix, h.left_ops[operator_index],
        )
        source_right = _longrange_transfer_right(
            AR, identity_matrix, h.right_ops[operator_index],
        )
        for exponential_index in 1:number_of_exponentials
            decay = h.decays[operator_index, exponential_index]
            left_operator = X -> X - decay .* _environment_transfer_left(AL, X)
            right_operator = X -> X - decay .* transfer_right(AR, X)
            left_seed = OL0 === nothing ? zeros(T, chi, chi) :
                        Matrix{T}(OL0[operator_index, exponential_index])
            right_seed = OR0 === nothing ? zeros(T, chi, chi) :
                         Matrix{T}(OR0[operator_index, exponential_index])
            left_tail, left_info = _solve_uniform_environment(
                left_operator,
                source_left,
                left_seed;
                tol=env_tol,
                maxiter=env_maxiter,
                krylovdim=env_krylovdim,
            )
            right_tail, right_info = _solve_uniform_environment(
                right_operator,
                source_right,
                right_seed;
                tol=env_tol,
                maxiter=env_maxiter,
                krylovdim=env_krylovdim,
            )
            OL[operator_index, exponential_index] = Matrix{T}(left_tail)
            OR[operator_index, exponential_index] = Matrix{T}(right_tail)
            coefficient = h.coeffs[operator_index, exponential_index]
            OL_sum[operator_index] .+= coefficient .* left_tail
            OR_sum[operator_index] .+= coefficient .* right_tail
            tail_converged &= left_info.converged && right_info.converged
            tail_residual = max(
                tail_residual, left_info.residual, right_info.residual,
            )
            tail_numops += left_info.numops + right_info.numops
            tail_numiter += left_info.numiter + right_info.numiter
        end
    end

    # Complete a left endpoint with O_R, and a right endpoint with O_L.
    # This is the asymmetric-channel closure that is accidentally reversed in
    # the MATLAB vtdvp_lr.m half-chain source.
    hamiltonian_source_left = _longrange_transfer_left(
        AL, identity_matrix, h.onsite,
    )
    hamiltonian_source_right = _longrange_transfer_right(
        AR, identity_matrix, h.onsite,
    )
    for operator_index in 1:number_of_operators
        hamiltonian_source_left .+= _longrange_transfer_left(
            AL, OL_sum[operator_index], h.right_ops[operator_index],
        )
        hamiltonian_source_right .+= _longrange_transfer_right(
            AR, OR_sum[operator_index], h.left_ops[operator_index],
        )
    end

    energy_left = _tensor_trace_pair(hamiltonian_source_left, fixed.R)
    energy_right = _tensor_trace_pair(
        left_density_tensor, hamiltonian_source_right,
    )
    energy = (energy_left + energy_right) / 2
    return (
        AC=center_tensor(AL, C),
        L=fixed.L,
        R=fixed.R,
        left_density_tensor=left_density_tensor,
        OL=OL,
        OR=OR,
        OL_sum=OL_sum,
        OR_sum=OR_sum,
        source_left=hamiltonian_source_left,
        source_right=hamiltonian_source_right,
        energy=energy,
        energy_left=energy_left,
        energy_right=energy_right,
        energy_mismatch=Float64(abs(energy_left - energy_right)),
        transfer_converged=fixed.converged,
        transfer_residual=max(fixed.residual_L, fixed.residual_R),
        tails_converged=tail_converged,
        tail_residual=tail_residual,
        tail_numops=tail_numops,
        tail_numiter=tail_numiter,
        physical_dimension=d,
    )
end

"""
    longrange_context(AL, AR, C, h; kwargs...)

Build fixed points, geometric endpoint tails, and the two projected
half-chain Hamiltonian environments for an [`ExponentialSumHamiltonian`](@ref).
The returned diagnostics include independent left/right energy estimates and
linear-solve residuals.
"""
function longrange_context(
    AL,
    AR,
    C,
    h::ExponentialSumHamiltonian;
    HL0=nothing,
    HR0=nothing,
    L0=nothing,
    R0=nothing,
    OL0=nothing,
    OR0=nothing,
    transfer_tol::Real=1e-12,
    transfer_maxiter::Integer=500,
    transfer_krylovdim::Integer=30,
    env_tol::Real=1e-12,
    env_maxiter::Integer=200,
    env_krylovdim::Integer=30,
)
    tail = _longrange_tail_context(
        AL,
        AR,
        C,
        h;
        L0=L0,
        R0=R0,
        OL0=OL0,
        OR0=OR0,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
    )
    chi = size(C, 1)
    T = promote_type(eltype(AL), eltype(AR), eltype(C), eltype(h))
    identity_matrix = Matrix{T}(I, chi, chi)
    left_seed = _longrange_matrix_seed(HL0, "HL0", chi, T)
    right_seed = _longrange_matrix_seed(HR0, "HR0", chi, T)
    left_operator = X -> X - _environment_transfer_left(AL, X) +
                         identity_matrix .* _tensor_trace_pair(X, tail.R)
    right_operator = X -> X - transfer_right(AR, X) +
                          identity_matrix .* _tensor_trace_pair(
                              tail.left_density_tensor, X,
                          )
    HL, left_info = _solve_uniform_environment(
        left_operator,
        tail.source_left,
        left_seed;
        tol=env_tol,
        maxiter=env_maxiter,
        krylovdim=env_krylovdim,
    )
    HR, right_info = _solve_uniform_environment(
        right_operator,
        tail.source_right,
        right_seed;
        tol=env_tol,
        maxiter=env_maxiter,
        krylovdim=env_krylovdim,
    )
    HL .-= identity_matrix .* _tensor_trace_pair(HL, tail.R)
    HR .-= identity_matrix .* _tensor_trace_pair(
        tail.left_density_tensor, HR,
    )

    half_residual = max(left_info.residual, right_info.residual)
    return merge(tail, (
        HL=HL,
        HR=HR,
        environments_converged=tail.tails_converged &&
                               left_info.converged && right_info.converged,
        environment_residual=max(tail.tail_residual, half_residual),
        environment_numops=tail.tail_numops + left_info.numops +
                           right_info.numops,
        environment_numiter=tail.tail_numiter + left_info.numiter +
                            right_info.numiter,
    ))
end

"""Evaluate the energy density of an exponential-sum Hamiltonian on a uMPS."""
function longrange_energy(AL, AR, C, h::ExponentialSumHamiltonian; kwargs...)
    return _longrange_tail_context(AL, AR, C, h; kwargs...).energy
end

"""Apply the long-range one-site-centre effective Hamiltonian."""
function apply_hac_longrange(AC, h::ExponentialSumHamiltonian, context)
    chi, _, d = size(AC)
    size(AC) == (chi, chi, d) || throw(DimensionMismatch(
        "AC must have equal left and right bond dimensions",
    ))
    size(h.onsite) == (d, d) || throw(DimensionMismatch(
        "Hamiltonian and AC physical dimensions differ",
    ))
    size(context.HL) == (chi, chi) || throw(DimensionMismatch(
        "HL has the wrong size",
    ))
    size(context.HR) == (chi, chi) || throw(DimensionMismatch(
        "HR has the wrong size",
    ))

    @tensor onsite_term[l, r, so] := h.onsite[so, si] * AC[l, r, si]
    @tensor left_environment[lb, r, s] := context.HL[l, lb] * AC[l, r, s]
    @tensor right_environment[l, rb, s] := AC[l, r, s] * context.HR[r, rb]
    result = onsite_term + left_environment + right_environment
    number_of_operators, number_of_exponentials = size(h.coeffs)
    for operator_index in 1:number_of_operators
        @tensor closes_left[lb, r, so] :=
            context.OL_sum[operator_index][l, lb] * AC[l, r, si] *
            h.right_ops[operator_index][so, si]
        @tensor closes_right[l, rb, so] :=
            h.left_ops[operator_index][so, si] * AC[l, r, si] *
            context.OR_sum[operator_index][r, rb]
        result .+= closes_left + closes_right
        for exponential_index in 1:number_of_exponentials
            @tensor crosses_centre[lb, rb, s] :=
                context.OL[operator_index, exponential_index][l, lb] *
                AC[l, r, s] *
                context.OR[operator_index, exponential_index][r, rb]
            result .+= h.coeffs[operator_index, exponential_index] *
                       h.decays[operator_index, exponential_index] .*
                       crosses_centre
        end
    end
    return result
end

"""Apply the long-range centre-matrix effective Hamiltonian."""
function apply_hc_longrange(C, h::ExponentialSumHamiltonian, context)
    chi = size(C, 1)
    size(C) == (chi, chi) || throw(DimensionMismatch("C must be square"))
    size(context.HL) == (chi, chi) || throw(DimensionMismatch(
        "HL has the wrong size",
    ))
    size(context.HR) == (chi, chi) || throw(DimensionMismatch(
        "HR has the wrong size",
    ))
    @tensor left_environment[lb, r] := context.HL[l, lb] * C[l, r]
    @tensor right_environment[l, rb] := C[l, r] * context.HR[r, rb]
    result = left_environment + right_environment
    number_of_operators, number_of_exponentials = size(h.coeffs)
    for operator_index in 1:number_of_operators
        for exponential_index in 1:number_of_exponentials
            @tensor crosses_bond[lb, rb] :=
                context.OL[operator_index, exponential_index][l, lb] *
                C[l, r] *
                context.OR[operator_index, exponential_index][r, rb]
            result .+= h.coeffs[operator_index, exponential_index] .*
                       crosses_bond
        end
    end
    return result
end

"""Perform one fixed-bond-dimension long-range VUMPS iteration."""
function longrange_vumps_step(
    AL,
    AR,
    C,
    h::ExponentialSumHamiltonian;
    HL0=nothing,
    HR0=nothing,
    L0=nothing,
    R0=nothing,
    OL0=nothing,
    OR0=nothing,
    eig_tol::Real=1e-12,
    eig_maxiter::Integer=500,
    eig_krylovdim::Integer=30,
    transfer_tol::Real=1e-12,
    transfer_maxiter::Integer=500,
    transfer_krylovdim::Integer=30,
    env_tol::Real=1e-12,
    env_maxiter::Integer=200,
    env_krylovdim::Integer=30,
)
    _check_longrange_state(AL, AR, C, h)
    context = longrange_context(
        AL,
        AR,
        C,
        h;
        HL0=HL0,
        HR0=HR0,
        L0=L0,
        R0=R0,
        OL0=OL0,
        OR0=OR0,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
    )
    energy_AC, ACnew, residual_AC, converged_AC = _lowest_local_eigenpair(
        X -> apply_hac_longrange(X, h, context),
        context.AC;
        eig_tol=eig_tol,
        maxiter=eig_maxiter,
        krylovdim=eig_krylovdim,
    )
    energy_C, Cnew, residual_C, converged_C = _lowest_local_eigenpair(
        X -> apply_hc_longrange(X, h, context),
        C;
        eig_tol=eig_tol,
        maxiter=eig_maxiter,
        krylovdim=eig_krylovdim,
    )
    update = mixed_canonical_update(ACnew, Cnew)
    energy_after = _longrange_tail_context(
        update.AL,
        update.AR,
        update.C,
        h;
        L0=context.L,
        R0=context.R,
        OL0=context.OL,
        OR0=context.OR,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
    )
    return (
        AL=update.AL,
        AR=update.AR,
        C=update.C,
        AC=ACnew,
        HL=context.HL,
        HR=context.HR,
        L=context.L,
        R=context.R,
        OL=context.OL,
        OR=context.OR,
        energy=Float64(real(energy_after.energy)),
        previous_energy=Float64(real(context.energy)),
        energy_mismatch=energy_after.energy_mismatch,
        effective_energy_AC=energy_AC,
        effective_energy_C=energy_C,
        residual=update.residual,
        residuals=update.residuals,
        eig_residual=max(residual_AC, residual_C),
        eig_converged=converged_AC && converged_C,
        transfer_converged=context.transfer_converged,
        transfer_residual=context.transfer_residual,
        environments_converged=context.environments_converged,
        environment_residual=context.environment_residual,
        converged_inner=converged_AC && converged_C &&
                        context.transfer_converged &&
                        context.environments_converged,
    )
end

"""
Iterate VUMPS for an exponential-sum Hamiltonian, optionally growing the bond
dimension from the equivalent triangular-MPO two-site residual.
"""
function longrange_vumps(
    AL,
    AR,
    C,
    h::ExponentialSumHamiltonian;
    maxiter::Integer=100,
    tol::Real=1e-10,
    max_bond::Integer=size(AL, 1),
    expand_tol::Real=1e-1,
    expansion_atol::Real=1e-14,
    expansion_rtol::Real=1e-12,
    expansion_max_add::Integer=typemax(Int),
    kwargs...,
)
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    expand_tol >= 0 || throw(ArgumentError("expand_tol must be nonnegative"))
    initial_bond, _ = _check_longrange_state(AL, AR, C, h)
    max_bond >= initial_bond || throw(ArgumentError(
        "max_bond ($max_bond) cannot be smaller than the input bond dimension ($initial_bond)",
    ))

    current_AL = copy(AL)
    current_AR = copy(AR)
    current_C = copy(C)
    HL = HR = L = R = OL = OR = nothing
    history = NamedTuple[]
    growth_history = NamedTuple[]
    final_step = nothing
    converged = false
    initial_residuals = infinite_residuals(current_AL, current_AR, current_C)
    previous_residual = max(initial_residuals.center_left,
                            initial_residuals.center_right)
    for iteration in 1:maxiter
        growth_attempted = false
        growth_happened = false
        if size(current_C, 1) < max_bond && previous_residual <= expand_tol
            growth_attempted = true
            expansion = expand_bond_dimension(
                current_AL, current_AR, current_C, h;
                max_bond=max_bond,
                max_add=expansion_max_add,
                atol=expansion_atol,
                rtol=expansion_rtol,
                transfer_tol=get(kwargs, :transfer_tol, 1e-12),
                transfer_maxiter=get(kwargs, :transfer_maxiter, 500),
                transfer_krylovdim=get(kwargs, :transfer_krylovdim, 30),
                env_tol=get(kwargs, :env_tol, 1e-12),
                env_maxiter=get(kwargs, :env_maxiter, 200),
                env_krylovdim=get(kwargs, :env_krylovdim, 30),
            )
            push!(growth_history, (
                iteration=iteration,
                old_bond=expansion.old_bond,
                new_bond=expansion.new_bond,
                added_bond=expansion.added_bond,
                expanded=expansion.expanded,
                projected_residual=expansion.residual_norm,
                singular_values=expansion.singular_values,
                threshold=expansion.threshold,
            ))
            if expansion.expanded
                growth_happened = true
                current_AL, current_AR, current_C =
                    expansion.AL, expansion.AR, expansion.C
                HL = HR = L = R = OL = OR = nothing
            end
        end
        step = longrange_vumps_step(
            current_AL,
            current_AR,
            current_C,
            h;
            HL0=HL,
            HR0=HR,
            L0=L,
            R0=R,
            OL0=OL,
            OR0=OR,
            kwargs...,
        )
        current_AL, current_AR, current_C = step.AL, step.AR, step.C
        HL, HR, L, R, OL, OR =
            step.HL, step.HR, step.L, step.R, step.OL, step.OR
        push!(history, (
            iteration=iteration,
            bond_dimension=size(current_C, 1),
            energy=step.energy,
            residual=step.residual,
            eig_residual=step.eig_residual,
            transfer_residual=step.transfer_residual,
            environment_residual=step.environment_residual,
            energy_mismatch=step.energy_mismatch,
        ))
        final_step = step
        previous_residual = step.residual
        fixed_point_converged = step.residual <= tol && step.converged_inner
        adaptive_complete = size(current_C, 1) >= max_bond ||
                            previous_residual > expand_tol ||
                            (growth_attempted && !growth_happened)
        converged = fixed_point_converged && adaptive_complete
        converged && break
    end

    final_context = longrange_context(
        current_AL,
        current_AR,
        current_C,
        h;
        HL0=HL,
        HR0=HR,
        L0=L,
        R0=R,
        OL0=OL,
        OR0=OR,
        transfer_tol=get(kwargs, :transfer_tol, 1e-12),
        transfer_maxiter=get(kwargs, :transfer_maxiter, 500),
        transfer_krylovdim=get(kwargs, :transfer_krylovdim, 30),
        env_tol=get(kwargs, :env_tol, 1e-12),
        env_maxiter=get(kwargs, :env_maxiter, 200),
        env_krylovdim=get(kwargs, :env_krylovdim, 30),
    )
    return (
        AL=current_AL,
        AR=current_AR,
        C=current_C,
        AC=center_tensor(current_AL, current_C),
        HL=final_context.HL,
        HR=final_context.HR,
        L=final_context.L,
        R=final_context.R,
        OL=final_context.OL,
        OR=final_context.OR,
        energy=Float64(real(final_context.energy)),
        energy_mismatch=final_context.energy_mismatch,
        residual=final_step.residual,
        residuals=final_step.residuals,
        converged=converged,
        bond_dimension=size(current_C, 1),
        iterations=length(history),
        history=history,
        growth_history=growth_history,
        transfer_converged=final_context.transfer_converged,
        environments_converged=final_context.environments_converged,
    )
end

function longrange_vumps(
    h::ExponentialSumHamiltonian,
    chi::Integer;
    rng::AbstractRNG=Random.default_rng(),
    T::Type{<:Number}=ComplexF64,
    kwargs...,
)
    initial = random_infinite_mps(
        chi, size(h.onsite, 1); rng=rng, T=T,
    )
    return longrange_vumps(initial.AL, initial.AR, initial.C, h; kwargs...)
end

"""Take one long-range uniform-TDVP step, with optional bond growth."""
function longrange_tdvp_step(
    AL,
    AR,
    C,
    h::ExponentialSumHamiltonian,
    dt::Number;
    time::Symbol=:real,
    tol::Real=1e-12,
    maxiter::Integer=100,
    krylovdim::Integer=30,
    HL0=nothing,
    HR0=nothing,
    L0=nothing,
    R0=nothing,
    OL0=nothing,
    OR0=nothing,
    transfer_tol::Real=1e-12,
    transfer_maxiter::Integer=500,
    transfer_krylovdim::Integer=30,
    env_tol::Real=1e-12,
    env_maxiter::Integer=200,
    env_krylovdim::Integer=30,
    max_bond::Integer=size(AL, 1),
    expansion_atol::Real=1e-14,
    expansion_rtol::Real=1e-12,
    expansion_max_add::Integer=typemax(Int),
)
    chi, _ = _check_longrange_state(AL, AR, C, h)
    max_bond >= chi || throw(ArgumentError(
        "max_bond ($max_bond) cannot be smaller than the input bond dimension ($chi)",
    ))
    expansion = if max_bond == chi
        # Avoid constructing a second, equivalent MPO context on the default
        # fixed-dimension path.
        expand_bond_dimension(
            AL, AR, C, identity;
            max_bond=max_bond,
            max_add=expansion_max_add,
            atol=expansion_atol,
            rtol=expansion_rtol,
        )
    else
        expand_bond_dimension(
            AL, AR, C, h;
            max_bond=max_bond,
            max_add=expansion_max_add,
            atol=expansion_atol,
            rtol=expansion_rtol,
            transfer_tol=transfer_tol,
            transfer_maxiter=transfer_maxiter,
            transfer_krylovdim=transfer_krylovdim,
            env_tol=env_tol,
            env_maxiter=env_maxiter,
            env_krylovdim=env_krylovdim,
        )
    end
    if expansion.expanded
        AL, AR, C = expansion.AL, expansion.AR, expansion.C
        HL0 = HR0 = L0 = R0 = OL0 = OR0 = nothing
    end
    parameter = _tdvp_parameter(dt, time)
    context = longrange_context(
        AL,
        AR,
        C,
        h;
        HL0=HL0,
        HR0=HR0,
        L0=L0,
        R0=R0,
        OL0=OL0,
        OR0=OR0,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
    )
    ACnew, info_AC = _exponentiate_tensor(
        X -> apply_hac_longrange(X, h, context),
        context.AC,
        parameter;
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
    Cnew, info_C = _exponentiate_tensor(
        X -> apply_hc_longrange(X, h, context),
        C,
        parameter;
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
    update = mixed_canonical_update(ACnew, Cnew)
    energy_after = _longrange_tail_context(
        update.AL,
        update.AR,
        update.C,
        h;
        L0=context.L,
        R0=context.R,
        OL0=context.OL,
        OR0=context.OR,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
    )
    return (
        AL=update.AL,
        AR=update.AR,
        C=update.C,
        AC=ACnew,
        HL=context.HL,
        HR=context.HR,
        L=context.L,
        R=context.R,
        OL=context.OL,
        OR=context.OR,
        energy_before=Float64(real(context.energy)),
        energy_after=Float64(real(energy_after.energy)),
        energy_mismatch=energy_after.energy_mismatch,
        residual=update.residual,
        residuals=update.residuals,
        bond_dimension=size(update.C, 1),
        expansion=expansion,
        krylov_converged=info_AC.converged && info_C.converged,
        krylov_error=max(info_AC.error, info_C.error),
        transfer_converged=context.transfer_converged,
        environments_converged=context.environments_converged,
        environment_residual=context.environment_residual,
    )
end

# Typed overloads give all three infinite Hamiltonian representations the same
# high-level entry points; the explicit long-range names remain available for
# MATLAB source mapping and discoverability.
vumps_step(AL, AR, C, h::ExponentialSumHamiltonian; kwargs...) =
    longrange_vumps_step(AL, AR, C, h; kwargs...)
vumps(AL, AR, C, h::ExponentialSumHamiltonian; kwargs...) =
    longrange_vumps(AL, AR, C, h; kwargs...)
vumps(h::ExponentialSumHamiltonian, chi::Integer; kwargs...) =
    longrange_vumps(h, chi; kwargs...)
infinite_tdvp_step(
    AL, AR, C, h::ExponentialSumHamiltonian, dt::Number; kwargs...,
) = longrange_tdvp_step(AL, AR, C, h, dt; kwargs...)
