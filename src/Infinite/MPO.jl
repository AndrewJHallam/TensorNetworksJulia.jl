# VUMPS and uniform TDVP for lower-triangular infinite Hamiltonian MPOs.
#
# This file is intentionally self-contained apart from the finite-MPO actions
# and uniform-MPS/Krylov helpers defined by the rest of TensorNetworksJulia.

"""
    TriangularMPO(W; lambda=nothing, atol=1e-12, rtol=1e-10)
    TriangularMPO(W, lambda; kwargs...)

A translation-invariant lower-triangular Hamiltonian MPO.  `W` uses the
package order `(physical output, physical input, MPO left, MPO right)`.  Each
diagonal auxiliary block must be a scalar multiple `lambda[q] * I`; the first
and last blocks must be the identity.  If `lambda` is omitted it is inferred
from the diagonal blocks.

The triangular restriction is deliberate: it is what makes the extensive
Hamiltonian environments well defined channel by channel in the thermodynamic
limit.  A general uniform MPO is not accepted by this type.
"""
struct TriangularMPO{A,V}
    W::A
    lambda::V
end

function TriangularMPO(
    W::AbstractArray{<:Number,4};
    lambda=nothing,
    atol::Real=1e-12,
    rtol::Real=1e-10,
)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    d_out, d_in, w_left, w_right = size(W)
    d_out == d_in || throw(DimensionMismatch(
        "MPO physical dimensions must agree; got ($d_out, $d_in)",
    ))
    w_left == w_right || throw(DimensionMismatch(
        "a uniform triangular MPO needs equal auxiliary dimensions; got " *
        "($w_left, $w_right)",
    ))
    d_out > 0 || throw(ArgumentError("the physical dimension must be positive"))
    w_left >= 2 || throw(ArgumentError(
        "a Hamiltonian MPO needs distinct initial and terminal channels",
    ))
    all(isfinite, W) || throw(ArgumentError("W contains non-finite entries"))

    total_scale = max(1.0, Float64(norm(W)))
    triangular_tolerance = Float64(atol + rtol * total_scale)
    for qleft in 1:w_left, qright in (qleft + 1):w_right
        norm(view(W, :, :, qleft, qright)) <= triangular_tolerance ||
            throw(ArgumentError(
                "W is not lower triangular: block ($qleft, $qright) is nonzero",
            ))
    end

    identity_physical = Matrix{eltype(W)}(I, d_out, d_out)
    inferred = [tr(Matrix(view(W, :, :, q, q))) / d_out for q in 1:w_left]
    for q in 1:w_left
        diagonal = Matrix(view(W, :, :, q, q))
        scale = max(1.0, Float64(norm(diagonal)))
        norm(diagonal - inferred[q] .* identity_physical) <=
            Float64(atol + rtol * scale) || throw(ArgumentError(
                "diagonal MPO block $q is not a scalar multiple of the identity",
            ))
    end

    weights = if lambda === nothing
        inferred
    else
        lambda isa AbstractVector || throw(ArgumentError(
            "lambda must be an AbstractVector",
        ))
        eltype(lambda) <: Number || throw(ArgumentError(
            "lambda must contain numbers",
        ))
        length(lambda) == w_left || throw(DimensionMismatch(
            "lambda has length $(length(lambda)); expected $w_left",
        ))
        Tlambda = promote_type(eltype(inferred), eltype(lambda))
        converted = Vector{Tlambda}(lambda)
        for q in 1:w_left
            isapprox(converted[q], inferred[q]; atol=atol, rtol=rtol) ||
                throw(ArgumentError(
                    "lambda[$q] does not match diagonal MPO block $q",
                ))
        end
        converted
    end
    all(isfinite, weights) || throw(ArgumentError(
        "lambda contains non-finite entries",
    ))
    isapprox(first(weights), one(first(weights)); atol=atol, rtol=rtol) ||
        throw(ArgumentError("the first MPO channel must propagate the identity"))
    isapprox(last(weights), one(last(weights)); atol=atol, rtol=rtol) ||
        throw(ArgumentError("the last MPO channel must propagate the identity"))

    stored_W = Array(W)
    return TriangularMPO{typeof(stored_W),typeof(weights)}(stored_W, weights)
end

TriangularMPO(W::AbstractArray{<:Number,4}, lambda::AbstractVector; kwargs...) =
    TriangularMPO(W; lambda=lambda, kwargs...)

Base.size(H::TriangularMPO) = size(H.W)
Base.eltype(H::TriangularMPO) = promote_type(eltype(H.W), eltype(H.lambda))

"""
    finite_mpo(H::TriangularMPO, N)

Terminate a lower-triangular uniform Hamiltonian MPO on an open chain of
length `N`.  The initial auxiliary state is the final (bottom) channel and
the terminal state is the first channel.  This is also the boundary
convention used by the finite MPO constructors in this package.
"""
function finite_mpo(H::TriangularMPO, N::Integer)
    N >= 1 || throw(ArgumentError("N must be positive"))
    W = H.W
    d = size(W, 1)
    auxiliary_dimension = size(W, 3)
    T = eltype(W)

    if N == 1
        onsite = Array(view(W, :, :, auxiliary_dimension, 1))
        return [reshape(onsite, d, d, 1, 1)]
    end

    mpo = Vector{Array{T,4}}(undef, N)
    left = Array(view(W, :, :, auxiliary_dimension, :))
    mpo[1] = reshape(left, d, d, 1, auxiliary_dimension)
    for site in 2:(N - 1)
        mpo[site] = copy(W)
    end
    right = Array(view(W, :, :, :, 1))
    mpo[N] = reshape(right, d, d, auxiliary_dimension, 1)
    return mpo
end

function _check_triangular_mpo_state(AL, AR, C, H::TriangularMPO)
    chi, d = _check_mixed_canonical(AL, AR, C)
    size(H.W, 1) == d || throw(DimensionMismatch(
        "MPO physical dimension $(size(H.W, 1)) does not match MPS dimension $d",
    ))
    return chi, d, size(H.W, 3)
end

# Environment matrices retain the finite-code orientation `(ket, bra)`.
function _mpo_left_block(X, A, O)
    @tensor Y[r, rb] := X[l, lb] * A[l, r, sin] * O[sout, sin] *
                        conj(A[lb, rb, sout])
    return Y
end

function _mpo_right_block(A, O, X)
    @tensor Y[l, lb] := (A[l, r, sin] * X[r, rb]) * O[sout, sin] *
                        conj(A[lb, rb, sout])
    return Y
end

_mpo_trace_pair(X, Y) = sum(X .* Y)

function _check_mpo_environment_seed(seed, chi, w, name)
    seed === nothing && return nothing
    ndims(seed) == 3 || throw(DimensionMismatch("$name must have three axes"))
    size(seed) == (chi, w, chi) || throw(DimensionMismatch(
        "$name must have size ($chi, $w, $chi); got $(size(seed))",
    ))
    return seed
end

function _mpo_channel_seed(seed, q, T, chi)
    seed === nothing && return zeros(T, chi, chi)
    return Matrix{T}(view(seed, :, q, :))
end

function _relative_equation_residual(lhs, rhs)
    return Float64(norm(lhs - rhs) / max(1.0, Float64(norm(rhs))))
end

function _solve_triangular_mpo_channel(
    transfer,
    source,
    seed,
    lambda,
    fixed,
    dual;
    tol::Real,
    maxiter::Integer,
    krylovdim::Integer,
    unity_tol::Real,
)
    if abs(lambda) <= unity_tol
        result = copy(source)
        return result, zero(promote_type(eltype(source), typeof(lambda))), (
            converged=true,
            residual=0.0,
            numops=0,
            numiter=0,
        )
    end

    is_unity = abs(lambda - one(lambda)) <= unity_tol
    if is_unity
        # Remove the extensive component before inverting I-T.  The added
        # rank-one term fixes the null-space gauge; the returned environment
        # is subsequently put in the zero-gauge convention.
        component = _mpo_trace_pair(source, dual)
        rhs = source - fixed .* component
        if iszero(norm(rhs))
            result = zero.(source)
            return result, component, (
                converged=true,
                residual=0.0,
                numops=0,
                numiter=0,
            )
        end
        operator = X -> X - transfer(X) +
                          fixed .* _mpo_trace_pair(X, dual)
        result, info = _solve_uniform_environment(
            operator,
            rhs,
            seed;
            tol=tol,
            maxiter=maxiter,
            krylovdim=krylovdim,
        )
        result .-= fixed .* _mpo_trace_pair(result, dual)
        residual = _relative_equation_residual(result - transfer(result), rhs)
        return result, component, merge(info, (residual=residual,))
    end

    if iszero(norm(source))
        result = zero.(source)
        component = zero(promote_type(eltype(source), typeof(lambda)))
        return result, component, (
            converged=true,
            residual=0.0,
            numops=0,
            numiter=0,
        )
    end
    operator = X -> X - lambda .* transfer(X)
    result, info = _solve_uniform_environment(
        operator,
        source,
        seed;
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
    residual = _relative_equation_residual(operator(result), source)
    component = zero(promote_type(eltype(source), typeof(lambda)))
    return result, component, merge(info, (residual=residual,))
end

"""
    uniform_mpo_environments(AL, AR, C, H::TriangularMPO; kwargs...)

Construct the left and right thermodynamic-limit Hamiltonian environments of
a mixed-canonical one-site uniform MPS.  The returned `FL` and `FR` tensors use
`(ket bond, MPO bond, bra bond)` ordering.  Singular identity channels are
solved in a projected, zero-gauge convention, and the removed terminal scalar
is returned as the energy density.
"""
function uniform_mpo_environments(
    AL,
    AR,
    C,
    H::TriangularMPO;
    FL0=nothing,
    FR0=nothing,
    L0=nothing,
    R0=nothing,
    transfer_tol::Real=1e-12,
    transfer_maxiter::Integer=500,
    transfer_krylovdim::Integer=30,
    env_tol::Real=1e-12,
    env_maxiter::Integer=200,
    env_krylovdim::Integer=30,
    unity_tol::Real=1e-12,
)
    chi, d, w = _check_triangular_mpo_state(AL, AR, C, H)
    env_tol > 0 || throw(ArgumentError("env_tol must be positive"))
    env_maxiter > 0 || throw(ArgumentError("env_maxiter must be positive"))
    env_krylovdim > 0 || throw(ArgumentError("env_krylovdim must be positive"))
    unity_tol >= 0 || throw(ArgumentError("unity_tol must be nonnegative"))
    _check_mpo_environment_seed(FL0, chi, w, "FL0")
    _check_mpo_environment_seed(FR0, chi, w, "FR0")

    fixed = transfer_fixed_points(
        AL,
        AR;
        L0=L0,
        R0=R0,
        tol=transfer_tol,
        maxiter=transfer_maxiter,
        krylovdim=transfer_krylovdim,
    )
    # `transfer_fixed_points` stores the conventional left density.  The
    # finite Hamiltonian environments use its transpose as their index order.
    left_density = Matrix(transpose(fixed.L))
    right_density = Matrix(fixed.R)

    T = promote_type(eltype(AL), eltype(AR), eltype(C), eltype(H))
    identity_bond = Matrix{T}(I, chi, chi)
    identity_physical = Matrix{T}(I, d, d)
    left_density = Matrix{T}(left_density)
    right_density = Matrix{T}(right_density)
    FL = zeros(T, chi, w, chi)
    FR = zeros(T, chi, w, chi)
    FL[:, w, :] .= identity_bond
    FR[:, 1, :] .= identity_bond

    left_components = zeros(T, w)
    right_components = zeros(T, w)
    left_residuals = zeros(Float64, w)
    right_residuals = zeros(Float64, w)
    left_converged = trues(w)
    right_converged = trues(w)
    environment_numops = 0

    for q in (w - 1):-1:1
        source = zeros(T, chi, chi)
        for qp in (q + 1):w
            source .+= _mpo_left_block(
                view(FL, :, qp, :),
                AL,
                view(H.W, :, :, qp, q),
            )
        end
        transfer = X -> _mpo_left_block(X, AL, identity_physical)
        result, component, info = _solve_triangular_mpo_channel(
            transfer,
            source,
            _mpo_channel_seed(FL0, q, T, chi),
            H.lambda[q],
            identity_bond,
            right_density;
            tol=env_tol,
            maxiter=env_maxiter,
            krylovdim=env_krylovdim,
            unity_tol=unity_tol,
        )
        FL[:, q, :] .= result
        left_components[q] = component
        left_residuals[q] = info.residual
        left_converged[q] = info.converged
        environment_numops += info.numops
    end

    for q in 2:w
        source = zeros(T, chi, chi)
        for qp in 1:(q - 1)
            source .+= _mpo_right_block(
                AR,
                view(H.W, :, :, q, qp),
                view(FR, :, qp, :),
            )
        end
        transfer = X -> _mpo_right_block(AR, identity_physical, X)
        result, component, info = _solve_triangular_mpo_channel(
            transfer,
            source,
            _mpo_channel_seed(FR0, q, T, chi),
            H.lambda[q],
            identity_bond,
            left_density;
            tol=env_tol,
            maxiter=env_maxiter,
            krylovdim=env_krylovdim,
            unity_tol=unity_tol,
        )
        FR[:, q, :] .= result
        right_components[q] = component
        right_residuals[q] = info.residual
        right_converged[q] = info.converged
        environment_numops += info.numops
    end

    left_energy = left_components[1]
    right_energy = right_components[w]
    complex_energy = (left_energy + right_energy) / 2
    environment_residual = max(maximum(left_residuals), maximum(right_residuals))

    return (
        AC=center_tensor(AL, C),
        FL=FL,
        FR=FR,
        L=fixed.L,
        R=fixed.R,
        energy=Float64(real(complex_energy)),
        complex_energy=complex_energy,
        left_energy=left_energy,
        right_energy=right_energy,
        energy_mismatch=Float64(abs(left_energy - right_energy)),
        energy_imaginary_part=Float64(abs(imag(complex_energy))),
        left_channel_components=left_components,
        right_channel_components=right_components,
        left_channel_residuals=left_residuals,
        right_channel_residuals=right_residuals,
        transfer_converged=fixed.converged,
        transfer_residual=max(fixed.residual_L, fixed.residual_R),
        environments_converged=all(left_converged) && all(right_converged),
        environment_residual=environment_residual,
        environment_numops=environment_numops,
    )
end

"""Evaluate the energy density of a mixed-canonical state and triangular MPO."""
function uniform_mpo_energy(AL, AR, C, H::TriangularMPO; kwargs...)
    return uniform_mpo_environments(AL, AR, C, H; kwargs...).energy
end

"""
    bond_expansion_coefficients(AL, AR, C, H::TriangularMPO; kwargs...)

Build the thermodynamic MPO environments and project the corresponding
two-site effective action into the tangent complements.  This is the MPO
counterpart of the `calculateY` branch in `vumps_mpo.m`.
"""
function bond_expansion_coefficients(
    AL,
    AR,
    C,
    H::TriangularMPO;
    VL=left_tangent_basis(AL),
    VR=right_tangent_basis(AR),
    kwargs...,
)
    context = uniform_mpo_environments(AL, AR, C, H; kwargs...)
    action = theta -> apply_heff2(theta, context.FL, H.W, H.W, context.FR)
    return bond_expansion_coefficients(
        AL, AR, C, action; VL=VL, VR=VR,
    )
end

"""Increase a uniform MPS bond dimension using a triangular-MPO residual."""
function expand_bond_dimension(
    AL,
    AR,
    C,
    H::TriangularMPO;
    max_bond::Integer,
    max_add::Integer=typemax(Int),
    atol::Real=1e-14,
    rtol::Real=1e-12,
    kwargs...,
)
    context = uniform_mpo_environments(AL, AR, C, H; kwargs...)
    action = theta -> apply_heff2(theta, context.FL, H.W, H.W, context.FR)
    expansion = expand_bond_dimension(
        AL, AR, C, action;
        max_bond=max_bond, max_add=max_add, atol=atol, rtol=rtol,
    )
    return merge(expansion, (
        environment_energy=context.energy,
        environment_residual=context.environment_residual,
        environments_converged=context.environments_converged,
    ))
end

function _mpo_vumps_local_update(
    AL,
    AR,
    C,
    H::TriangularMPO,
    context;
    eig_tol::Real,
    eig_maxiter::Integer,
    eig_krylovdim::Integer,
)
    energy_AC, ACnew, residual_AC, converged_AC = _lowest_local_eigenpair(
        _one_site_action(context.AC,context.FL,H.W,context.FR),
        context.AC;
        eig_tol=eig_tol,
        maxiter=eig_maxiter,
        krylovdim=eig_krylovdim,
    )
    energy_C, Cnew, residual_C, converged_C = _lowest_local_eigenpair(
        _bond_action(C,context.FL,context.FR),
        C;
        eig_tol=eig_tol,
        maxiter=eig_maxiter,
        krylovdim=eig_krylovdim,
    )
    update = mixed_canonical_update(ACnew, Cnew)
    return merge(update, (
        effective_AC=ACnew,
        effective_energy_AC=energy_AC,
        effective_energy_C=energy_C,
        eig_residual=max(residual_AC, residual_C),
        eig_converged=converged_AC && converged_C,
    ))
end

"""Perform one fixed-bond-dimension VUMPS iteration for a `TriangularMPO`."""
function vumps_step(
    AL,
    AR,
    C,
    H::TriangularMPO;
    FL0=nothing,
    FR0=nothing,
    L0=nothing,
    R0=nothing,
    eig_tol::Real=1e-12,
    eig_maxiter::Integer=500,
    eig_krylovdim::Integer=30,
    transfer_tol::Real=1e-12,
    transfer_maxiter::Integer=500,
    transfer_krylovdim::Integer=30,
    env_tol::Real=1e-12,
    env_maxiter::Integer=200,
    env_krylovdim::Integer=30,
    unity_tol::Real=1e-12,
)
    _check_triangular_mpo_state(AL, AR, C, H)
    context = uniform_mpo_environments(
        AL, AR, C, H;
        FL0=FL0, FR0=FR0, L0=L0, R0=R0,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
        unity_tol=unity_tol,
    )
    update = _mpo_vumps_local_update(
        AL, AR, C, H, context;
        eig_tol=eig_tol,
        eig_maxiter=eig_maxiter,
        eig_krylovdim=eig_krylovdim,
    )
    next_context = uniform_mpo_environments(
        update.AL, update.AR, update.C, H;
        FL0=context.FL, FR0=context.FR, L0=context.L, R0=context.R,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
        unity_tol=unity_tol,
    )
    return (
        AL=update.AL,
        AR=update.AR,
        C=update.C,
        AC=next_context.AC,
        effective_AC=update.effective_AC,
        FL=next_context.FL,
        FR=next_context.FR,
        L=next_context.L,
        R=next_context.R,
        energy=next_context.energy,
        previous_energy=context.energy,
        effective_energy_AC=update.effective_energy_AC,
        effective_energy_C=update.effective_energy_C,
        residual=update.residual,
        residuals=update.residuals,
        eig_residual=update.eig_residual,
        eig_converged=update.eig_converged,
        transfer_converged=next_context.transfer_converged,
        transfer_residual=next_context.transfer_residual,
        environments_converged=next_context.environments_converged,
        environment_residual=next_context.environment_residual,
        energy_mismatch=next_context.energy_mismatch,
        converged_inner=update.eig_converged &&
                        next_context.transfer_converged &&
                        next_context.environments_converged,
    )
end

"""
Iterate VUMPS for a `TriangularMPO`, optionally growing up to `max_bond` from
the projected two-site MPO residual.
"""
function vumps(
    AL,
    AR,
    C,
    H::TriangularMPO;
    maxiter::Integer=100,
    tol::Real=1e-10,
    max_bond::Integer=size(AL, 1),
    expand_tol::Real=1e-1,
    expansion_atol::Real=1e-14,
    expansion_rtol::Real=1e-12,
    expansion_max_add::Integer=typemax(Int),
    FL0=nothing,
    FR0=nothing,
    L0=nothing,
    R0=nothing,
    eig_tol::Real=1e-12,
    eig_maxiter::Integer=500,
    eig_krylovdim::Integer=30,
    transfer_tol::Real=1e-12,
    transfer_maxiter::Integer=500,
    transfer_krylovdim::Integer=30,
    env_tol::Real=1e-12,
    env_maxiter::Integer=200,
    env_krylovdim::Integer=30,
    unity_tol::Real=1e-12,
)
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    expand_tol >= 0 || throw(ArgumentError("expand_tol must be nonnegative"))
    initial_bond, _, _ = _check_triangular_mpo_state(AL, AR, C, H)
    max_bond >= initial_bond || throw(ArgumentError(
        "max_bond ($max_bond) cannot be smaller than the input bond dimension ($initial_bond)",
    ))

    current_AL = copy(AL)
    current_AR = copy(AR)
    current_C = copy(C)
    context = uniform_mpo_environments(
        current_AL, current_AR, current_C, H;
        FL0=FL0, FR0=FR0, L0=L0, R0=R0,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
        unity_tol=unity_tol,
    )
    history = NamedTuple[]
    growth_history = NamedTuple[]
    final_update = nothing
    converged = false
    initial_residuals = infinite_residuals(current_AL, current_AR, current_C)
    previous_residual = max(initial_residuals.center_left,
                            initial_residuals.center_right)

    for iteration in 1:maxiter
        growth_attempted = false
        growth_happened = false
        if size(current_C, 1) < max_bond && previous_residual <= expand_tol
            growth_attempted = true
            action = theta -> apply_heff2(
                theta, context.FL, H.W, H.W, context.FR,
            )
            expansion = expand_bond_dimension(
                current_AL, current_AR, current_C, action;
                max_bond=max_bond,
                max_add=expansion_max_add,
                atol=expansion_atol,
                rtol=expansion_rtol,
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
                context = uniform_mpo_environments(
                    current_AL, current_AR, current_C, H;
                    transfer_tol=transfer_tol,
                    transfer_maxiter=transfer_maxiter,
                    transfer_krylovdim=transfer_krylovdim,
                    env_tol=env_tol,
                    env_maxiter=env_maxiter,
                    env_krylovdim=env_krylovdim,
                    unity_tol=unity_tol,
                )
            end
        end
        update = _mpo_vumps_local_update(
            current_AL, current_AR, current_C, H, context;
            eig_tol=eig_tol,
            eig_maxiter=eig_maxiter,
            eig_krylovdim=eig_krylovdim,
        )
        current_AL, current_AR, current_C = update.AL, update.AR, update.C
        context = uniform_mpo_environments(
            current_AL, current_AR, current_C, H;
            FL0=context.FL, FR0=context.FR, L0=context.L, R0=context.R,
            transfer_tol=transfer_tol,
            transfer_maxiter=transfer_maxiter,
            transfer_krylovdim=transfer_krylovdim,
            env_tol=env_tol,
            env_maxiter=env_maxiter,
            env_krylovdim=env_krylovdim,
            unity_tol=unity_tol,
        )
        push!(history, (
            iteration=iteration,
            bond_dimension=size(current_C, 1),
            energy=context.energy,
            residual=update.residual,
            eig_residual=update.eig_residual,
            transfer_residual=context.transfer_residual,
            environment_residual=context.environment_residual,
            energy_mismatch=context.energy_mismatch,
        ))
        final_update = update
        previous_residual = update.residual
        fixed_point_converged = update.residual <= tol && update.eig_converged &&
                                context.transfer_converged &&
                                context.environments_converged
        adaptive_complete = size(current_C, 1) >= max_bond ||
                            previous_residual > expand_tol ||
                            (growth_attempted && !growth_happened)
        converged = fixed_point_converged && adaptive_complete
        converged && break
    end

    return (
        AL=current_AL,
        AR=current_AR,
        C=current_C,
        AC=context.AC,
        FL=context.FL,
        FR=context.FR,
        L=context.L,
        R=context.R,
        energy=context.energy,
        residual=final_update.residual,
        residuals=final_update.residuals,
        eig_residual=final_update.eig_residual,
        converged=converged,
        bond_dimension=size(current_C, 1),
        iterations=length(history),
        history=history,
        growth_history=growth_history,
        transfer_converged=context.transfer_converged,
        transfer_residual=context.transfer_residual,
        environments_converged=context.environments_converged,
        environment_residual=context.environment_residual,
        energy_mismatch=context.energy_mismatch,
    )
end

function vumps(
    H::TriangularMPO,
    chi::Integer;
    d::Integer=size(H.W, 1),
    rng::AbstractRNG=Random.default_rng(),
    T::Type{<:Number}=ComplexF64,
    kwargs...,
)
    d == size(H.W, 1) || throw(DimensionMismatch(
        "d=$d does not match the MPO physical dimension $(size(H.W, 1))",
    ))
    initial = random_infinite_mps(chi, d; rng=rng, T=T)
    return vumps(initial.AL, initial.AR, initial.C, H; kwargs...)
end

"""Take one uniform-TDVP step for a `TriangularMPO`, with optional growth."""
function infinite_tdvp_step(
    AL,
    AR,
    C,
    H::TriangularMPO,
    dt::Number;
    time::Symbol=:real,
    tol::Real=1e-12,
    maxiter::Integer=100,
    krylovdim::Integer=30,
    FL0=nothing,
    FR0=nothing,
    L0=nothing,
    R0=nothing,
    transfer_tol::Real=1e-12,
    transfer_maxiter::Integer=500,
    transfer_krylovdim::Integer=30,
    env_tol::Real=1e-12,
    env_maxiter::Integer=200,
    env_krylovdim::Integer=30,
    unity_tol::Real=1e-12,
    max_bond::Integer=size(AL, 1),
    expansion_atol::Real=1e-14,
    expansion_rtol::Real=1e-12,
    expansion_max_add::Integer=typemax(Int),
)
    chi, _, _ = _check_triangular_mpo_state(AL, AR, C, H)
    max_bond >= chi || throw(ArgumentError(
        "max_bond ($max_bond) cannot be smaller than the input bond dimension ($chi)",
    ))
    parameter = _tdvp_parameter(dt, time)
    context = uniform_mpo_environments(
        AL, AR, C, H;
        FL0=FL0, FR0=FR0, L0=L0, R0=R0,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
        unity_tol=unity_tol,
    )
    action = theta -> apply_heff2(
        theta, context.FL, H.W, H.W, context.FR,
    )
    expansion = expand_bond_dimension(
        AL, AR, C, action;
        max_bond=max_bond,
        max_add=expansion_max_add,
        atol=expansion_atol,
        rtol=expansion_rtol,
    )
    if expansion.expanded
        AL, AR, C = expansion.AL, expansion.AR, expansion.C
        context = uniform_mpo_environments(
            AL, AR, C, H;
            transfer_tol=transfer_tol,
            transfer_maxiter=transfer_maxiter,
            transfer_krylovdim=transfer_krylovdim,
            env_tol=env_tol,
            env_maxiter=env_maxiter,
            env_krylovdim=env_krylovdim,
            unity_tol=unity_tol,
        )
    end
    ACnew, info_AC = _exponentiate_tensor(
        _one_site_action(context.AC,context.FL,H.W,context.FR),
        context.AC,
        parameter;
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
    Cnew, info_C = _exponentiate_tensor(
        _bond_action(C,context.FL,context.FR),
        C,
        parameter;
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
    update = mixed_canonical_update(ACnew, Cnew)
    next_context = uniform_mpo_environments(
        update.AL, update.AR, update.C, H;
        FL0=context.FL, FR0=context.FR, L0=context.L, R0=context.R,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
        unity_tol=unity_tol,
    )
    return (
        AL=update.AL,
        AR=update.AR,
        C=update.C,
        AC=next_context.AC,
        evolved_AC=ACnew,
        FL=next_context.FL,
        FR=next_context.FR,
        L=next_context.L,
        R=next_context.R,
        energy_before=context.energy,
        energy_after=next_context.energy,
        residual=update.residual,
        residuals=update.residuals,
        bond_dimension=size(update.C, 1),
        expansion=expansion,
        krylov_converged=info_AC.converged && info_C.converged,
        krylov_error=max(info_AC.error, info_C.error),
        transfer_converged=next_context.transfer_converged,
        transfer_residual=next_context.transfer_residual,
        environments_converged=next_context.environments_converged,
        environment_residual=next_context.environment_residual,
        energy_mismatch=next_context.energy_mismatch,
    )
end

# Explicit names make the MATLAB-to-Julia mapping discoverable while the typed
# overloads above preserve the package's existing high-level API.
vumps_mpo_step(AL, AR, C, H::TriangularMPO; kwargs...) =
    vumps_step(AL, AR, C, H; kwargs...)
vumps_mpo(AL, AR, C, H::TriangularMPO; kwargs...) =
    vumps(AL, AR, C, H; kwargs...)
vumps_mpo(H::TriangularMPO, chi::Integer; kwargs...) =
    vumps(H, chi; kwargs...)
infinite_tdvp_mpo_step(AL, AR, C, H::TriangularMPO, dt::Number; kwargs...) =
    infinite_tdvp_step(AL, AR, C, H, dt; kwargs...)
