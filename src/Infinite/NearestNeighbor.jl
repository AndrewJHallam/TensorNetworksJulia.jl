# VUMPS and uniform TDVP for nearest-neighbour Hamiltonians.

_tensor_trace_pair(X, Y) = sum(X .* Y)

# Tensor-network left transfer action in the index orientation used by the
# MATLAB Hamiltonian environments.  For a conventional density matrix `L`,
# this map acts on `transpose(L)`.
function _environment_transfer_left(A, X)
    @tensor Y[r, rb] := X[l, lb] * A[l, r, s] * conj(A[lb, rb, s])
    return Y
end

function _left_hamiltonian_source(AL, h)
    @tensor source[r, rb] := AL[l, m, si] * AL[m, r, ti] *
                             h[so, to, si, ti] *
                             conj(AL[l, mb, so]) * conj(AL[mb, rb, to])
    return source
end

function _right_hamiltonian_source(AR, h)
    # Close the right bond before introducing the left bra tensor.
    @tensor density[m, mb, ti, to] := AR[m, r, ti] * conj(AR[mb, r, to])
    @tensor weighted[m, mb, si, so] := density[m, mb, ti, to] * h[so, to, si, ti]
    @tensor source[l, lb] := (AR[l, m, si] * weighted[m, mb, si, so]) *
                             conj(AR[lb, mb, so])
    return source
end

function _solve_uniform_environment(
    op,
    source,
    seed;
    tol::Real,
    maxiter::Integer,
    krylovdim::Integer,
)
    tol > 0 || throw(ArgumentError("environment tolerance must be positive"))
    maxiter > 0 || throw(ArgumentError("environment maxiter must be positive"))
    krylovdim > 0 || throw(ArgumentError("environment krylovdim must be positive"))
    dims = size(source)
    length(source) == 1 && begin
        basis = ones(eltype(source), dims)
        coefficient = only(op(basis))
        abs(coefficient) > eps(Float64) || throw(ArgumentError(
            "singular one-dimensional environment equation",
        ))
        result = source ./ coefficient
        residual = norm(source - op(result))
        return result, (
            converged=residual <= tol * max(1, norm(source)),
            residual=Float64(residual),
            numops=1,
            numiter=1,
        )
    end

    vector_op = v -> vec(op(reshape(v, dims)))
    kdim = min(length(source), max(2, Int(krylovdim)))
    solution, info = linsolve(
        vector_op,
        vec(source),
        vec(seed);
        tol=tol,
        maxiter=maxiter,
        krylovdim=kdim,
    )
    return copy(reshape(solution, dims)), (
        converged=info.converged == 1,
        residual=Float64(info.normres),
        numops=info.numops,
        numiter=info.numiter,
    )
end

function _uniform_context(
    AL,
    AR,
    C,
    h;
    HL0=nothing,
    HR0=nothing,
    L0=nothing,
    R0=nothing,
    transfer_tol::Real=1e-12,
    transfer_maxiter::Integer=500,
    transfer_krylovdim::Integer=30,
    env_tol::Real=1e-12,
    env_maxiter::Integer=200,
    env_krylovdim::Integer=30,
)
    chi, _ = _check_mixed_canonical(AL, AR, C; h=h)
    fixed = transfer_fixed_points(
        AL,
        AR;
        L0=L0,
        R0=R0,
        tol=transfer_tol,
        maxiter=transfer_maxiter,
        krylovdim=transfer_krylovdim,
    )
    L_tensor = transpose(fixed.L)
    identity_matrix = Matrix{promote_type(eltype(AL), eltype(h))}(I, chi, chi)
    source_left = _left_hamiltonian_source(AL, h)
    source_right = _right_hamiltonian_source(AR, h)
    left_seed = HL0 === nothing ? zeros(eltype(source_left), chi, chi) :
                                  Matrix{eltype(source_left)}(HL0)
    right_seed = HR0 === nothing ? zeros(eltype(source_right), chi, chi) :
                                   Matrix{eltype(source_right)}(HR0)

    left_operator = X -> X - _environment_transfer_left(AL, X) +
                         identity_matrix .* _tensor_trace_pair(X, fixed.R)
    right_operator = X -> X - transfer_right(AR, X) +
                          identity_matrix .* _tensor_trace_pair(L_tensor, X)
    HL, left_info = _solve_uniform_environment(
        left_operator,
        source_left,
        left_seed;
        tol=env_tol,
        maxiter=env_maxiter,
        krylovdim=env_krylovdim,
    )
    HR, right_info = _solve_uniform_environment(
        right_operator,
        source_right,
        right_seed;
        tol=env_tol,
        maxiter=env_maxiter,
        krylovdim=env_krylovdim,
    )
    HL .-= identity_matrix .* _tensor_trace_pair(HL, fixed.R)
    HR .-= identity_matrix .* _tensor_trace_pair(L_tensor, HR)

    return (
        AC=center_tensor(AL, C),
        HL=HL,
        HR=HR,
        L=fixed.L,
        R=fixed.R,
        energy=Float64(real(uniform_energy(AL, AR, C, h))),
        transfer_converged=fixed.converged,
        transfer_residual=max(fixed.residual_L, fixed.residual_R),
        environments_converged=left_info.converged && right_info.converged,
        environment_residual=max(left_info.residual, right_info.residual),
        environment_numops=left_info.numops + right_info.numops,
    )
end

"""
    apply_hac(AC, AL, AR, h, HL, HR)

Apply the VUMPS one-site-centre effective Hamiltonian.  The four terms are
the right bond interaction, left bond interaction, and the two half-infinite
Hamiltonian environments.
"""
function apply_hac(AC, AL, AR, h, HL, HR)
    chi, d = _check_mixed_canonical(AL, AR, Matrix{eltype(AC)}(I, size(AC, 1),
                                                               size(AC, 2)); h=h)
    size(AC) == (chi, chi, d) || throw(DimensionMismatch(
        "AC must have size ($chi, $chi, $d)",
    ))
    size(HL) == (chi, chi) || throw(DimensionMismatch("HL has the wrong size"))
    size(HR) == (chi, chi) || throw(DimensionMismatch("HR has the wrong size"))

    @tensor right_term[l, rb, so] := AC[l, m, si] * AR[m, r, ti] *
                                      h[so, to, si, ti] * conj(AR[rb, r, to])
    @tensor left_term[lb, r, to] := AL[l, m, si] * AC[m, r, ti] *
                                     h[so, to, si, ti] * conj(AL[l, lb, so])
    @tensor left_environment[lb, r, s] := HL[l, lb] * AC[l, r, s]
    @tensor right_environment_term[l, rb, s] := AC[l, r, s] * HR[r, rb]
    return right_term + left_term + left_environment + right_environment_term
end

"""Apply the VUMPS centre-matrix effective Hamiltonian."""
function apply_hc(C, AL, AR, h, HL, HR)
    chi, _ = _check_mixed_canonical(AL, AR, C; h=h)
    size(HL) == (chi, chi) || throw(DimensionMismatch("HL has the wrong size"))
    size(HR) == (chi, chi) || throw(DimensionMismatch("HR has the wrong size"))

    @tensor interaction[lb, rb] := AL[l, m, si] * C[m, n] *
                                    AR[n, r, ti] * h[so, to, si, ti] *
                                    conj(AL[l, lb, so]) * conj(AR[rb, r, to])
    @tensor left_environment[lb, r] := HL[l, lb] * C[l, r]
    @tensor right_environment_term[l, rb] := C[l, r] * HR[r, rb]
    return interaction + left_environment + right_environment_term
end

"""
    vumps_step(AL, AR, C, h; kwargs...)

Perform one fixed-bond-dimension VUMPS iteration.  This is the direct Julia
counterpart of the non-adaptive core of `VUMPS/vumps.m`.
"""
function vumps_step(
    AL,
    AR,
    C,
    h;
    HL0=nothing,
    HR0=nothing,
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
)
    _check_mixed_canonical(AL, AR, C; h=h)
    context = _uniform_context(
        AL, AR, C, h;
        HL0=HL0, HR0=HR0, L0=L0, R0=R0,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
    )
    energy_AC, ACnew, residual_AC, converged_AC = _lowest_local_eigenpair(
        X -> apply_hac(X, AL, AR, h, context.HL, context.HR),
        context.AC;
        eig_tol=eig_tol,
        maxiter=eig_maxiter,
        krylovdim=eig_krylovdim,
    )
    energy_C, Cnew, residual_C, converged_C = _lowest_local_eigenpair(
        X -> apply_hc(X, AL, AR, h, context.HL, context.HR),
        C;
        eig_tol=eig_tol,
        maxiter=eig_maxiter,
        krylovdim=eig_krylovdim,
    )
    update = mixed_canonical_update(ACnew, Cnew)
    energy = Float64(real(uniform_energy(update.AL, update.AR, update.C, h)))
    return (
        AL=update.AL,
        AR=update.AR,
        C=update.C,
        AC=ACnew,
        HL=context.HL,
        HR=context.HR,
        L=context.L,
        R=context.R,
        energy=energy,
        previous_energy=context.energy,
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
    vumps(AL, AR, C, h; maxiter=100, tol=1e-10,
          max_bond=size(AL, 1), kwargs...)
    vumps(h, chi; d=size(h,1), rng=Random.default_rng(), T=ComplexF64, kwargs...)

Iterate VUMPS until the centre-consistency residual falls below `tol`.  The
default `max_bond` equals the input bond dimension and therefore gives the
fixed-dimension algorithm.  If `max_bond` is larger, a projected two-site
residual expansion is attempted whenever the previous consistency residual is
at most `expand_tol`; cached environments are reset after every growth event.
"""
function vumps(
    AL,
    AR,
    C,
    h;
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
    initial_bond, _ = _check_mixed_canonical(AL, AR, C; h=h)
    max_bond >= initial_bond || throw(ArgumentError(
        "max_bond ($max_bond) cannot be smaller than the input bond dimension ($initial_bond)",
    ))

    current_AL = copy(AL)
    current_AR = copy(AR)
    current_C = copy(C)
    HL = HR = L = R = nothing
    history = NamedTuple[]
    growth_history = NamedTuple[]
    final_step = nothing
    converged = false
    previous_residuals = infinite_residuals(current_AL, current_AR, current_C)
    previous_residual = max(previous_residuals.center_left,
                            previous_residuals.center_right)

    for iteration in 1:maxiter
        growth_attempted = false
        growth_happened = false
        if size(current_AL, 1) < max_bond && previous_residual <= expand_tol
            growth_attempted = true
            expansion = expand_bond_dimension(
                current_AL, current_AR, current_C, h;
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
                current_AL = expansion.AL
                current_AR = expansion.AR
                current_C = expansion.C
                # Environment and transfer seeds have the old dimensions.
                HL = HR = L = R = nothing
            end
        end
        step = vumps_step(
            current_AL, current_AR, current_C, h;
            HL0=HL, HR0=HR, L0=L, R0=R, kwargs...,
        )
        current_AL, current_AR, current_C = step.AL, step.AR, step.C
        HL, HR, L, R = step.HL, step.HR, step.L, step.R
        push!(history, (
            iteration=iteration,
            bond_dimension=size(step.C, 1),
            energy=step.energy,
            residual=step.residual,
            eig_residual=step.eig_residual,
            transfer_residual=step.transfer_residual,
            environment_residual=step.environment_residual,
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

    final_context = _uniform_context(
        current_AL, current_AR, current_C, h;
        HL0=HL, HR0=HR, L0=L, R0=R,
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
        energy=final_context.energy,
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

function vumps(
    h::AbstractArray{<:Number,4},
    chi::Integer;
    d::Integer=size(h, 1),
    rng::AbstractRNG=Random.default_rng(),
    T::Type{<:Number}=ComplexF64,
    kwargs...,
)
    initial = random_infinite_mps(chi, d; rng=rng, T=T)
    return vumps(initial.AL, initial.AR, initial.C, h; kwargs...)
end

"""
    infinite_tdvp_step(AL, AR, C, h, dt; time=:real, kwargs...)

Take one fixed-bond-dimension uniform-TDVP step by evolving the `AC` and `C`
effective problems and reconstructing the canonical tensors.  The centre uses
the same exponential sign as `AC`, matching the canonical MATLAB `vtdvp.m`.
"""
function infinite_tdvp_step(
    AL,
    AR,
    C,
    h,
    dt::Number;
    time::Symbol=:real,
    tol::Real=1e-12,
    maxiter::Integer=100,
    krylovdim::Integer=30,
    HL0=nothing,
    HR0=nothing,
    L0=nothing,
    R0=nothing,
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
    chi, _ = _check_mixed_canonical(AL, AR, C; h=h)
    max_bond >= chi || throw(ArgumentError(
        "max_bond ($max_bond) cannot be smaller than the input bond dimension ($chi)",
    ))
    expansion = expand_bond_dimension(
        AL, AR, C, h;
        max_bond=max_bond,
        max_add=expansion_max_add,
        atol=expansion_atol,
        rtol=expansion_rtol,
    )
    if expansion.expanded
        AL, AR, C = expansion.AL, expansion.AR, expansion.C
        # Seeds supplied by a caller refer to the pre-expansion bond space.
        HL0 = HR0 = L0 = R0 = nothing
    end
    parameter = _tdvp_parameter(dt, time)
    context = _uniform_context(
        AL, AR, C, h;
        HL0=HL0, HR0=HR0, L0=L0, R0=R0,
        transfer_tol=transfer_tol,
        transfer_maxiter=transfer_maxiter,
        transfer_krylovdim=transfer_krylovdim,
        env_tol=env_tol,
        env_maxiter=env_maxiter,
        env_krylovdim=env_krylovdim,
    )
    ACnew, info_AC = _exponentiate_tensor(
        X -> apply_hac(X, AL, AR, h, context.HL, context.HR),
        context.AC,
        parameter;
        tol=tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    Cnew, info_C = _exponentiate_tensor(
        X -> apply_hc(X, AL, AR, h, context.HL, context.HR),
        C,
        parameter;
        tol=tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    update = mixed_canonical_update(ACnew, Cnew)
    energy_after = Float64(real(uniform_energy(update.AL, update.AR, update.C, h)))
    return (
        AL=update.AL,
        AR=update.AR,
        C=update.C,
        AC=ACnew,
        HL=context.HL,
        HR=context.HR,
        L=context.L,
        R=context.R,
        energy_before=context.energy,
        energy_after=energy_after,
        residual=update.residual,
        residuals=update.residuals,
        bond_dimension=size(update.C, 1),
        expansion=expansion,
        krylov_converged=info_AC.converged && info_C.converged,
        krylov_error=max(info_AC.error, info_C.error),
        transfer_converged=context.transfer_converged,
        environments_converged=context.environments_converged,
    )
end

vtdvp_step(args...; kwargs...) = infinite_tdvp_step(args...; kwargs...)
