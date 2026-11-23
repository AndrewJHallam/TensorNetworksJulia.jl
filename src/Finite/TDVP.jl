# Finite projector-splitting time-dependent variational principle algorithms.

"""
    apply_keff(C, FL, FR)

Apply the zero-site (bond) effective Hamiltonian used by projector-splitting
TDVP.  `C` is a centre matrix and `FL`/`FR` use the package environment order
`(ket bond, MPO bond, bra bond)`.
"""
function apply_keff(
    C::AbstractMatrix{<:Number},
    FL::AbstractArray{<:Number,3},
    FR::AbstractArray{<:Number,3},
)
    size(FL, 1) == size(C, 1) || throw(DimensionMismatch(
        "FL ket bond and C left bond differ"))
    size(FR, 1) == size(C, 2) || throw(DimensionMismatch(
        "FR ket bond and C right bond differ"))
    size(FL, 2) == size(FR, 2) || throw(DimensionMismatch(
        "left and right environment MPO bonds differ"))
    size(FL, 3) == size(C, 1) || throw(DimensionMismatch(
        "FL bra bond and C left bond differ"))
    size(FR, 3) == size(C, 2) || throw(DimensionMismatch(
        "FR bra bond and C right bond differ"))

    @tensor KC[alout, arout] := FL[alin, w, alout] *
                                C[alin, arin] *
                                FR[arin, w, arout]
    return KC
end

function _tdvp_parameter(dt::Number, time::Symbol)
    if time === :real
        return -im * dt
    elseif time in (:imag, :imaginary, :gradient)
        return -dt
    end
    throw(ArgumentError(
        "time must be :real, :imaginary (or :imag), or :gradient",
    ))
end

function _check_tdvp_options(tol, maxiter, krylovdim)
    tol > 0 || throw(ArgumentError("tol must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    krylovdim > 0 || throw(ArgumentError("krylovdim must be positive"))
    return nothing
end

function _check_tdvp_scalar_type(psi, H, parameter)
    Tstate = _promoted_eltype(psi)
    Tresult = promote_type(Tstate, _promoted_eltype(H), typeof(parameter))
    Tresult <: Tstate || throw(ArgumentError(
        "the MPS element type $Tstate cannot store TDVP results of type " *
        "$Tresult; use a complex-valued MPS (for example T=ComplexF64) " *
        "for real-time or complex-Hamiltonian evolution",
    ))
    return nothing
end

function _exponentiate_tensor(
    op,
    tensor::AbstractArray,
    parameter::Number;
    tol::Real,
    maxiter::Integer,
    krylovdim::Integer,
)
    _check_tdvp_options(tol, maxiter, krylovdim)
    dims = size(tensor)
    x = vec(copy(tensor))

    if iszero(parameter) || iszero(norm(x))
        return copy(tensor), (
            converged=true,
            error=0.0,
            numops=0,
            numiter=0,
        )
    end

    vector_op = v -> vec(op(reshape(v, dims)))
    if length(x) == 1
        Ax = vector_op(x)
        lambda = dot(x, Ax) / dot(x, x)
        residual = norm(Ax - lambda * x)
        result = exp(parameter * lambda) .* x
        return copy(reshape(result, dims)), (
            converged=true,
            error=Float64(residual),
            numops=1,
            numiter=1,
        )
    end

    kdim = min(length(x), max(2, Int(krylovdim)))
    result, info = exponentiate(
        vector_op,
        parameter,
        x;
        ishermitian=true,
        tol=tol,
        maxiter=maxiter,
        krylovdim=kdim,
    )
    return copy(reshape(result, dims)), (
        converged=info.converged == 1,
        error=Float64(info.normres),
        numops=info.numops,
        numiter=info.numiter,
    )
end

function _record_exponential!(convergence, errors, operations, iterations, info)
    push!(convergence, info.converged)
    push!(errors, info.error)
    push!(operations, info.numops)
    push!(iterations, info.numiter)
    return nothing
end

function _evolve_one_site(
    A,
    FL,
    W,
    FR,
    dt;
    time,
    tol,
    maxiter,
    krylovdim,
)
    parameter = _tdvp_parameter(dt, time)
    return _exponentiate_tensor(
        _one_site_action(A,FL,W,FR),
        A,
        parameter;
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
end

function _evolve_two_site(
    AA,
    FL,
    W1,
    W2,
    FR,
    dt;
    time,
    tol,
    maxiter,
    krylovdim,
)
    parameter = _tdvp_parameter(dt, time)
    return _exponentiate_tensor(
        _two_site_action(AA,FL,W1,W2,FR),
        AA,
        parameter;
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
end

function _evolve_bond(
    C,
    FL,
    FR,
    dt;
    time,
    tol,
    maxiter,
    krylovdim,
)
    parameter = _tdvp_parameter(dt, time)
    return _exponentiate_tensor(
        _bond_action(C,FL,FR),
        C,
        parameter;
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
end

function _tdvp_energy(psi, H, norm_squared)
    return Float64(real(expectation(psi, H) / norm_squared))
end

function _tdvp_diagnostics(
    psi,
    H,
    energy_before,
    norm_before,
    convergence,
    errors,
    operations,
    iterations;
    extras...,
)
    norm_after = Float64(real(overlap(psi, psi)))
    energy_after = _tdvp_energy(psi, H, norm_after)
    base = (
        energy_before=energy_before,
        energy_after=energy_after,
        norm_before=norm_before,
        norm_after=norm_after,
        krylov_converged=all(convergence),
        krylov_errors=errors,
        num_exponentials=length(convergence),
        num_operator_applications=sum(operations),
        num_krylov_iterations=sum(iterations),
    )
    return merge(base, (; extras...))
end

"""
    tdvp1_step!(psi, H, dt; time=:real, tol=1e-12,
                maxiter=100, krylovdim=30)

Advance an open-boundary MPS by one symmetric one-site projector-splitting
TDVP step.  The traversal and signs follow the canonical MATLAB
`tdvp_finite.m`: site centres evolve forward for half steps, bond centres
evolve backward for half steps, and the left boundary site receives the full
step.  `psi` is mutated and its bond dimensions are unchanged.

For `time=:real`, the local exponential is `exp(-im*dt*H_eff)`; for
`time=:imaginary`, it is `exp(-dt*H_eff)`.  Real-time evolution requires a
complex-valued MPS container.  As in the MATLAB QR sweep, the final scalar
centre is discarded, fixing global phase and normalizing the represented
state.
"""
function tdvp1_step!(
    psi::AbstractVector,
    H::AbstractVector,
    dt::Number;
    time::Symbol=:real,
    tol::Real=1e-12,
    maxiter::Integer=100,
    krylovdim::Integer=30,
)
    N = _check_chain(psi, H)
    _check_tdvp_options(tol, maxiter, krylovdim)
    parameter = _tdvp_parameter(dt, time)
    _check_tdvp_scalar_type(psi, H, parameter)

    norm_before = Float64(real(overlap(psi, psi)))
    norm_before > 0 || throw(ArgumentError("the input MPS has zero norm"))
    energy_before = _tdvp_energy(psi, H, norm_before)
    FL = collect(left_environments(psi, H))
    FR = collect(right_environments(psi, H))

    convergence = Bool[]
    errors = Float64[]
    operations = Int[]
    iterations = Int[]
    C = ones(eltype(last(psi)), 1, 1)

    # Right boundary to left boundary: forward site half-step followed by
    # backward bond half-step.
    for n in N:-1:2
        AC = _absorb_right(psi[n], C)
        AC, info = _evolve_one_site(
            AC, FL[n], H[n], FR[n + 1], dt / 2;
            time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        _record_exponential!(convergence, errors, operations, iterations, info)
        C, AR = right_orthogonalize(AC)
        psi[n] = AR
        FR[n] = right_environment(AR, H[n], FR[n + 1])

        C, info = _evolve_bond(
            C, FL[n], FR[n], -dt / 2;
            time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        _record_exponential!(convergence, errors, operations, iterations, info)
    end

    # The left boundary site receives a full step and reverses the sweep.
    AC = _absorb_right(psi[1], C)
    AC, info = _evolve_one_site(
        AC, FL[1], H[1], FR[2], dt;
        time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    _record_exponential!(convergence, errors, operations, iterations, info)
    C, AL = left_orthogonalize(AC)
    psi[1] = AL
    FL[2] = left_environment(FL[1], AL, H[1])

    # Left boundary back to the right boundary.
    for n in 2:N
        C, info = _evolve_bond(
            C, FL[n], FR[n], -dt / 2;
            time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        _record_exponential!(convergence, errors, operations, iterations, info)
        AC = _absorb_left(C, psi[n])
        AC, info = _evolve_one_site(
            AC, FL[n], H[n], FR[n + 1], dt / 2;
            time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        _record_exponential!(convergence, errors, operations, iterations, info)
        C, AL = left_orthogonalize(AC)
        psi[n] = AL
        FL[n + 1] = left_environment(FL[n], AL, H[n])
    end

    return _tdvp_diagnostics(
        psi, H, energy_before, norm_before,
        convergence, errors, operations, iterations,
    )
end

"""
    tdvp2_step!(psi, H, dt; time=:real, maxdim=typemax(Int), cutoff=0,
                tol=1e-12, maxiter=100, krylovdim=30)

Advance an open-boundary MPS by one symmetric two-site projector-splitting
TDVP step.  Two-site centres evolve forward while the retained one-site
centre evolves backward, exactly mirroring `tdvp_finite_2site.m`.  SVD splits
obey `maxdim` and the relative discarded-weight `cutoff`.
"""
function tdvp2_step!(
    psi::AbstractVector,
    H::AbstractVector,
    dt::Number;
    time::Symbol=:real,
    maxdim::Integer=typemax(Int),
    cutoff::Real=0.0,
    tol::Real=1e-12,
    maxiter::Integer=100,
    krylovdim::Integer=30,
)
    N = _check_chain(psi, H)
    N >= 2 || throw(ArgumentError("two-site TDVP requires at least two sites"))
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative"))
    _check_tdvp_options(tol, maxiter, krylovdim)
    parameter = _tdvp_parameter(dt, time)
    _check_tdvp_scalar_type(psi, H, parameter)

    norm_before = Float64(real(overlap(psi, psi)))
    norm_before > 0 || throw(ArgumentError("the input MPS has zero norm"))
    energy_before = _tdvp_energy(psi, H, norm_before)
    FL = collect(left_environments(psi, H))
    FR = collect(right_environments(psi, H))

    convergence = Bool[]
    errors = Float64[]
    operations = Int[]
    iterations = Int[]
    discarded_weight = 0.0
    AC = psi[N]

    # Right-to-left half sweep.  The one-site centre left by each SVD is
    # evolved backward to remove the overlap between neighbouring projectors.
    for n in (N - 1):-1:2
        AA = contract_two_sites(psi[n], AC)
        AA, info = _evolve_two_site(
            AA, FL[n], H[n], H[n + 1], FR[n + 2], dt / 2;
            time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        _record_exponential!(convergence, errors, operations, iterations, info)
        AC, AR, discarded = split_right(AA; maxdim=maxdim, cutoff=cutoff)
        psi[n + 1] = AR
        discarded_weight = max(discarded_weight, Float64(real(discarded)))
        FR[n + 1] = right_environment(AR, H[n + 1], FR[n + 2])

        AC, info = _evolve_one_site(
            AC, FL[n], H[n], FR[n + 1], -dt / 2;
            time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        _record_exponential!(convergence, errors, operations, iterations, info)
    end

    # Full two-site step at the left boundary and sweep reversal.
    AA = contract_two_sites(psi[1], AC)
    AA, info = _evolve_two_site(
        AA, FL[1], H[1], H[2], FR[3], dt;
        time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    _record_exponential!(convergence, errors, operations, iterations, info)
    AL, AC, discarded = split_left(AA; maxdim=maxdim, cutoff=cutoff)
    psi[1] = AL
    discarded_weight = max(discarded_weight, Float64(real(discarded)))
    FL[2] = left_environment(FL[1], AL, H[1])

    # Left-to-right half sweep.
    for n in 2:(N - 1)
        AC, info = _evolve_one_site(
            AC, FL[n], H[n], FR[n + 1], -dt / 2;
            time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        _record_exponential!(convergence, errors, operations, iterations, info)
        AA = contract_two_sites(AC, psi[n + 1])
        AA, info = _evolve_two_site(
            AA, FL[n], H[n], H[n + 1], FR[n + 2], dt / 2;
            time=time, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        _record_exponential!(convergence, errors, operations, iterations, info)
        AL, AC, discarded = split_left(AA; maxdim=maxdim, cutoff=cutoff)
        psi[n] = AL
        discarded_weight = max(discarded_weight, Float64(real(discarded)))
        FL[n + 1] = left_environment(FL[n], AL, H[n])
    end

    # Complete the left-canonical sweep.  Discarding this final scalar is the
    # same normalization/global-phase convention as the MATLAB routine.
    _, AL = left_orthogonalize(AC)
    psi[N] = AL

    max_bond = maximum(size(psi[n], 2) for n in 1:(N - 1))
    return _tdvp_diagnostics(
        psi, H, energy_before, norm_before,
        convergence, errors, operations, iterations;
        discarded_weight=discarded_weight,
        max_bond=max_bond,
    )
end
