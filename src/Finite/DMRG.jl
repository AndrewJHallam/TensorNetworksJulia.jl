# Finite open-boundary density-matrix renormalization group algorithms.

"""
    apply_heff1(A, FL, W, FR)

Apply the one-site effective Hamiltonian without forming it explicitly.

The index order is the package convention corresponding to the MATLAB
`apply_H_to_A` network: `A[left, right, physical]`,
`W[physical_out, physical_in, mpo_left, mpo_right]`, and
`FL`/`FR[ket, mpo, bra]`.  The result has the same index order as `A`.
"""
function apply_heff1(
    A::AbstractArray{<:Number,3},
    FL::AbstractArray{<:Number,3},
    W::AbstractArray{<:Number,4},
    FR::AbstractArray{<:Number,3},
)
    @tensor HA[alout, arout, sout] := FL[alin, wl, alout] *
        A[alin, arin, sin] * W[sout, sin, wl, wr] * FR[arin, wr, arout]
    return HA
end

"""
    apply_heff2(AA, FL, W1, W2, FR)

Apply the two-site effective Hamiltonian without forming it explicitly.
`AA` and the returned tensor use `(left, right, physical_1, physical_2)`
index order.
"""
function apply_heff2(
    AA::AbstractArray{<:Number,4},
    FL::AbstractArray{<:Number,3},
    W1::AbstractArray{<:Number,4},
    W2::AbstractArray{<:Number,4},
    FR::AbstractArray{<:Number,3},
)
    @tensor HAA[alout, arout, s1out, s2out] := FL[alin, w0, alout] *
        AA[alin, arin, s1in, s2in] * W1[s1out, s1in, w0, w1] *
        W2[s2out, s2in, w1, w2] * FR[arin, w2, arout]
    return HAA
end

function _check_chain(psi::AbstractVector, H::AbstractVector)
    isempty(psi) && throw(ArgumentError("an MPS must contain at least one site"))
    length(psi) == length(H) || throw(DimensionMismatch(
        "MPS and MPO lengths differ: $(length(psi)) and $(length(H))",
    ))
    all(ndims(A) == 3 for A in psi) ||
        throw(ArgumentError("every MPS tensor must have three indices"))
    all(ndims(W) == 4 for W in H) ||
        throw(ArgumentError("every MPO tensor must have four indices"))
    return length(psi)
end

function _check_eigensolver_keywords(eig_tol, maxiter, krylovdim)
    eig_tol > 0 || throw(ArgumentError("eig_tol must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    krylovdim > 0 || throw(ArgumentError("krylovdim must be positive"))
    return nothing
end

"""Return the lowest Ritz pair and an independently checked residual."""
function _lowest_local_eigenpair(
    op,
    x0::AbstractArray;
    eig_tol::Real,
    maxiter::Integer,
    krylovdim::Integer,
)
    _check_eigensolver_keywords(eig_tol, maxiter, krylovdim)
    dims = size(x0)
    x = vec(copy(x0))
    xnorm = norm(x)
    iszero(xnorm) && throw(ArgumentError("the local eigensolver seed is zero"))
    x ./= xnorm

    vector_op = v -> vec(op(reshape(v, dims)))

    # KrylovKit needs a Krylov space larger than the requested eigenspace.
    # A one-dimensional effective Hilbert space is already its own solution.
    if length(x) == 1
        y = vector_op(x)
        lambda = dot(x, y)
        residual = norm(y - lambda * x)
        return real(lambda), copy(reshape(x, dims)), Float64(residual),
               residual <= eig_tol * max(1, abs(lambda))
    end

    kdim = min(length(x), max(2, Int(krylovdim)))
    values, vectors, _ = eigsolve(
        vector_op,
        x,
        1,
        :SR;
        ishermitian=true,
        tol=eig_tol,
        maxiter=maxiter,
        krylovdim=kdim,
    )
    isempty(values) && error("KrylovKit returned no Ritz value")

    lambda = values[1]
    v = vectors[1]
    residual = norm(vector_op(v) - lambda * v)
    threshold = eig_tol * max(1, abs(lambda))
    return real(lambda), copy(reshape(v, dims)), Float64(residual), residual <= threshold
end

function _solve_one_site(A, FL, W, FR; eig_tol, maxiter, krylovdim)
    return _lowest_local_eigenpair(
        _one_site_action(A,FL,W,FR),
        A;
        eig_tol=eig_tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
end

function _solve_two_site(AA, FL, W1, W2, FR; eig_tol, maxiter, krylovdim)
    return _lowest_local_eigenpair(
        _two_site_action(AA,FL,W1,W2,FR),
        AA;
        eig_tol=eig_tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
end

function _record_local!(local_energies, residuals, local_convergence, result)
    energy, tensor, residual, converged = result
    push!(local_energies, Float64(energy))
    push!(residuals, residual)
    push!(local_convergence, converged)
    return tensor
end

"""
    dmrg1_sweep!(psi, H; eig_tol=1e-10, maxiter=200, krylovdim=30)

Perform one complete one-site DMRG sweep, from the right boundary to the left
and back to the right.  `psi` is mutated.  The returned named tuple contains
the final expectation value and diagnostics for every local Krylov solve:
`(energy, local_energies, residuals, converged)`.
"""
function dmrg1_sweep!(
    psi::AbstractVector,
    H::AbstractVector;
    eig_tol::Real=1e-10,
    maxiter::Integer=200,
    krylovdim::Integer=30,
)
    N = _check_chain(psi, H)
    _check_eigensolver_keywords(eig_tol, maxiter, krylovdim)

    FL = collect(left_environments(psi, H))
    FR = collect(right_environments(psi, H))
    length(FL) == N + 1 || throw(DimensionMismatch("left_environments must return N + 1 tensors"))
    length(FR) == N + 1 || throw(DimensionMismatch("right_environments must return N + 1 tensors"))

    local_energies = Float64[]
    residuals = Float64[]
    local_convergence = Bool[]

    # MATLAB's first half-sweep starts at N.  Each QR moves the centre one
    # site left while FR is rebuilt from the already-optimised right block.
    for n in N:-1:2
        result = _solve_one_site(
            psi[n], FL[n], H[n], FR[n + 1];
            eig_tol=eig_tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        AC = _record_local!(local_energies, residuals, local_convergence, result)
        C, AR = right_orthogonalize(AC)
        psi[n] = AR
        psi[n - 1] = _absorb_right(psi[n - 1], C)
        FR[n] = right_environment(AR, H[n], FR[n + 1])
    end

    # Optimise the left boundary and turn the sweep around.
    result = _solve_one_site(
        psi[1], FL[1], H[1], FR[2];
        eig_tol=eig_tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    AC = _record_local!(local_energies, residuals, local_convergence, result)
    if N == 1
        psi[1] = AC
    else
        C, AL = left_orthogonalize(AC)
        psi[1] = AL
        psi[2] = _absorb_left(C, psi[2])
        FL[2] = left_environment(FL[1], AL, H[1])
    end

    # Return to the right boundary, rebuilding FL behind the centre.
    for n in 2:N
        result = _solve_one_site(
            psi[n], FL[n], H[n], FR[n + 1];
            eig_tol=eig_tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        AC = _record_local!(local_energies, residuals, local_convergence, result)
        if n < N
            C, AL = left_orthogonalize(AC)
            psi[n] = AL
            psi[n + 1] = _absorb_left(C, psi[n + 1])
            FL[n + 1] = left_environment(FL[n], AL, H[n])
        else
            psi[n] = AC
        end
    end

    normalize!(psi)
    energy = Float64(real(expectation(psi, H)))
    return (
        energy=energy,
        local_energies=local_energies,
        residuals=residuals,
        converged=all(local_convergence),
    )
end

"""
    dmrg2_sweep!(psi, H; eig_tol=1e-10, maxiter=200, krylovdim=30,
                 maxdim=typemax(Int), cutoff=0.0)

Perform one complete two-site DMRG sweep.  SVD truncation is delegated to
`split_right` while moving left and to `split_left` while moving right.  The
returned `discarded_weight` is the largest discarded weight in the sweep.
"""
function dmrg2_sweep!(
    psi::AbstractVector,
    H::AbstractVector;
    eig_tol::Real=1e-10,
    maxiter::Integer=200,
    krylovdim::Integer=30,
    maxdim::Integer=typemax(Int),
    cutoff::Real=0.0,
)
    N = _check_chain(psi, H)
    N >= 2 || throw(ArgumentError("two-site DMRG requires at least two sites"))
    _check_eigensolver_keywords(eig_tol, maxiter, krylovdim)
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative"))

    FL = collect(left_environments(psi, H))
    FR = collect(right_environments(psi, H))
    length(FL) == N + 1 || throw(DimensionMismatch("left_environments must return N + 1 tensors"))
    length(FR) == N + 1 || throw(DimensionMismatch("right_environments must return N + 1 tensors"))

    local_energies = Float64[]
    residuals = Float64[]
    local_convergence = Bool[]
    discarded_weight = 0.0

    # Move from the right boundary toward bond (1, 2).  Bond (1, 2) is
    # handled separately because its split reverses the sweep direction.
    for n in (N - 1):-1:2
        AA = contract_two_sites(psi[n], psi[n + 1])
        result = _solve_two_site(
            AA, FL[n], H[n], H[n + 1], FR[n + 2];
            eig_tol=eig_tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        AAC = _record_local!(local_energies, residuals, local_convergence, result)
        AC, AR, discarded = split_right(AAC; maxdim=maxdim, cutoff=cutoff)
        psi[n] = AC
        psi[n + 1] = AR
        discarded_weight = max(discarded_weight, Float64(real(discarded)))
        FR[n + 1] = right_environment(AR, H[n + 1], FR[n + 2])
    end

    # Left boundary and sweep reversal.
    AA = contract_two_sites(psi[1], psi[2])
    result = _solve_two_site(
        AA, FL[1], H[1], H[2], FR[3];
        eig_tol=eig_tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    AAC = _record_local!(local_energies, residuals, local_convergence, result)
    AL, AC, discarded = split_left(AAC; maxdim=maxdim, cutoff=cutoff)
    psi[1] = AL
    psi[2] = AC
    discarded_weight = max(discarded_weight, Float64(real(discarded)))
    FL[2] = left_environment(FL[1], AL, H[1])

    # Move back to the right boundary.  As in the MATLAB routine, bond 1
    # is not solved a second time during the same sweep.
    for n in 2:(N - 1)
        AA = contract_two_sites(psi[n], psi[n + 1])
        result = _solve_two_site(
            AA, FL[n], H[n], H[n + 1], FR[n + 2];
            eig_tol=eig_tol, maxiter=maxiter, krylovdim=krylovdim,
        )
        AAC = _record_local!(local_energies, residuals, local_convergence, result)
        AL, AC, discarded = split_left(AAC; maxdim=maxdim, cutoff=cutoff)
        psi[n] = AL
        psi[n + 1] = AC
        discarded_weight = max(discarded_weight, Float64(real(discarded)))
        FL[n + 1] = left_environment(FL[n], AL, H[n])
    end

    normalize!(psi)
    energy = Float64(real(expectation(psi, H)))
    return (
        energy=energy,
        local_energies=local_energies,
        residuals=residuals,
        discarded_weight=discarded_weight,
        converged=all(local_convergence),
    )
end

"""
    dmrg!(psi, H; algorithm=:two_site, nsweeps=10, energy_tol=1e-10,
          eig_tol=1e-10, maxiter=200, krylovdim=30,
          maxdim=typemax(Int), cutoff=0.0)

Run repeated one- or two-site DMRG sweeps.  `algorithm` is `:one_site` or
`:two_site`.  Convergence requires both a sufficiently small change in energy
and converged local Krylov solves.  The MPS is mutated and a driver diagnostic
`(energy, converged, sweeps, history)` is returned.
"""
function dmrg!(
    psi::AbstractVector,
    H::AbstractVector;
    algorithm::Symbol=:two_site,
    nsweeps::Integer=10,
    energy_tol::Real=1e-10,
    eig_tol::Real=1e-10,
    maxiter::Integer=200,
    krylovdim::Integer=30,
    maxdim::Integer=typemax(Int),
    cutoff::Real=0.0,
)
    _check_chain(psi, H)
    nsweeps > 0 || throw(ArgumentError("nsweeps must be positive"))
    energy_tol >= 0 || throw(ArgumentError("energy_tol must be nonnegative"))
    algorithm in (:one_site, :two_site) || throw(ArgumentError(
        "algorithm must be :one_site or :two_site",
    ))

    history = NamedTuple[]
    previous_energy = Inf
    driver_converged = false

    for sweep in 1:nsweeps
        diagnostics = if algorithm === :one_site
            dmrg1_sweep!(
                psi, H;
                eig_tol=eig_tol, maxiter=maxiter, krylovdim=krylovdim,
            )
        else
            dmrg2_sweep!(
                psi, H;
                eig_tol=eig_tol, maxiter=maxiter, krylovdim=krylovdim,
                maxdim=maxdim, cutoff=cutoff,
            )
        end

        energy_change = isfinite(previous_energy) ?
            abs(diagnostics.energy - previous_energy) : Inf
        entry = merge(diagnostics, (sweep=sweep, energy_change=energy_change))
        push!(history, entry)

        energy_scale = max(1.0, abs(diagnostics.energy))
        driver_converged = sweep > 1 && diagnostics.converged &&
                           energy_change <= energy_tol * energy_scale
        driver_converged && break
        previous_energy = diagnostics.energy
    end

    return (
        energy=history[end].energy,
        converged=driver_converged,
        sweeps=length(history),
        history=history,
    )
end
