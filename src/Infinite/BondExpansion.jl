# Residual-based bond-dimension expansion for one-site uniform MPS.

"""
    left_tangent_basis(AL)

Construct an orthonormal basis `VL[left, tangent, physical]` for the
left-gauge tangent complement of a left-canonical uniform MPS tensor.  It
satisfies

`sum(conj(AL[l,r,s]) * VL[l,a,s], l, s) == 0`

and has `tangent = (d - 1) * chi` columns.
"""
function left_tangent_basis(AL::AbstractArray{<:Number,3})
    chi, d = _check_uniform_tensor(AL, "AL")
    tangent_dimension = (d - 1) * chi
    tangent_dimension == 0 && return zeros(eltype(AL), chi, 0, d)

    # Rows use the composite index (left, physical), with `left` fast.
    left_matrix = reshape(permutedims(AL, (1, 3, 2)), d * chi, chi)
    decomposition = svd(left_matrix; full=true)
    complement = decomposition.U[:, (chi + 1):(d * chi)]
    return Array(permutedims(reshape(complement, chi, d, tangent_dimension),
                             (1, 3, 2)))
end

"""
    right_tangent_basis(AR)

Construct an orthonormal basis `VR[tangent, right, physical]` for the
right-gauge tangent complement of a right-canonical uniform MPS tensor.  The
complex conjugation in this convention matches the original MATLAB
`rightGaugeTangentBasis.m` and the contraction used by
[`bond_expansion_coefficients`](@ref).
"""
function right_tangent_basis(AR::AbstractArray{<:Number,3})
    chi, d = _check_uniform_tensor(AR, "AR")
    tangent_dimension = (d - 1) * chi
    tangent_dimension == 0 && return zeros(eltype(AR), 0, chi, d)

    # Columns use the composite index (physical, right), with `physical`
    # fast, exactly as in the MATLAB reshape.
    right_matrix = reshape(permutedims(AR, (1, 3, 2)), chi, d * chi)
    decomposition = svd(right_matrix; full=true)
    complement = adjoint(decomposition.Vt)[:, (chi + 1):(d * chi)]
    return Array(permutedims(conj.(reshape(complement, d, chi,
                                           tangent_dimension)), (3, 2, 1)))
end

function _check_expansion_isometries(AL, AR)
    chi, d = _check_uniform_tensor(AL, "AL")
    _check_uniform_tensor(AR, "AR") == (chi, d) || throw(DimensionMismatch(
        "AL and AR must have the same size",
    ))
    T = promote_type(eltype(AL), eltype(AR))
    left_gram = zeros(T, chi, chi)
    right_gram = zeros(T, chi, chi)
    for s in 1:d
        ALs = Matrix(view(AL, :, :, s))
        ARs = Matrix(view(AR, :, :, s))
        left_gram .+= adjoint(ALs) * ALs
        right_gram .+= ARs * adjoint(ARs)
    end
    identity_matrix = Matrix{T}(I, chi, chi)
    real_type = typeof(real(zero(eltype(AL))))
    machine_epsilon = real_type <: AbstractFloat ? eps(real_type) : eps(Float64)
    tolerance = 100 * machine_epsilon^(2 / 3) * max(1.0, sqrt(Float64(chi)))
    left_error = norm(left_gram - identity_matrix)
    right_error = norm(right_gram - identity_matrix)
    max(left_error, right_error) <= tolerance || throw(ArgumentError(
        "bond expansion requires left/right canonical tensors; " *
        "isometry errors are ($left_error, $right_error)",
    ))
    return nothing
end

function _check_tangent_bases(VL, VR, chi, d)
    tangent_dimension = (d - 1) * chi
    size(VL) == (chi, tangent_dimension, d) || throw(DimensionMismatch(
        "VL must have size ($chi, $tangent_dimension, $d); got $(size(VL))",
    ))
    size(VR) == (tangent_dimension, chi, d) || throw(DimensionMismatch(
        "VR must have size ($tangent_dimension, $chi, $d); got $(size(VR))",
    ))
    return tangent_dimension
end

function _project_two_site_action(VL, VR, action_theta)
    @tensor Y[a, b] := conj(VL[l, a, so]) *
                       action_theta[l, r, so, to] * conj(VR[b, r, to])
    return Y
end

function _apply_two_site_hamiltonian(theta, h)
    @tensor action_theta[l, r, so, to] := h[so, to, si, ti] *
                                              theta[l, r, si, ti]
    return action_theta
end

"""
    bond_expansion_coefficients(AL, AR, C, h; VL, VR)
    bond_expansion_coefficients(AL, AR, C, two_site_action; VL, VR)

Project a two-site action on `AL*C*AR` into both tangent complements.  The
resulting matrix is the `Y` matrix in `VUMPS/vumps.m::calculateY`; its singular
vectors select new virtual directions and its singular values measure their
importance.  The callable form accepts `two_site_action(theta)` and is shared
by the MPO and long-range extensions.
"""
function bond_expansion_coefficients(
    AL,
    AR,
    C,
    h;
    VL=left_tangent_basis(AL),
    VR=right_tangent_basis(AR),
)
    chi, d = _check_mixed_canonical(AL, AR, C; h=h)
    _check_expansion_isometries(AL, AR)
    tangent_dimension = _check_tangent_bases(VL, VR, chi, d)
    tangent_dimension == 0 && return zeros(
        promote_type(eltype(AL), eltype(AR), eltype(C), eltype(h)), 0, 0,
    )

    theta = two_site_center(AL, C, AR)
    return _project_two_site_action(
        VL, VR, _apply_two_site_hamiltonian(theta, h),
    )
end


function bond_expansion_coefficients(
    AL,
    AR,
    C,
    two_site_action::Function;
    VL=left_tangent_basis(AL),
    VR=right_tangent_basis(AR),
)
    chi, d = _check_mixed_canonical(AL, AR, C)
    _check_expansion_isometries(AL, AR)
    tangent_dimension = _check_tangent_bases(VL, VR, chi, d)
    tangent_dimension == 0 && return zeros(
        promote_type(eltype(AL), eltype(AR), eltype(C)), 0, 0,
    )
    theta = two_site_center(AL, C, AR)
    action_theta = two_site_action(theta)
    size(action_theta) == size(theta) || throw(DimensionMismatch(
        "two_site_action returned size $(size(action_theta)); expected $(size(theta))",
    ))
    return _project_two_site_action(VL, VR, action_theta)
end

function _no_bond_expansion(AL, AR, C, chi; singular_values=Float64[],
                            residual_norm=0.0, threshold=0.0,
                            numerical_rank=0)
    return (
        AL=copy(AL), AR=copy(AR), C=copy(C),
        old_bond=chi, new_bond=chi, added_bond=0, expanded=false,
        singular_values=singular_values,
        retained_singular_values=Float64[],
        numerical_rank=numerical_rank,
        residual_norm=Float64(residual_norm),
        discarded_projected_weight=sum(abs2, singular_values),
        threshold=Float64(threshold),
    )
end

function _expand_from_projected_action(
    AL,
    AR,
    C,
    VL,
    VR,
    Y;
    max_bond::Integer,
    max_add::Integer,
    atol::Real,
    rtol::Real,
    action_scale::Real,
)
    chi, d = _check_mixed_canonical(AL, AR, C)
    available = min(max_bond - chi, max_add, (d - 1) * chi)
    available == 0 && return _no_bond_expansion(
        AL, AR, C, chi; threshold=atol,
    )

    decomposition = svd(Y; full=false)
    singular_values = Float64.(real.(decomposition.S))
    largest = maximum(singular_values; init=0.0)
    real_type = typeof(real(zero(eltype(AL))))
    machine_epsilon = real_type <: AbstractFloat ? eps(real_type) : eps(Float64)
    numerical_floor = 10 * machine_epsilon * max(1.0, Float64(action_scale)) *
                      max(1.0, sqrt(Float64(length(Y))))
    threshold = max(Float64(atol), Float64(rtol) * largest, numerical_floor)
    numerical_rank = count(value -> value > threshold, singular_values)
    added = min(available, numerical_rank)

    added == 0 && return _no_bond_expansion(
        AL, AR, C, chi;
        singular_values=singular_values,
        residual_norm=norm(Y),
        threshold=threshold,
        numerical_rank=numerical_rank,
    )

    U = decomposition.U[:, 1:added]
    Vt = decomposition.Vt[1:added, :]
    @tensor BL[l, a, s] := VL[l, q, s] * U[q, a]
    @tensor BR[a, r, s] := Vt[a, q] * VR[q, r, s]

    new_bond = chi + added
    T = promote_type(eltype(AL), eltype(AR), eltype(C), eltype(Y))
    ALnew = zeros(T, new_bond, new_bond, d)
    ARnew = zeros(T, new_bond, new_bond, d)
    Cnew = zeros(T, new_bond, new_bond)
    ALnew[1:chi, 1:chi, :] .= AL
    ALnew[1:chi, (chi + 1):new_bond, :] .= BL
    ARnew[1:chi, 1:chi, :] .= AR
    ARnew[(chi + 1):new_bond, 1:chi, :] .= BR
    Cnew[1:chi, 1:chi] .= C
    discarded_weight = added < length(singular_values) ?
        sum(abs2, @view(singular_values[(added + 1):end])) : 0.0

    return (
        AL=ALnew,
        AR=ARnew,
        C=Cnew,
        old_bond=chi,
        new_bond=new_bond,
        added_bond=added,
        expanded=true,
        singular_values=singular_values,
        retained_singular_values=singular_values[1:added],
        numerical_rank=numerical_rank,
        residual_norm=Float64(norm(Y)),
        discarded_projected_weight=Float64(discarded_weight),
        threshold=threshold,
    )
end

"""
    expand_bond_dimension(AL, AR, C, h; max_bond, max_add, atol, rtol)

Increase a mixed-canonical uniform MPS bond dimension using the projected
two-site Hamiltonian residual from the original MATLAB
`increase_bond_dim`.  Singular directions of the residual are retained up to
`max_bond` and `max_add`.  Values not exceeding
`max(atol, rtol * maximum(singular_values))` are discarded.

The returned `AL`, `AR`, and `C` embed the input state *exactly*: the new block
of `C` is zero.  `AL` and `AR` remain left- and right-canonical, respectively.
The next VUMPS or TDVP local solve can then populate the added sector.
"""
function expand_bond_dimension(
    AL,
    AR,
    C,
    h;
    max_bond::Integer,
    max_add::Integer=typemax(Int),
    atol::Real=1e-14,
    rtol::Real=1e-12,
)
    chi, d = _check_mixed_canonical(AL, AR, C; h=h)
    max_bond >= chi || throw(ArgumentError(
        "max_bond ($max_bond) cannot be smaller than the current bond dimension ($chi)",
    ))
    max_add >= 0 || throw(ArgumentError("max_add must be nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))

    available = min(max_bond - chi, max_add, (d - 1) * chi)
    available == 0 && return _no_bond_expansion(
        AL, AR, C, chi; threshold=atol,
    )
    _check_expansion_isometries(AL, AR)

    VL = left_tangent_basis(AL)
    VR = right_tangent_basis(AR)
    theta = two_site_center(AL, C, AR)
    action_theta = _apply_two_site_hamiltonian(theta, h)
    Y = _project_two_site_action(VL, VR, action_theta)
    return _expand_from_projected_action(
        AL, AR, C, VL, VR, Y;
        max_bond=max_bond, max_add=max_add, atol=atol, rtol=rtol,
        action_scale=norm(action_theta),
    )
end

"""
    expand_bond_dimension(AL, AR, C, two_site_action; kwargs...)

Model-independent form of [`expand_bond_dimension`](@ref).  The callable must
map a two-site centre tensor to a tensor with the same shape.  This is the
shared padding/rank-selection core used by infinite MPO and long-range
Hamiltonians.
"""
function expand_bond_dimension(
    AL,
    AR,
    C,
    two_site_action::Function;
    max_bond::Integer,
    max_add::Integer=typemax(Int),
    atol::Real=1e-14,
    rtol::Real=1e-12,
)
    chi, d = _check_mixed_canonical(AL, AR, C)
    max_bond >= chi || throw(ArgumentError(
        "max_bond ($max_bond) cannot be smaller than the current bond dimension ($chi)",
    ))
    max_add >= 0 || throw(ArgumentError("max_add must be nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    available = min(max_bond - chi, max_add, (d - 1) * chi)
    available == 0 && return _no_bond_expansion(
        AL, AR, C, chi; threshold=atol,
    )
    _check_expansion_isometries(AL, AR)

    VL = left_tangent_basis(AL)
    VR = right_tangent_basis(AR)
    theta = two_site_center(AL, C, AR)
    action_theta = two_site_action(theta)
    size(action_theta) == size(theta) || throw(DimensionMismatch(
        "two_site_action returned size $(size(action_theta)); expected $(size(theta))",
    ))
    Y = _project_two_site_action(VL, VR, action_theta)
    return _expand_from_projected_action(
        AL, AR, C, VL, VR, Y;
        max_bond=max_bond, max_add=max_add, atol=atol, rtol=rtol,
        action_scale=norm(action_theta),
    )
end
