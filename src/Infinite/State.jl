# One-site-unit-cell infinite-MPS state and canonicalization primitives.

function _check_uniform_tensor(A, name="A")
    _check_rank(A, 3, name)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch(
        "$name must have equal left and right bond dimensions; got $(size(A))",
    ))
    return size(A, 1), size(A, 3)
end

function _check_two_site_hamiltonian(h, d)
    _check_rank(h, 4, "h")
    size(h) == (d, d, d, d) || throw(DimensionMismatch(
        "h must have size ($d, $d, $d, $d); got $(size(h))",
    ))
    return nothing
end

function _check_mixed_canonical(AL, AR, C; h=nothing)
    chi, d = _check_uniform_tensor(AL, "AL")
    _check_uniform_tensor(AR, "AR") == (chi, d) || throw(DimensionMismatch(
        "AL and AR sizes differ: $(size(AL)) and $(size(AR))",
    ))
    _check_rank(C, 2, "C")
    size(C) == (chi, chi) || throw(DimensionMismatch(
        "C must have size ($chi, $chi); got $(size(C))",
    ))
    h === nothing || _check_two_site_hamiltonian(h, d)
    return chi, d
end

"""
    transfer_right(A, X)

Apply the right transfer map `X ↦ Σₛ Aˢ X Aˢ†` for a uniform MPS tensor
`A[left,right,physical]`.
"""
function transfer_right(
    A::AbstractArray{<:Number,3},
    X::AbstractMatrix{<:Number},
)
    chi, _ = _check_uniform_tensor(A)
    size(X) == (chi, chi) || throw(DimensionMismatch(
        "X must have size ($chi, $chi); got $(size(X))",
    ))
    @tensor Y[l, lb] := A[l, r, s] * X[r, rb] * conj(A[lb, rb, s])
    return Y
end

"""Apply the left transfer map `X ↦ Σₛ Aˢ† X Aˢ`."""
function transfer_left(
    A::AbstractArray{<:Number,3},
    X::AbstractMatrix{<:Number},
)
    chi, _ = _check_uniform_tensor(A)
    size(X) == (chi, chi) || throw(DimensionMismatch(
        "X must have size ($chi, $chi); got $(size(X))",
    ))
    @tensor Y[r, rb] := conj(A[l, r, s]) * X[l, lb] * A[lb, rb, s]
    return Y
end

function _positive_density(X::AbstractMatrix)
    hermitian_part = Hermitian((X + adjoint(X)) / 2)
    decomposition = eigen(hermitian_part)
    scale = maximum(abs, decomposition.values; init=0.0)
    floor_value = max(eps(Float64) * max(1.0, Float64(scale)), 0.0)
    values = max.(real.(decomposition.values), floor_value)
    density = decomposition.vectors * Diagonal(values) * adjoint(decomposition.vectors)
    trace_value = real(tr(density))
    trace_value > 0 || return Matrix{eltype(X)}(I, size(X, 1), size(X, 2)) /
                            size(X, 1)
    return Matrix(density / trace_value)
end

function _dominant_fixedpoint(
    A,
    direction::Symbol;
    x0=nothing,
    tol::Real=1e-12,
    maxiter::Integer=500,
    krylovdim::Integer=30,
)
    chi, _ = _check_uniform_tensor(A)
    direction in (:left, :right) || throw(ArgumentError(
        "direction must be :left or :right",
    ))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    krylovdim > 0 || throw(ArgumentError("krylovdim must be positive"))

    T = eltype(A)
    seed = x0 === nothing ? Matrix{T}(I, chi, chi) / chi : Matrix{T}(x0)
    size(seed) == (chi, chi) || throw(DimensionMismatch(
        "fixed-point seed must have size ($chi, $chi)",
    ))
    map = direction === :right ? X -> transfer_right(A, X) :
                                 X -> transfer_left(A, X)

    if chi == 1
        fixed = ones(T, 1, 1)
        image = map(fixed)
        lambda = image[1]
        residual = norm(image - lambda .* fixed)
        return fixed, lambda, Float64(residual), true
    end

    dims = (chi, chi)
    vector_map = v -> vec(map(reshape(v, dims)))
    kdim = min(chi^2, max(2, Int(krylovdim)))
    values, vectors, info = eigsolve(
        vector_map,
        vec(seed),
        1,
        :LM;
        tol=tol,
        maxiter=maxiter,
        krylovdim=kdim,
        ishermitian=false,
    )
    isempty(values) && error("KrylovKit returned no transfer eigenvalue")
    raw = reshape(vectors[1], dims)
    phase_reference = tr(raw)
    if abs(phase_reference) > eps(Float64)
        raw ./= phase_reference / abs(phase_reference)
    end
    fixed = _positive_density(raw)
    image = map(fixed)
    lambda = dot(vec(fixed), vec(image)) / dot(vec(fixed), vec(fixed))
    residual = norm(image - lambda .* fixed) / max(norm(image), eps(Float64))
    converged = info.converged >= 1 && residual <= 10 * sqrt(tol)
    return fixed, lambda, Float64(residual), converged
end

"""
    transfer_fixed_points(AL, AR; kwargs...)

Find the right density matrix of the left-canonical tensor `AL` and the left
density matrix of the right-canonical tensor `AR`.
"""
function transfer_fixed_points(
    AL,
    AR;
    L0=nothing,
    R0=nothing,
    tol::Real=1e-12,
    maxiter::Integer=500,
    krylovdim::Integer=30,
)
    chi, d = _check_uniform_tensor(AL, "AL")
    _check_uniform_tensor(AR, "AR") == (chi, d) || throw(DimensionMismatch(
        "AL and AR must have the same size",
    ))
    R, lambda_R, residual_R, converged_R = _dominant_fixedpoint(
        AL, :right; x0=R0, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    L, lambda_L, residual_L, converged_L = _dominant_fixedpoint(
        AR, :left; x0=L0, tol=tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    return (
        L=L,
        R=R,
        lambda_L=lambda_L,
        lambda_R=lambda_R,
        residual_L=residual_L,
        residual_R=residual_R,
        converged=converged_L && converged_R,
    )
end

function _positive_sqrt_and_inverse(X::AbstractMatrix)
    decomposition = eigen(Hermitian((X + adjoint(X)) / 2))
    largest = maximum(abs, decomposition.values; init=1.0)
    floor_value = max(eps(Float64) * max(1.0, Float64(largest)), 1e-14)
    values = max.(real.(decomposition.values), floor_value)
    root = decomposition.vectors * Diagonal(sqrt.(values)) *
           adjoint(decomposition.vectors)
    inverse_root = decomposition.vectors * Diagonal(inv.(sqrt.(values))) *
                   adjoint(decomposition.vectors)
    return Matrix(root), Matrix(inverse_root)
end

"""
    canonicalize_uniform(A; kwargs...)

Convert an injective one-site uniform tensor into mutually consistent left-
and right-canonical tensors and a centre matrix.  The construction uses the
positive transfer fixed points and satisfies `AL*C ≈ C*AR`.
"""
function canonicalize_uniform(
    A::AbstractArray{<:Number,3};
    tol::Real=1e-12,
    maxiter::Integer=500,
    krylovdim::Integer=30,
)
    chi, d = _check_uniform_tensor(A)
    right, lambda_R, residual_R, converged_R = _dominant_fixedpoint(
        A, :right; tol=tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    left, lambda_L, residual_L, converged_L = _dominant_fixedpoint(
        A, :left; tol=tol, maxiter=maxiter, krylovdim=krylovdim,
    )
    lambda = sqrt(abs(lambda_L * lambda_R))
    lambda > eps(Float64) || throw(ArgumentError(
        "uniform tensor has a vanishing dominant transfer eigenvalue",
    ))
    X, Xinv = _positive_sqrt_and_inverse(right)
    Y, Yinv = _positive_sqrt_and_inverse(left)

    T = promote_type(eltype(A), eltype(X), eltype(Y))
    AL = Array{T,3}(undef, chi, chi, d)
    AR = Array{T,3}(undef, chi, chi, d)
    normalization = sqrt(lambda)
    for s in 1:d
        As = Matrix(view(A, :, :, s))
        AL[:, :, s] .= (Y * As * Yinv) / normalization
        AR[:, :, s] .= (Xinv * As * X) / normalization
    end
    C = Y * X
    C ./= norm(C)
    residuals = infinite_residuals(AL, AR, C)
    return (
        AL=AL,
        AR=AR,
        C=Matrix(C),
        residuals=residuals,
        transfer_residual=max(residual_L, residual_R),
        converged=converged_L && converged_R,
    )
end

"""Construct a random, consistent mixed-canonical one-site uniform MPS."""
function random_infinite_mps(
    chi::Integer,
    d::Integer;
    rng::AbstractRNG=Random.default_rng(),
    T::Type{<:Number}=ComplexF64,
    tol::Real=1e-12,
)
    chi >= 1 || throw(ArgumentError("chi must be positive"))
    d >= 1 || throw(ArgumentError("d must be positive"))
    _is_floating_scalar_type(T) || throw(ArgumentError(
        "T must be a real or complex floating-point type",
    ))
    A = _random_array(rng, T, chi, chi, d)
    return canonicalize_uniform(A; tol=tol)
end

"""Return the one-site centre tensor `AC = AL*C`."""
function center_tensor(AL, C)
    chi, _ = _check_uniform_tensor(AL, "AL")
    size(C) == (chi, chi) || throw(DimensionMismatch(
        "C and AL bond dimensions differ",
    ))
    @tensor AC[l, r, s] := AL[l, m, s] * C[m, r]
    return AC
end

function _right_center_tensor(C, AR)
    chi, _ = _check_uniform_tensor(AR, "AR")
    size(C) == (chi, chi) || throw(DimensionMismatch(
        "C and AR bond dimensions differ",
    ))
    @tensor AC[l, r, s] := C[l, m] * AR[m, r, s]
    return AC
end

"""Return the mixed-canonical two-site centre `AL*C*AR`."""
function two_site_center(AL, C, AR)
    _check_mixed_canonical(AL, AR, C)
    @tensor theta[l, r, s, t] := AL[l, a, s] * C[a, b] * AR[b, r, t]
    return theta
end

function _polar_isometry(M::AbstractMatrix)
    decomposition = svd(M; full=false)
    return Matrix(decomposition.U * decomposition.Vt)
end

"""
    mixed_canonical_update(AC, C)

Recover left- and right-canonical tensors from independently updated `AC` and
`C` using the polar-decomposition gauge matching used by VUMPS.  Returns a
named tuple containing `AL`, `AR`, normalized `C`, and consistency residuals.
"""
function mixed_canonical_update(AC, C)
    chi, d = _check_uniform_tensor(AC, "AC")
    size(C) == (chi, chi) || throw(DimensionMismatch(
        "C and AC bond dimensions differ",
    ))
    Cnew = Matrix(C)
    cnorm = norm(Cnew)
    cnorm > 0 || throw(ArgumentError("C must have nonzero norm"))
    Cnew ./= cnorm

    left_AC = reshape(permutedims(AC, (3, 1, 2)), d * chi, chi)
    right_AC = reshape(permutedims(AC, (1, 3, 2)), chi, d * chi)
    Q_AC_left = _polar_isometry(left_AC)
    Q_AC_right = _polar_isometry(right_AC)
    Q_C = _polar_isometry(Cnew)

    left_matrix = Q_AC_left * adjoint(Q_C)
    right_matrix = adjoint(Q_C) * Q_AC_right
    AL = permutedims(reshape(left_matrix, d, chi, chi), (2, 3, 1))
    AR = permutedims(reshape(right_matrix, chi, d, chi), (1, 3, 2))
    residuals = infinite_residuals(AL, AR, Cnew; AC=AC)
    return (
        AL=Array(AL),
        AR=Array(AR),
        C=Cnew,
        residual=max(residuals.center_left, residuals.center_right),
        residuals=residuals,
    )
end

"""Return canonical and centre-consistency residuals for a mixed uMPS."""
function infinite_residuals(AL, AR, C; AC=center_tensor(AL, C))
    chi, d = _check_mixed_canonical(AL, AR, C)
    size(AC) == (chi, chi, d) || throw(DimensionMismatch(
        "AC must have size ($chi, $chi, $d)",
    ))
    left_gram = zeros(promote_type(eltype(AL), eltype(AR)), chi, chi)
    right_gram = zeros(eltype(left_gram), chi, chi)
    for s in 1:d
        ALs = Matrix(view(AL, :, :, s))
        ARs = Matrix(view(AR, :, :, s))
        left_gram .+= adjoint(ALs) * ALs
        right_gram .+= ARs * adjoint(ARs)
    end
    identity_matrix = Matrix{eltype(left_gram)}(I, chi, chi)
    left_centre = center_tensor(AL, C)
    right_centre = _right_center_tensor(C, AR)
    scale = max(norm(AC), eps(Float64))
    return (
        left_canonical=Float64(norm(left_gram - identity_matrix)),
        right_canonical=Float64(norm(right_gram - identity_matrix)),
        center_left=Float64(norm(AC - left_centre) / scale),
        center_right=Float64(norm(AC - right_centre) / scale),
        center_consistency=Float64(norm(left_centre - right_centre) /
                                   max(norm(left_centre), eps(Float64))),
    )
end

"""Evaluate the nearest-neighbour energy density of a mixed-canonical uMPS."""
function uniform_energy(AL, AR, C, h)
    _, d = _check_mixed_canonical(AL, AR, C; h=h)
    theta = two_site_center(AL, C, AR)
    @tensor Htheta[l, r, so, to] := h[so, to, si, ti] * theta[l, r, si, ti]
    denominator = dot(vec(theta), vec(theta))
    abs(denominator) > eps(Float64) || throw(ArgumentError(
        "the two-site centre has zero norm",
    ))
    return dot(vec(theta), vec(Htheta)) / denominator
end

"""
    ising_two_site(; J=-1, hx=0, hy=0, hz=0, shift=0,
                   interaction=:z)

Construct the translation-invariant two-site Hamiltonian density

`J S⊗S + 1/2 (q⊗I + I⊗q)`,

where `q = hx*X + hy*Y + hz*Z - shift*I`.  Summing this density over bonds
counts each onsite term once.
"""
function ising_two_site(;
    J::Number=-1,
    hx::Number=0,
    hy::Number=0,
    hz::Number=0,
    shift::Number=0,
    interaction::Symbol=:z,
)
    interaction in (:x, :z) || throw(ArgumentError(
        "interaction must be :x or :z",
    ))
    scalar_type = promote_type(Float64, typeof(J), typeof(hx), typeof(hy),
                               typeof(hz), typeof(shift))
    T = scalar_type <: Real && iszero(hy) ? scalar_type :
        promote_type(scalar_type, ComplexF64)
    I2 = Matrix{T}(I, 2, 2)
    X = T[0 1; 1 0]
    Y = zeros(T, 2, 2)
    if !iszero(hy)
        Y[1, 2] = convert(T, -im)
        Y[2, 1] = convert(T, im)
    end
    Z = T[1 0; 0 -1]
    S = interaction === :x ? X : Z
    onsite = hx .* X .+ hy .* Y .+ hz .* Z .- shift .* I2
    matrix = J .* kron(S, S) .+
             0.5 .* (kron(onsite, I2) .+ kron(I2, onsite))
    # `kron` uses site 2 as the fast composite index; Julia's first reshape
    # index is fast, hence the temporary (site2,site1,site2,site1) order.
    return permutedims(reshape(matrix, 2, 2, 2, 2), (2, 1, 4, 3))
end

"""Convert `h[out1,out2,in1,in2]` to a dense two-site matrix."""
function two_site_matrix(h::AbstractArray{<:Number,4})
    d = size(h, 1)
    _check_two_site_hamiltonian(h, d)
    return reshape(permutedims(h, (2, 1, 4, 3)), d^2, d^2)
end
