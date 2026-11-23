# Shared dense-MPS contractions and QR/SVD factorizations.
#
# These routines are representation-level operations used by finite sweeps
# and by algorithms that manipulate uniform-MPS tensors.

function _absorb_left(C::AbstractMatrix, A::AbstractArray)
    _check_rank(A, 3, "A")
    size(C, 2) == size(A, 1) || throw(DimensionMismatch(
        "cannot absorb a $(size(C)) bond matrix into an MPS tensor $(size(A))"))
    @tensor B[a, r, s] := C[a, l] * A[l, r, s]
    return B
end

function _absorb_right(A::AbstractArray, C::AbstractMatrix)
    _check_rank(A, 3, "A")
    size(A, 2) == size(C, 1) || throw(DimensionMismatch(
        "cannot absorb a $(size(C)) bond matrix into an MPS tensor $(size(A))"))
    @tensor B[l, b, p] := A[l, r, p] * C[r, b]
    return B
end

"""
    contract_two_sites(A, B)

Contract the shared bond of adjacent MPS tensors, returning a tensor with
shape `(left(A), right(B), physical(A), physical(B))`.
"""
function contract_two_sites(A::AbstractArray, B::AbstractArray)
    _check_rank(A, 3, "A")
    _check_rank(B, 3, "B")
    size(A, 2) == size(B, 1) || throw(DimensionMismatch(
        "adjacent MPS bonds differ: $(size(A, 2)) and $(size(B, 1))"))
    @tensor AA[l, r, s, t] := A[l, m, s] * B[m, r, t]
    return AA
end

"""
    left_orthogonalize(A) -> C, AL

QR-factor a site tensor so that `AL` is left isometric and `A == AL * C`
in the MPS sense: `A[l,r,s] = sum_k AL[l,k,s] * C[k,r]`.
"""
function left_orthogonalize(A::AbstractArray)
    _check_rank(A, 3, "A")
    Dl, Dr, d = size(A)
    matrix = reshape(permutedims(A, (1, 3, 2)), Dl * d, Dr)
    factorization = qr(matrix)
    rank = min(size(matrix)...)
    Q = Matrix(factorization.Q)[:, 1:rank]
    C = Matrix(factorization.R)[1:rank, :]
    AL = permutedims(reshape(Q, Dl, d, rank), (1, 3, 2))
    return C, AL
end

"""
    right_orthogonalize(A) -> C, AR

QR-factor a site tensor from the right.  `AR` is right isometric and
`A[l,r,s] = sum_k C[l,k] * AR[k,r,s]`.
"""
function right_orthogonalize(A::AbstractArray)
    _check_rank(A, 3, "A")
    Dl, Dr, d = size(A)
    matrix = reshape(permutedims(A, (1, 3, 2)), Dl, d * Dr)
    factorization = qr(adjoint(matrix))
    rank = min(size(matrix)...)
    Q = Matrix(factorization.Q)[:, 1:rank]
    R = Matrix(factorization.R)[1:rank, :]
    C = Matrix(adjoint(R))
    AR = permutedims(reshape(Matrix(adjoint(Q)), rank, d, Dr), (1, 3, 2))
    return C, AR
end

"""
    split_left(AA; maxdim=typemax(Int), cutoff=0)
        -> AL, AC, discarded_weight

Split a two-site centre with an SVD, placing the singular values on the
second site.  `cutoff` is the maximum relative discarded squared singular
value weight, subject to the hard `maxdim` bound.
"""
function split_left(AA::AbstractArray;
                    maxdim::Integer=typemax(Int), cutoff::Real=0)
    _check_rank(AA, 4, "AA")
    Dl, Dr, d1, d2 = size(AA)
    matrix = reshape(permutedims(AA, (1, 3, 4, 2)), Dl * d1, d2 * Dr)
    factorization = svd(matrix; full=false)
    rank, discarded_weight = _svd_rank(factorization.S, maxdim, cutoff)

    U = view(factorization.U, :, 1:rank)
    singular_values = view(factorization.S, 1:rank)
    Vt = view(factorization.Vt, 1:rank, :)

    AL = permutedims(reshape(Array(U), Dl, d1, rank), (1, 3, 2))
    centre = Diagonal(singular_values) * Vt
    AC = permutedims(reshape(Array(centre), rank, d2, Dr), (1, 3, 2))
    return AL, AC, discarded_weight
end

"""
    split_right(AA; maxdim=typemax(Int), cutoff=0)
        -> AC, AR, discarded_weight

Split a two-site centre with an SVD, placing the singular values on the
first site.  The reported discarded weight is relative squared Frobenius
weight.
"""
function split_right(AA::AbstractArray;
                     maxdim::Integer=typemax(Int), cutoff::Real=0)
    _check_rank(AA, 4, "AA")
    Dl, Dr, d1, d2 = size(AA)
    matrix = reshape(permutedims(AA, (1, 3, 4, 2)), Dl * d1, d2 * Dr)
    factorization = svd(matrix; full=false)
    rank, discarded_weight = _svd_rank(factorization.S, maxdim, cutoff)

    U = view(factorization.U, :, 1:rank)
    singular_values = view(factorization.S, 1:rank)
    Vt = view(factorization.Vt, 1:rank, :)

    centre = U * Diagonal(singular_values)
    AC = permutedims(reshape(Array(centre), Dl, d1, rank), (1, 3, 2))
    AR = permutedims(reshape(Array(Vt), rank, d2, Dr), (1, 3, 2))
    return AC, AR, discarded_weight
end
