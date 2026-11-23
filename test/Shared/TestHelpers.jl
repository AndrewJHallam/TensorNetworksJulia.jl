# Shared imports and independent dense-reference helpers.

using Test
using LinearAlgebra
using Random
using TensorNetworksJulia

const TEST_ATOL = 1.0e-11

"""Expand an open-boundary MPS without using any package contractions."""
function dense_state(psi)
    N = length(psi)
    @assert N > 0
    d = size(psi[1], 3)
    @assert size(psi[1], 1) == 1
    @assert size(psi[end], 2) == 1
    @assert all(size(A, 3) == d for A in psi)

    state = zeros(ComplexF64, d^N)
    physical = zeros(Int, N)
    for basis_index in 0:(d^N - 1)
        remainder = basis_index
        for site in N:-1:1
            physical[site] = mod(remainder, d) + 1
            remainder = div(remainder, d)
        end

        bond = ComplexF64[1]
        for site in 1:N
            A = psi[site]
            @assert length(bond) == size(A, 1)
            next_bond = zeros(ComplexF64, size(A, 2))
            for left in axes(A, 1), right in axes(A, 2)
                next_bond[right] +=
                    bond[left] * A[left, right, physical[site]]
            end
            bond = next_bond
        end
        state[basis_index + 1] = bond[1]
    end
    return state
end

function left_canonical_gram(A)
    gram = zeros(ComplexF64, size(A, 2), size(A, 2))
    for physical in axes(A, 3)
        As = A[:, :, physical]
        gram .+= As' * As
    end
    return gram
end

function right_canonical_gram(A)
    gram = zeros(ComplexF64, size(A, 1), size(A, 1))
    for physical in axes(A, 3)
        As = A[:, :, physical]
        gram .+= As * As'
    end
    return gram
end

function absorb_right(AL, C)
    A = zeros(
        promote_type(eltype(AL), eltype(C)),
        size(AL, 1),
        size(C, 2),
        size(AL, 3),
    )
    for physical in axes(AL, 3)
        A[:, :, physical] .= AL[:, :, physical] * C
    end
    return A
end

function absorb_left(C, AR)
    A = zeros(
        promote_type(eltype(C), eltype(AR)),
        size(C, 1),
        size(AR, 2),
        size(AR, 3),
    )
    for physical in axes(AR, 3)
        A[:, :, physical] .= C * AR[:, :, physical]
    end
    return A
end

function embedded_operator(operators)
    return reduce(kron, operators)
end

function exact_open_ising(
    N;
    J,
    hx,
    hy,
    hz,
    shift = 0,
    interaction = :z,
)
    identity2 = ComplexF64[1 0; 0 1]
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    interaction_operator = if interaction === :x
        X
    elseif interaction === :z
        Z
    else
        error("unsupported Ising interaction $(interaction)")
    end

    H = zeros(ComplexF64, 2^N, 2^N)
    for site in 1:N
        for (coefficient, operator) in ((hx, X), (hy, Y), (hz, Z))
            operators = [copy(identity2) for _ in 1:N]
            operators[site] = operator
            H .+= coefficient .* embedded_operator(operators)
        end
        operators = [copy(identity2) for _ in 1:N]
        H .-= shift .* embedded_operator(operators)
    end
    for site in 1:(N - 1)
        operators = [copy(identity2) for _ in 1:N]
        operators[site] = interaction_operator
        operators[site + 1] = interaction_operator
        H .+= J .* embedded_operator(operators)
    end
    return H
end

function nonincreasing(values; atol = 1.0e-8)
    return all(values[index + 1] <= values[index] + atol for
               index in 1:(length(values) - 1))
end
