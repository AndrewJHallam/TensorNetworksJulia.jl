# Open-boundary finite model constructors and their TEBD bond densities.
#
# Every tensor follows the package convention
# `(physical_output, physical_input, left_mpo_bond, right_mpo_bond)`.

function _finite_mpo_coefficient_vector(value, count::Integer, name::AbstractString)
    count >= 0 || throw(ArgumentError("coefficient count must be nonnegative"))
    if value isa Number
        return fill(value, count)
    end
    value isa AbstractVector || throw(ArgumentError(
        "$name must be a scalar or a vector of length $count",
    ))
    length(value) == count || throw(DimensionMismatch(
        "$name has length $(length(value)); expected $count",
    ))
    all(entry -> entry isa Number, value) || throw(ArgumentError(
        "$name must contain numbers",
    ))
    return collect(value)
end

function _finite_mpo_scalar_type(collections...)
    types = Type[Float64]
    for collection in collections
        if collection isa Number
            push!(types, typeof(collection))
        elseif collection isa AbstractArray || collection isa Tuple
            for value in collection
                value isa Number || throw(ArgumentError(
                    "model coefficients must contain numbers",
                ))
                push!(types, typeof(value))
            end
        else
            throw(ArgumentError("model coefficients must be numeric"))
        end
    end
    return promote_type(types...)
end

function _finite_mpo_onsite_terms(onsite, N::Integer)
    terms = if onsite isa AbstractMatrix
        [onsite for _ in 1:N]
    elseif onsite isa AbstractVector
        length(onsite) == N || throw(DimensionMismatch(
            "onsite has length $(length(onsite)); expected $N",
        ))
        collect(onsite)
    else
        throw(ArgumentError(
            "onsite must be a square matrix or a vector of $N square matrices",
        ))
    end

    for (site, term) in pairs(terms)
        term isa AbstractMatrix || throw(ArgumentError(
            "onsite[$site] must be a matrix",
        ))
        eltype(term) <: Number || throw(ArgumentError(
            "onsite[$site] must contain numbers",
        ))
        size(term, 1) == size(term, 2) || throw(DimensionMismatch(
            "onsite[$site] must be square; got size $(size(term))",
        ))
    end
    return terms
end

function _finite_mpo_couplings(couplings, channels::Integer, bonds::Integer)
    channels >= 0 || throw(ArgumentError("channels must be nonnegative"))
    bonds >= 0 || throw(ArgumentError("bonds must be nonnegative"))

    if couplings isa Number
        return fill(couplings, channels, bonds)
    elseif couplings isa AbstractVector
        length(couplings) == channels || throw(DimensionMismatch(
            "couplings has length $(length(couplings)); expected one value " *
            "for each of the $channels interaction channels",
        ))
        all(value -> value isa Number, couplings) || throw(ArgumentError(
            "couplings must contain numbers",
        ))
        result = Matrix{eltype(couplings)}(undef, channels, bonds)
        for channel in 1:channels, bond in 1:bonds
            result[channel, bond] = couplings[channel]
        end
        return result
    elseif couplings isa AbstractMatrix
        size(couplings) == (channels, bonds) || throw(DimensionMismatch(
            "couplings has size $(size(couplings)); expected " *
            "($channels, $bonds)",
        ))
        eltype(couplings) <: Number || throw(ArgumentError(
            "couplings must contain numbers",
        ))
        return Matrix(couplings)
    end

    throw(ArgumentError(
        "couplings must be a scalar, a length-$channels channel vector, " *
        "or a ($channels, $bonds) channel-by-bond matrix",
    ))
end

"""
    nearest_neighbor_mpo(N, onsite, left_ops, right_ops; couplings=1)

Construct an open-boundary MPO for

`sum(n=1:N, onsite[n]) + sum(n=1:N-1, a, couplings[a,n] *
left_ops[a]_n * right_ops[a]_(n+1))`.

`onsite` may be one square matrix, used at every site, or a vector of `N`
square matrices. `left_ops` and `right_ops` are equally sized vectors of
endpoint operators. `couplings` may be a scalar, a vector with one constant
per interaction channel, or a `channels × (N-1)` matrix of bond-dependent
coefficients.

The auxiliary dimension is `length(left_ops) + 2`. Physical matrix indices
are `(output, input)`, and site 1 is the most-significant Kronecker factor,
matching [`mpo_matrix`](@ref). For `N == 1`, the result is exactly the onsite
operator and has shape `(d, d, 1, 1)`.
"""
function nearest_neighbor_mpo(
    N::Integer,
    onsite,
    left_ops::AbstractVector,
    right_ops::AbstractVector;
    couplings=1,
)
    N >= 1 || throw(ArgumentError("N must be positive"))
    length(left_ops) == length(right_ops) || throw(DimensionMismatch(
        "left_ops and right_ops must have the same length",
    ))

    onsite_terms = _finite_mpo_onsite_terms(onsite, N)
    d = size(first(onsite_terms), 1)
    d >= 1 || throw(ArgumentError("the physical dimension must be positive"))
    for (site, term) in pairs(onsite_terms)
        size(term) == (d, d) || throw(DimensionMismatch(
            "onsite[$site] has size $(size(term)); expected ($d, $d)",
        ))
    end

    channels = length(left_ops)
    for (name, operators) in (("left_ops", left_ops),
                              ("right_ops", right_ops))
        for (channel, operator) in pairs(operators)
            operator isa AbstractMatrix || throw(ArgumentError(
                "$name[$channel] must be a matrix",
            ))
            eltype(operator) <: Number || throw(ArgumentError(
                "$name[$channel] must contain numbers",
            ))
            size(operator) == (d, d) || throw(DimensionMismatch(
                "$name[$channel] has size $(size(operator)); expected ($d, $d)",
            ))
        end
    end

    coupling_matrix = _finite_mpo_couplings(couplings, channels, N - 1)
    # Promote from values rather than container eltypes.  In particular, this
    # keeps an otherwise valid heterogeneous `Number[...]` channel vector from
    # degrading the tensor element type to the abstract type `Number`.
    T = _finite_mpo_scalar_type(
        onsite_terms...,
        left_ops...,
        right_ops...,
        coupling_matrix,
    )

    converted_onsite = [Matrix{T}(term) for term in onsite_terms]
    converted_left = [Matrix{T}(operator) for operator in left_ops]
    converted_right = [Matrix{T}(operator) for operator in right_ops]
    converted_couplings = Matrix{T}(coupling_matrix)
    all(term -> all(isfinite, term), converted_onsite) || throw(ArgumentError(
        "onsite contains a non-finite entry",
    ))
    all(operator -> all(isfinite, operator), converted_left) ||
        throw(ArgumentError("left_ops contains a non-finite entry"))
    all(operator -> all(isfinite, operator), converted_right) ||
        throw(ArgumentError("right_ops contains a non-finite entry"))
    all(isfinite, converted_couplings) || throw(ArgumentError(
        "couplings contains a non-finite entry",
    ))

    if N == 1
        return [Array(reshape(copy(converted_onsite[1]), d, d, 1, 1))]
    end

    auxiliary_dimension = channels + 2
    initial_channel = auxiliary_dimension
    final_channel = 1
    identity_physical = Matrix{T}(I, d, d)
    mpo = Vector{Array{T,4}}(undef, N)

    left_boundary = zeros(T, d, d, 1, auxiliary_dimension)
    left_boundary[:, :, 1, final_channel] .= converted_onsite[1]
    for channel in 1:channels
        auxiliary_channel = channel + 1
        left_boundary[:, :, 1, auxiliary_channel] .=
            converted_couplings[channel, 1] .* converted_left[channel]
    end
    left_boundary[:, :, 1, initial_channel] .= identity_physical
    mpo[1] = left_boundary

    for site in 2:(N - 1)
        bulk = zeros(T, d, d, auxiliary_dimension, auxiliary_dimension)
        bulk[:, :, final_channel, final_channel] .= identity_physical
        bulk[:, :, initial_channel, final_channel] .= converted_onsite[site]
        bulk[:, :, initial_channel, initial_channel] .= identity_physical
        for channel in 1:channels
            auxiliary_channel = channel + 1
            bulk[:, :, auxiliary_channel, final_channel] .=
                converted_right[channel]
            bulk[:, :, initial_channel, auxiliary_channel] .=
                converted_couplings[channel, site] .* converted_left[channel]
        end
        mpo[site] = bulk
    end

    right_boundary = zeros(T, d, d, auxiliary_dimension, 1)
    right_boundary[:, :, final_channel, 1] .= identity_physical
    for channel in 1:channels
        auxiliary_channel = channel + 1
        right_boundary[:, :, auxiliary_channel, 1] .= converted_right[channel]
    end
    right_boundary[:, :, initial_channel, 1] .= converted_onsite[N]
    mpo[N] = right_boundary
    return mpo
end

"""
    nearest_neighbor_bond_hamiltonians(
        N, onsite, left_ops, right_ops; couplings=1,
    )

Build the rank-4 bond Hamiltonians for exactly the same model accepted by
[`nearest_neighbor_mpo`](@ref).  Onsite terms are placed wholly on the only
incident bond at an open boundary and split equally between the two incident
bonds in the bulk.  Consequently, summing the returned densities reproduces
the finite MPO exactly.  Their axes are `(out1,out2,in1,in2)`, ready for
[`tebd_step!`](@ref).
"""
function nearest_neighbor_bond_hamiltonians(
    N::Integer,
    onsite,
    left_ops::AbstractVector,
    right_ops::AbstractVector;
    couplings=1,
)
    N >= 2 || throw(ArgumentError(
        "bond Hamiltonians require at least two sites",
    ))
    length(left_ops) == length(right_ops) || throw(DimensionMismatch(
        "left_ops and right_ops must have the same length",
    ))
    onsite_terms = _finite_mpo_onsite_terms(onsite, N)
    d = size(first(onsite_terms), 1)
    all(term -> size(term) == (d, d), onsite_terms) ||
        throw(DimensionMismatch("all onsite terms must have size ($d, $d)"))
    for (name, operators) in (("left_ops", left_ops),
                              ("right_ops", right_ops))
        for (channel, operator) in pairs(operators)
            operator isa AbstractMatrix || throw(ArgumentError(
                "$name[$channel] must be a matrix",
            ))
            eltype(operator) <: Number || throw(ArgumentError(
                "$name[$channel] must contain numbers",
            ))
            size(operator) == (d, d) || throw(DimensionMismatch(
                "$name[$channel] has size $(size(operator)); expected ($d, $d)",
            ))
        end
    end
    coupling_matrix = _finite_mpo_couplings(
        couplings, length(left_ops), N - 1,
    )
    scalar_types = Type[Float64, eltype(coupling_matrix)]
    append!(scalar_types, (eltype(term) for term in onsite_terms))
    append!(scalar_types, (eltype(operator) for operator in left_ops))
    append!(scalar_types, (eltype(operator) for operator in right_ops))
    T = promote_type(scalar_types...)
    converted_onsite = [Matrix{T}(term) for term in onsite_terms]
    converted_left = [Matrix{T}(operator) for operator in left_ops]
    converted_right = [Matrix{T}(operator) for operator in right_ops]
    converted_couplings = Matrix{T}(coupling_matrix)
    all(term -> all(isfinite, term), converted_onsite) || throw(ArgumentError(
        "onsite contains a non-finite entry",
    ))
    all(operator -> all(isfinite, operator), converted_left) ||
        throw(ArgumentError("left_ops contains a non-finite entry"))
    all(operator -> all(isfinite, operator), converted_right) ||
        throw(ArgumentError("right_ops contains a non-finite entry"))
    all(isfinite, converted_couplings) || throw(ArgumentError(
        "couplings contains a non-finite entry",
    ))

    identity_physical = Matrix{T}(I, d, d)
    densities = Vector{Array{T,4}}(undef, N - 1)
    for bond in 1:(N - 1)
        left_weight = bond == 1 ? one(T) : one(T) / 2
        right_weight = bond == N - 1 ? one(T) : one(T) / 2
        matrix = left_weight .* kron(converted_onsite[bond], identity_physical) .+
                 right_weight .* kron(identity_physical, converted_onsite[bond + 1])
        for channel in eachindex(converted_left)
            matrix .+= converted_couplings[channel, bond] .*
                      kron(converted_left[channel], converted_right[channel])
        end
        densities[bond] = permutedims(
            reshape(matrix, d, d, d, d), (2, 1, 4, 3),
        )
    end
    return densities
end

"""
    xyz_mpo(N; Jx=0, Jy=0, Jz=0, hx=0, hy=0, hz=0, shift=0)

Construct the open-boundary spin-1/2 Hamiltonian

`sum(n=1:N-1, Jx[n] X_n X_(n+1) + Jy[n] Y_n Y_(n+1) +
Jz[n] Z_n Z_(n+1)) + sum(n=1:N, hx[n] X_n + hy[n] Y_n +
hz[n] Z_n - shift[n] I_n)`.

Each keyword accepts either a scalar or an explicit vector: exchange vectors
have length `N-1`, while field and shift vectors have length `N`. The local
operators are Pauli matrices with eigenvalues `±1`, as in the MATLAB
`xyz_mpo.m`; they are not spin operators `S = sigma/2`.
"""
function xyz_mpo(
    N::Integer;
    Jx=0,
    Jy=0,
    Jz=0,
    hx=0,
    hy=0,
    hz=0,
    shift=0,
)
    N >= 1 || throw(ArgumentError("N must be positive"))
    Jx_values = _finite_mpo_coefficient_vector(Jx, N - 1, "Jx")
    Jy_values = _finite_mpo_coefficient_vector(Jy, N - 1, "Jy")
    Jz_values = _finite_mpo_coefficient_vector(Jz, N - 1, "Jz")
    hx_values = _finite_mpo_coefficient_vector(hx, N, "hx")
    hy_values = _finite_mpo_coefficient_vector(hy, N, "hy")
    hz_values = _finite_mpo_coefficient_vector(hz, N, "hz")
    shift_values = _finite_mpo_coefficient_vector(shift, N, "shift")

    T = promote_type(
        ComplexF64,
        _finite_mpo_scalar_type(
            Jx_values, Jy_values, Jz_values,
            hx_values, hy_values, hz_values, shift_values,
        ),
    )
    identity2 = Matrix{T}(I, 2, 2)
    X = T[0 1; 1 0]
    Y = T[0 -im; im 0]
    Z = T[1 0; 0 -1]

    onsite = Vector{Matrix{T}}(undef, N)
    for site in 1:N
        onsite[site] = hx_values[site] .* X .+
                       hy_values[site] .* Y .+
                       hz_values[site] .* Z .-
                       shift_values[site] .* identity2
    end
    couplings = zeros(T, 3, N - 1)
    N > 1 && begin
        couplings[1, :] .= Jx_values
        couplings[2, :] .= Jy_values
        couplings[3, :] .= Jz_values
    end
    return nearest_neighbor_mpo(
        N, onsite, [X, Y, Z], [X, Y, Z]; couplings=couplings,
    )
end

"""
    bose_hubbard_mpo(N; hopping=1, U=0, potential=0,
                     chemical_potential=0, nmax=3)

Construct the truncated open-boundary Bose-Hubbard Hamiltonian

`-sum(n, hopping[n] bdag_n b_(n+1) + conj(hopping[n]) b_n bdag_(n+1))
 + sum(n, U[n]/2 * number_n * (number_n - 1)
              + (potential[n] - chemical_potential[n]) * number_n)`.

The local basis is `|0>, ..., |nmax>`, so its dimension is `nmax + 1`.
`hopping` is a scalar or length-`N-1` vector. `U`, `potential`, and
`chemical_potential` are scalars or length-`N` vectors. A linear tilt matching
the MATLAB `BH_mpo(N,J,U,Delta)` is obtained with
`potential = Delta .* collect(1:N)`.
"""
function bose_hubbard_mpo(
    N::Integer;
    hopping=1,
    U=0,
    potential=0,
    chemical_potential=0,
    nmax::Integer=3,
)
    N >= 1 || throw(ArgumentError("N must be positive"))
    nmax >= 0 || throw(ArgumentError("nmax must be nonnegative"))
    hopping_values = _finite_mpo_coefficient_vector(
        hopping, N - 1, "hopping",
    )
    interaction_values = _finite_mpo_coefficient_vector(U, N, "U")
    potential_values = _finite_mpo_coefficient_vector(
        potential, N, "potential",
    )
    chemical_values = _finite_mpo_coefficient_vector(
        chemical_potential, N, "chemical_potential",
    )
    T = _finite_mpo_scalar_type(
        hopping_values, interaction_values, potential_values, chemical_values,
    )

    d = nmax + 1
    annihilation = zeros(T, d, d)
    for occupation in 1:nmax
        annihilation[occupation, occupation + 1] = sqrt(occupation)
    end
    creation = Matrix(adjoint(annihilation))
    number = Matrix(Diagonal(T.(0:nmax)))

    onsite = Vector{Matrix{T}}(undef, N)
    for site in 1:N
        diagonal_entries = Vector{T}(undef, d)
        for occupation in 0:nmax
            diagonal_entries[occupation + 1] =
                interaction_values[site] * occupation * (occupation - 1) / 2 +
                (potential_values[site] - chemical_values[site]) * occupation
        end
        onsite[site] = Matrix(Diagonal(diagonal_entries))
    end

    couplings = zeros(T, 2, N - 1)
    for bond in 1:(N - 1)
        couplings[1, bond] = -hopping_values[bond]
        couplings[2, bond] = -conj(hopping_values[bond])
    end
    return nearest_neighbor_mpo(
        N,
        onsite,
        [creation, annihilation],
        [annihilation, creation];
        couplings=couplings,
    )
end

"""
    pxp_mpo(N; detuning=0, interaction=0)
    pxp_mpo(N, detuning, interaction)

Construct the edge-aware open-chain PXP Hamiltonian

`X_1 P_2 + sum(n=2:N-1, P_(n-1) X_n P_(n+1)) + P_(N-1) X_N
 + sum(n, detuning[n] Q_n) + sum(n, interaction[n] Q_n Q_(n+1))`,

where `P = (I-Z)/2 = |1><1|` and `Q = I-P = |0><0|`. Missing projectors
at an open edge are identities; consequently the constrained flip is `X` for
a one-site chain. `detuning` may be a scalar or length-`N` vector and
`interaction` a scalar or length-`N-1` vector. The positional method matches
the parameter order of the MATLAB `pxp_mpo(N,U,c)` constructor.
"""
function pxp_mpo(
    N::Integer;
    detuning=0,
    interaction=0,
)
    N >= 1 || throw(ArgumentError("N must be positive"))
    detuning_values = _finite_mpo_coefficient_vector(
        detuning, N, "detuning",
    )
    interaction_values = _finite_mpo_coefficient_vector(
        interaction, N - 1, "interaction",
    )
    T = _finite_mpo_scalar_type(detuning_values, interaction_values)
    identity2 = Matrix{T}(I, 2, 2)
    X = T[0 1; 1 0]
    P = T[0 0; 0 1]
    Q = T[1 0; 0 0]

    if N == 1
        local_operator = X .+ detuning_values[1] .* Q
        return [Array(reshape(local_operator, 2, 2, 1, 1))]
    elseif N == 2
        onsite = [detuning_values[site] .* Q for site in 1:2]
        couplings = reshape(
            T[one(T), one(T), interaction_values[1]], 3, 1,
        )
        return nearest_neighbor_mpo(
            2,
            onsite,
            [X, P, Q],
            [P, X, Q];
            couplings=couplings,
        )
    end

    auxiliary_dimension = 6
    mpo = Vector{Array{T,4}}(undef, N)

    left_boundary = zeros(T, 2, 2, 1, auxiliary_dimension)
    left_boundary[:, :, 1, 1] .= detuning_values[1] .* Q
    left_boundary[:, :, 1, 3] .= P
    left_boundary[:, :, 1, 4] .= interaction_values[1] .* Q
    left_boundary[:, :, 1, 5] .= X
    left_boundary[:, :, 1, 6] .= identity2
    mpo[1] = left_boundary

    for site in 2:(N - 1)
        bulk = zeros(T, 2, 2, auxiliary_dimension, auxiliary_dimension)
        bulk[:, :, 1, 1] .= identity2
        bulk[:, :, 2, 1] .= P
        bulk[:, :, 3, 2] .= X
        bulk[:, :, 4, 1] .= Q
        bulk[:, :, 6, 1] .= detuning_values[site] .* Q
        bulk[:, :, 6, 3] .= P
        bulk[:, :, 6, 4] .= interaction_values[site] .* Q
        bulk[:, :, 6, 6] .= identity2

        site == 2 && (bulk[:, :, 5, 1] .= P)
        site == N - 1 && (bulk[:, :, 6, 5] .= P)
        mpo[site] = bulk
    end

    right_boundary = zeros(T, 2, 2, auxiliary_dimension, 1)
    right_boundary[:, :, 1, 1] .= identity2
    right_boundary[:, :, 2, 1] .= P
    right_boundary[:, :, 4, 1] .= Q
    right_boundary[:, :, 5, 1] .= X
    right_boundary[:, :, 6, 1] .= detuning_values[N] .* Q
    mpo[N] = right_boundary
    return mpo
end

pxp_mpo(N::Integer, detuning::Number, interaction::Number) =
    pxp_mpo(N; detuning=detuning, interaction=interaction)

"""
    ising_mpo(N; J=-1, hx=0, hy=0, hz=0, shift=0, interaction=:x)

Construct the open-boundary MPO

`J ∑ Sₙ Sₙ₊₁ + ∑ (hx Xₙ + hy Yₙ + hz Zₙ - shift Iₙ)`,

where `S` is `X` for `interaction=:x` and `Z` for `interaction=:z`.
The default therefore matches the transverse-field-Ising convention used by
the finite MATLAB examples (`ising_mpox`).
"""
function ising_mpo(N::Integer;
                   J::Number=-1,
                   hx::Number=0,
                   hy::Number=0,
                   hz::Number=0,
                   shift::Number=0,
                   interaction::Symbol=:x)
    N >= 1 || throw(ArgumentError("N must be positive"))
    interaction in (:x, :z) || throw(ArgumentError(
        "interaction must be :x or :z"))

    scalar_type = promote_type(Float64, typeof(J), typeof(hx), typeof(hy),
                               typeof(hz), typeof(shift))
    T = scalar_type <: Real && iszero(hy) ? scalar_type :
        promote_type(scalar_type, ComplexF64)

    identity2 = Matrix{T}(I, 2, 2)
    X = T[0 1; 1 0]
    Z = T[1 0; 0 -1]
    Y = zeros(T, 2, 2)
    if !iszero(hy)
        Y[1, 2] = convert(T, -im)
        Y[2, 1] = convert(T, im)
    end
    onsite = hx .* X .+ hy .* Y .+ hz .* Z .- shift .* identity2
    coupling_operator = interaction === :x ? X : Z

    if N == 1
        return [Array(reshape(copy(onsite), 2, 2, 1, 1))]
    end

    left = zeros(T, 2, 2, 1, 3)
    left[:, :, 1, 1] .= onsite
    left[:, :, 1, 2] .= J .* coupling_operator
    left[:, :, 1, 3] .= identity2

    bulk = zeros(T, 2, 2, 3, 3)
    bulk[:, :, 1, 1] .= identity2
    bulk[:, :, 2, 1] .= coupling_operator
    bulk[:, :, 3, 1] .= onsite
    bulk[:, :, 3, 2] .= J .* coupling_operator
    bulk[:, :, 3, 3] .= identity2

    right = zeros(T, 2, 2, 3, 1)
    right[:, :, 1, 1] .= identity2
    right[:, :, 2, 1] .= coupling_operator
    right[:, :, 3, 1] .= onsite

    mpo = Vector{Array{T,4}}(undef, N)
    mpo[1] = left
    for n in 2:(N - 1)
        mpo[n] = copy(bulk)
    end
    mpo[N] = right
    return mpo
end

function ising_mpo(N::Integer, J::Number, hx::Number, hy::Number,
                   hz::Number, shift::Number; interaction::Symbol=:x)
    return ising_mpo(N; J=J, hx=hx, hy=hy, hz=hz, shift=shift,
                     interaction=interaction)
end
