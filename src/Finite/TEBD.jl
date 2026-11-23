# Finite-chain time-evolving block decimation for nearest-neighbour models.
#
# The sweep directions follow the canonical MATLAB implementation rather
# than applying SVDs independently to disjoint bonds.  Moving the
# orthogonality centre between gates makes every truncated SVD a Schmidt
# truncation of the full state.

"""
    two_site_gate(h, dt; time=:real)

Exponentiate a two-site Hamiltonian `h[out1,out2,in1,in2]`.  The returned
gate has the same axis order and represents `exp(-im*dt*h)` for real time or
`exp(-dt*h)` for imaginary time.
"""
function two_site_gate(
    h::AbstractArray{<:Number,4},
    dt::Number;
    time::Symbol=:real,
)
    d1out, d2out, d1in, d2in = size(h)
    d1out == d1in || throw(DimensionMismatch(
        "the first-site output/input dimensions differ: $d1out and $d1in",
    ))
    d2out == d2in || throw(DimensionMismatch(
        "the second-site output/input dimensions differ: $d2out and $d2in",
    ))
    parameter = _tdvp_parameter(dt, time)
    matrix = reshape(
        permutedims(h, (2, 1, 4, 3)),
        d1out * d2out,
        d1in * d2in,
    )
    gate_matrix = exp(parameter .* matrix)
    return permutedims(
        reshape(gate_matrix, d2out, d1out, d2in, d1in),
        (2, 1, 4, 3),
    )
end

function _check_gate_storage(A, gate, name)
    result_type = promote_type(eltype(A), eltype(gate))
    result_type <: eltype(A) || throw(ArgumentError(
        "$name has element type $(eltype(A)) and cannot store gate results " *
        "of type $result_type; use a compatible (usually complex) MPS",
    ))
    return nothing
end

"""
    apply_one_site_gate!(psi, site, gate)

Apply `gate[out,in]` to one MPS site.  The state is mutated and returned.
"""
function apply_one_site_gate!(
    psi::AbstractVector,
    site::Integer,
    gate::AbstractMatrix{<:Number},
)
    _check_mps(psi)
    1 <= site <= length(psi) || throw(BoundsError(psi, site))
    A = psi[site]
    d = size(A, 3)
    size(gate) == (d, d) || throw(DimensionMismatch(
        "gate must have size ($d, $d) at site $site; got $(size(gate))",
    ))
    _check_gate_storage(A, gate, "psi[$site]")
    @tensor updated[l, r, sout] := A[l, r, sin] * gate[sout, sin]
    psi[site] = updated
    return psi
end

"""
    apply_two_site_gate!(psi, bond, gate;
                         direction=:right, maxdim=typemax(Int), cutoff=0)

Apply `gate[out1,out2,in1,in2]` to sites `bond,bond+1`, then split the
two-site tensor.  `direction=:right` leaves the orthogonality centre on site
`bond+1`; `direction=:left` leaves it on site `bond`.  The routine does not
move the surrounding gauge, so direct callers should put the MPS centre on
the acted-on bond when using truncation.  `tebd_layer!` and `tebd_step!` do
this automatically.

Returns `(discarded_weight, kept_dim, orthogonality_center)`.
"""
function apply_two_site_gate!(
    psi::AbstractVector,
    bond::Integer,
    gate::AbstractArray{<:Number,4};
    direction::Symbol=:right,
    maxdim::Integer=typemax(Int),
    cutoff::Real=0.0,
)
    _check_mps(psi)
    1 <= bond < length(psi) || throw(BoundsError(psi, bond))
    maxdim >= 1 || throw(ArgumentError("maxdim must be at least 1"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative"))
    direction in (:left, :right) || throw(ArgumentError(
        "direction must be :left or :right",
    ))

    A = psi[bond]
    B = psi[bond + 1]
    d1 = size(A, 3)
    d2 = size(B, 3)
    size(gate) == (d1, d2, d1, d2) || throw(DimensionMismatch(
        "gate on bond $bond must have size ($d1, $d2, $d1, $d2); " *
        "got $(size(gate))",
    ))
    result_type = promote_type(eltype(A), eltype(B), eltype(gate))
    result_type <: eltype(A) || throw(ArgumentError(
        "psi[$bond] has element type $(eltype(A)) and cannot store the " *
        "two-site gate result of type $result_type",
    ))
    result_type <: eltype(B) || throw(ArgumentError(
        "psi[$(bond + 1)] has element type $(eltype(B)) and cannot store " *
        "the two-site gate result of type $result_type",
    ))

    centre = contract_two_sites(A, B)
    @tensor evolved[l, r, sout, tout] :=
        centre[l, r, sin, tin] * gate[sout, tout, sin, tin]

    if direction === :right
        left, right, discarded = split_left(
            evolved; maxdim=maxdim, cutoff=cutoff,
        )
        orthogonality_center = bond + 1
    else
        left, right, discarded = split_right(
            evolved; maxdim=maxdim, cutoff=cutoff,
        )
        orthogonality_center = bond
    end
    psi[bond] = left
    psi[bond + 1] = right
    return (
        discarded_weight=Float64(real(discarded)),
        kept_dim=size(left, 2),
        orthogonality_center=orthogonality_center,
    )
end

function _left_canonicalize_to_right!(psi)
    for site in 1:(length(psi) - 1)
        C, AL = left_orthogonalize(psi[site])
        psi[site] = AL
        psi[site + 1] = _absorb_left(C, psi[site + 1])
    end
    return psi
end

function _right_canonicalize_to_left!(psi)
    for site in length(psi):-1:2
        C, AR = right_orthogonalize(psi[site])
        psi[site] = AR
        psi[site - 1] = _absorb_right(psi[site - 1], C)
    end
    return psi
end

function _check_tebd_bonds(bonds, N)
    selected = sort!(collect(Int, bonds))
    all(bond -> 1 <= bond < N, selected) || throw(ArgumentError(
        "every TEBD bond must lie in 1:$(N - 1)",
    ))
    length(unique(selected)) == length(selected) || throw(ArgumentError(
        "TEBD bonds must not be repeated",
    ))
    all(diff(selected) .> 1) || throw(ArgumentError(
        "a TEBD layer may contain only disjoint bonds",
    ))
    return selected
end

function _tebd_layer_canonical!(
    psi,
    gates,
    selected;
    direction,
    maxdim,
    cutoff,
)
    N = length(psi)
    chosen = falses(max(N - 1, 0))
    chosen[selected] .= true
    discarded_weights = Float64[]
    kept_dimensions = Int[]

    if direction === :left
        # The centre starts at N and is transported to site 1.
        site = N
        while site > 1
            bond = site - 1
            if chosen[bond]
                diagnostic = apply_two_site_gate!(
                    psi, bond, gates[bond];
                    direction=:left, maxdim=maxdim, cutoff=cutoff,
                )
                push!(discarded_weights, diagnostic.discarded_weight)
                push!(kept_dimensions, diagnostic.kept_dim)
            else
                C, AR = right_orthogonalize(psi[site])
                psi[site] = AR
                psi[bond] = _absorb_right(psi[bond], C)
            end
            site -= 1
        end
    elseif direction === :right
        # The centre starts at 1 and is transported to site N.
        site = 1
        while site < N
            if chosen[site]
                diagnostic = apply_two_site_gate!(
                    psi, site, gates[site];
                    direction=:right, maxdim=maxdim, cutoff=cutoff,
                )
                push!(discarded_weights, diagnostic.discarded_weight)
                push!(kept_dimensions, diagnostic.kept_dim)
            else
                C, AL = left_orthogonalize(psi[site])
                psi[site] = AL
                psi[site + 1] = _absorb_left(C, psi[site + 1])
            end
            site += 1
        end
    else
        throw(ArgumentError("direction must be :left or :right"))
    end

    return (
        discarded_weights=discarded_weights,
        kept_dimensions=kept_dimensions,
    )
end

"""
    tebd_layer!(psi, gates, bonds; direction, maxdim=typemax(Int), cutoff=0)

Apply one layer of mutually disjoint two-site gates.  `gates` has one entry
per chain bond, while `bonds` selects the entries to apply.  A canonical
sweep is performed so truncations use the full-state Schmidt spectrum.
`direction=:left` sweeps right-to-left and leaves the centre at site 1;
`:right` sweeps left-to-right and leaves it at the last site.
"""
function tebd_layer!(
    psi::AbstractVector,
    gates::AbstractVector,
    bonds;
    direction::Symbol,
    maxdim::Integer=typemax(Int),
    cutoff::Real=0.0,
)
    _check_mps(psi)
    N = length(psi)
    length(gates) == N - 1 || throw(DimensionMismatch(
        "gates must contain one entry per bond; got $(length(gates)) for $N sites",
    ))
    maxdim >= 1 || throw(ArgumentError("maxdim must be at least 1"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative"))
    selected = _check_tebd_bonds(bonds, N)
    direction in (:left, :right) || throw(ArgumentError(
        "direction must be :left or :right",
    ))
    if direction === :left
        _left_canonicalize_to_right!(psi)
    else
        _right_canonicalize_to_left!(psi)
    end
    return _tebd_layer_canonical!(
        psi, gates, selected;
        direction=direction, maxdim=maxdim, cutoff=cutoff,
    )
end

function _check_bond_hamiltonians(psi, bond_hamiltonians)
    _check_mps(psi)
    N = length(psi)
    length(bond_hamiltonians) == N - 1 || throw(DimensionMismatch(
        "bond_hamiltonians must have length $(N - 1); " *
        "got $(length(bond_hamiltonians))",
    ))
    for bond in eachindex(bond_hamiltonians)
        h = bond_hamiltonians[bond]
        h isa AbstractArray{<:Number,4} || throw(ArgumentError(
            "bond_hamiltonians[$bond] must be a rank-4 numeric array",
        ))
        d1 = size(psi[bond], 3)
        d2 = size(psi[bond + 1], 3)
        size(h) == (d1, d2, d1, d2) || throw(DimensionMismatch(
            "bond_hamiltonians[$bond] must have size " *
            "($d1, $d2, $d1, $d2); got $(size(h))",
        ))
    end
    return N
end

function _tebd_gates(bond_hamiltonians, dt; time)
    return [two_site_gate(h, dt; time=time) for h in bond_hamiltonians]
end

function _append_layer_diagnostics!(discarded, dimensions, diagnostic)
    append!(discarded, diagnostic.discarded_weights)
    append!(dimensions, diagnostic.kept_dimensions)
    return nothing
end

function _tebd_first_order_stage!(
    psi,
    bond_hamiltonians,
    dt;
    time,
    maxdim,
    cutoff,
    discarded,
    dimensions,
)
    N = length(psi)
    gates = _tebd_gates(bond_hamiltonians, dt; time=time)
    _left_canonicalize_to_right!(psi)
    odd = _tebd_layer_canonical!(
        psi, gates, 1:2:(N - 1);
        direction=:left, maxdim=maxdim, cutoff=cutoff,
    )
    _append_layer_diagnostics!(discarded, dimensions, odd)
    even = _tebd_layer_canonical!(
        psi, gates, 2:2:(N - 1);
        direction=:right, maxdim=maxdim, cutoff=cutoff,
    )
    _append_layer_diagnostics!(discarded, dimensions, even)
    return nothing
end

function _tebd_second_order_stage!(
    psi,
    bond_hamiltonians,
    dt;
    time,
    maxdim,
    cutoff,
    discarded,
    dimensions,
)
    N = length(psi)
    odd_gates = _tebd_gates(bond_hamiltonians, dt / 2; time=time)
    even_gates = _tebd_gates(bond_hamiltonians, dt; time=time)
    _left_canonicalize_to_right!(psi)
    odd_left = _tebd_layer_canonical!(
        psi, odd_gates, 1:2:(N - 1);
        direction=:left, maxdim=maxdim, cutoff=cutoff,
    )
    _append_layer_diagnostics!(discarded, dimensions, odd_left)
    even = _tebd_layer_canonical!(
        psi, even_gates, 2:2:(N - 1);
        direction=:right, maxdim=maxdim, cutoff=cutoff,
    )
    _append_layer_diagnostics!(discarded, dimensions, even)
    odd_right = _tebd_layer_canonical!(
        psi, odd_gates, 1:2:(N - 1);
        direction=:left, maxdim=maxdim, cutoff=cutoff,
    )
    _append_layer_diagnostics!(discarded, dimensions, odd_right)
    return nothing
end

function _tebd_diagnostics(psi, norm_before, discarded, dimensions, order)
    norm_after = Float64(real(overlap(psi, psi)))
    maximum_discarded = isempty(discarded) ? 0.0 : maximum(discarded)
    maximum_bond = length(psi) == 1 ? 1 :
                   maximum(size(psi[site], 2) for site in 1:(length(psi) - 1))
    return (
        order=order,
        norm_before=norm_before,
        norm_after=norm_after,
        discarded_weight=maximum_discarded,
        total_discarded_weight=sum(discarded),
        discarded_weights=discarded,
        kept_dimensions=dimensions,
        max_bond=maximum_bond,
    )
end

"""
    tebd_step!(psi, bond_hamiltonians, dt;
               order=2, time=:real, maxdim=typemax(Int), cutoff=0,
               normalize=(time in (:imag, :imaginary, :gradient)))

Evolve an open MPS by one nearest-neighbour TEBD step.  The Hamiltonian is
`sum(bond_hamiltonians)` with one rank-4 density per bond; onsite terms must
therefore be distributed between adjacent bond densities.  Orders 1, 2, and
4 are supported.  Order 4 is the five-stage Suzuki composition and includes
a negative middle substep.

The returned diagnostics report norms, every relative discarded squared
weight, and the observed maximum bond dimension.  Imaginary time is
normalized by default; real time is left unnormalized so norm loss remains
visible.  Pass `normalize=true` to normalize either evolution explicitly.
"""
function tebd_step!(
    psi::AbstractVector,
    bond_hamiltonians::AbstractVector,
    dt::Number;
    order::Integer=2,
    time::Symbol=:real,
    maxdim::Integer=typemax(Int),
    cutoff::Real=0.0,
    normalize::Bool=(time in (:imag, :imaginary, :gradient)),
)
    _check_bond_hamiltonians(psi, bond_hamiltonians)
    order in (1, 2, 4) || throw(ArgumentError("order must be 1, 2, or 4"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be at least 1"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative"))
    parameter = _tdvp_parameter(dt, time)
    _check_tdvp_scalar_type(psi, bond_hamiltonians, parameter)

    norm_before = Float64(real(overlap(psi, psi)))
    norm_before > 0 || throw(ArgumentError("the input MPS has zero norm"))
    discarded = Float64[]
    dimensions = Int[]

    if order == 1
        _tebd_first_order_stage!(
            psi, bond_hamiltonians, dt;
            time=time, maxdim=maxdim, cutoff=cutoff,
            discarded=discarded, dimensions=dimensions,
        )
    elseif order == 2
        _tebd_second_order_stage!(
            psi, bond_hamiltonians, dt;
            time=time, maxdim=maxdim, cutoff=cutoff,
            discarded=discarded, dimensions=dimensions,
        )
    else
        p = 1 / (4 - cbrt(4.0))
        for coefficient in (p, p, 1 - 4p, p, p)
            _tebd_second_order_stage!(
                psi, bond_hamiltonians, coefficient * dt;
                time=time, maxdim=maxdim, cutoff=cutoff,
                discarded=discarded, dimensions=dimensions,
            )
        end
    end

    normalize && normalize!(psi)
    return _tebd_diagnostics(psi, norm_before, discarded, dimensions, order)
end

function tebd_step!(
    psi::AbstractVector,
    bond_hamiltonian::AbstractArray{<:Number,4},
    dt::Number;
    kwargs...,
)
    length(psi) >= 2 || throw(ArgumentError(
        "a two-site TEBD Hamiltonian requires at least two sites",
    ))
    return tebd_step!(
        psi, fill(bond_hamiltonian, length(psi) - 1), dt; kwargs...,
    )
end

"""
    tebd!(psi, bond_hamiltonians, dt, nsteps; kwargs...)

Apply `nsteps` calls to `tebd_step!` and return aggregate diagnostics plus the
per-step history.  `psi` is mutated.
"""
function tebd!(
    psi::AbstractVector,
    bond_hamiltonians,
    dt::Number,
    nsteps::Integer;
    kwargs...,
)
    nsteps >= 0 || throw(ArgumentError("nsteps must be non-negative"))
    norm_before = Float64(real(overlap(psi, psi)))
    history = NamedTuple[]
    for _ in 1:nsteps
        push!(history, tebd_step!(psi, bond_hamiltonians, dt; kwargs...))
    end
    maximum_discarded = isempty(history) ? 0.0 :
        maximum(step.discarded_weight for step in history)
    total_discarded = sum(step.total_discarded_weight for step in history;
                          init=0.0)
    maximum_bond = length(psi) == 1 ? 1 :
        maximum(size(psi[site], 2) for site in 1:(length(psi) - 1))
    return (
        steps=nsteps,
        norm_before=norm_before,
        norm_after=Float64(real(overlap(psi, psi))),
        discarded_weight=maximum_discarded,
        total_discarded_weight=total_discarded,
        max_bond=maximum_bond,
        history=history,
    )
end
