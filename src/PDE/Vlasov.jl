# Moderate-grid reference solver: dense evolution, followed by an MPS filter.
# No derivative MPOs or time evolution inside the MPS representation.
struct VlasovGrid
    x::Vector{Float64}
    v::Vector{Float64}
    dx::Float64
    dv::Float64
    kx::Vector{Float64}
    kv::Vector{Float64}
end

# MATLAB convention: the constant and (on even grids) Nyquist frequencies
# both have k=0. The Nyquist coefficient is preserved, but not translated.
function _vlasov_wavenumbers(n, period)
    h = n ÷ 2
    modes = iseven(n) ? vcat(0:(h - 1), 0, (1 - h):-1) : vcat(0:h, (-h):-1)
    return (2pi / period) .* modes
end

"""
    VlasovGrid(nx, nv; Lx=2pi, vmax=6)

Uniform grids on [0,Lx) and [-vmax,vmax), periodic in BOTH coordinates.
Cached wavenumbers use the MATLAB convention: the constant and Nyquist
frequencies are zero. Poisson solving and translations use FFTW.
Keep the velocity distribution negligible at the periodic seam.
"""
function VlasovGrid(nx::Integer, nv::Integer; Lx::Real=2pi, vmax::Real=6)
    nx >= 2 && nv >= 2 || throw(ArgumentError("grid sizes must be at least 2"))
    isfinite(Lx) && Lx > 0 && isfinite(vmax) && vmax > 0 ||
        throw(ArgumentError("Lx and vmax must be finite and positive"))
    dx, dv = Float64(Lx / nx), Float64(2vmax / nv)
    x = collect(0:(nx - 1)) .* dx
    v = -Float64(vmax) .+ collect(0:(nv - 1)) .* dv
    return VlasovGrid(x, v, dx, dv,
                      _vlasov_wavenumbers(nx, Lx), _vlasov_wavenumbers(nv, 2vmax))
end

function _check_vlasov_grid(f, grid)
    size(f) == (length(grid.x), length(grid.v)) ||
        throw(DimensionMismatch("distribution and Vlasov grid differ"))
    all(isfinite, f) || throw(ArgumentError("distribution must be finite"))
end

"""Self-consistent field, with convention ∂x E = ρ - mean(ρ)."""
function vlasov_field(f::AbstractMatrix{<:Real}, grid::VlasovGrid)
    _check_vlasov_grid(f, grid)
    rho = vec(sum(f; dims=2)) .* grid.dv
    rho .-= sum(rho) / length(rho)
    rho_hat = fft(rho)
    field_hat = zeros(ComplexF64, length(rho))
    for i in eachindex(grid.kx)
        if grid.kx[i] != 0
            field_hat[i] = rho_hat[i] / (im * grid.kx[i])
        end
    end
    return real.(ifft(field_hat))
end

function _check_vlasov_advection(advection)
    advection in (:fourier, :linear) ||
        throw(ArgumentError("advection must be :fourier or :linear"))
end

# fft(f, dimension) transforms each line independently. This is exactly
# real(ifft(exp(-im*k*shift).*fft(line))) in MATLAB, batched across lines.
function _vlasov_fourier_advect(f, speeds, dt, wavenumbers, dimension; nyquist::Symbol=:keep)
    nyquist in (:keep, :drop) || throw(ArgumentError("nyquist must be :keep or :drop"))
    rates = dimension == 1 ? reshape(wavenumbers, :, 1) .* reshape(speeds, 1, :) :
                             reshape(speeds, :, 1) .* reshape(wavenumbers, 1, :)
    angles = dt .* rates
    all(isfinite, angles) || throw(ArgumentError("Fourier shift phase must be finite"))
    transformed = exp.(-im .* angles) .* fft(f, dimension)
    if nyquist == :drop && iseven(size(f, dimension))
        selectdim(transformed, dimension, size(f, dimension) ÷ 2 + 1) .= 0
    end
    return real.(ifft(transformed, dimension))
end

# Each line has a constant translation. Linear interpolation on a periodic
# grid is positive and preserves the sum, but introduces numerical diffusion.
function _vlasov_advect(f, speeds, dt, spacing, dimension)
    out = similar(f)
    n = size(f, dimension)
    for line in eachindex(speeds)
        displacement = speeds[line] * dt / spacing
        isfinite(displacement) || throw(ArgumentError("advection displacement must be finite"))
        shift = mod(displacement, n)
        for i in 1:n
            # Wrap the INTEGER neighbours, not the floating coordinate: mod
            # of a tiny negative float can round to n at the periodic seam.
            q = i - 1 - shift
            j = floor(Int, q)
            a = q - j
            j0, j1 = mod(j, n) + 1, mod(j + 1, n) + 1
            if dimension == 1
                out[i, line] = (1 - a) * f[j0, line] + a * f[j1, line]
            else
                out[line, i] = (1 - a) * f[line, j0] + a * f[line, j1]
            end
        end
    end
    return out
end

"""
    vlasov_step(f, grid, dt; time=0, drive=(x,t)->0.0, advection=:fourier)

Strang split ∂t f + v ∂x f + (E[f]+drive(x,t)) ∂v f = 0:
half x shift, self field + midpoint drive, full v shift, half x shift.
Uses MATLAB-equivalent Fourier translations by default. `advection=:linear`
selects the earlier diffusive interpolation for comparison. Returns a new grid.
`drive` is an externally prescribed acceleration in the same sign convention
as E. There is no projection or clipping within this PDE step.
"""
function vlasov_step(f::AbstractMatrix{<:Real}, grid::VlasovGrid, dt::Real;
                     time::Real=0, drive=(x, t) -> 0.0, advection::Symbol=:fourier)
    _check_vlasov_grid(f, grid)
    _check_vlasov_advection(advection)
    isfinite(dt) && dt > 0 && isfinite(time) ||
        throw(ArgumentError("dt must be positive and time finite"))
    half = advection == :fourier ?
        _vlasov_fourier_advect(f, grid.v, dt / 2, grid.kx, 1) :
        _vlasov_advect(float.(f), grid.v, dt / 2, grid.dx, 1)
    field = vlasov_field(half, grid) .+ [drive(x, time + dt / 2) for x in grid.x]
    all(isfinite, field) || throw(ArgumentError("driving field must be finite"))
    if advection == :fourier
        kicked = _vlasov_fourier_advect(half, field, dt, grid.kv, 2)
        return _vlasov_fourier_advect(kicked, grid.v, dt / 2, grid.kx, 1)
    end
    kicked = _vlasov_advect(half, field, dt, grid.dv, 2)
    return _vlasov_advect(kicked, grid.v, dt / 2, grid.dx, 1)
end

"""Evolve densely for steps timesteps, without any MPS projection."""
function evolve_vlasov(f::AbstractMatrix{<:Real}, grid::VlasovGrid;
                       dt::Real, steps::Integer, time::Real=0,
                       drive=(x, t) -> 0.0, advection::Symbol=:fourier)
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    _check_vlasov_grid(f, grid)
    _check_vlasov_advection(advection)
    isfinite(dt) && dt > 0 && isfinite(time) ||
        throw(ArgumentError("dt must be positive and time finite"))
    current = float.(f)
    for step in 1:steps
        current = vlasov_step(current, grid, dt; time=time + (step - 1) * dt,
                              drive=drive, advection=advection)
    end
    return current
end

"""Physical diagnostics; energy excludes external driving potential."""
function vlasov_diagnostics(f::AbstractMatrix{<:Real}, grid::VlasovGrid)
    E = vlasov_field(f, grid)
    cell = grid.dx * grid.dv
    mass = cell * sum(f)
    momentum = cell * sum(f .* reshape(grid.v, 1, :))
    kinetic = cell * sum(f .* reshape(grid.v .^ 2, 1, :)) / 2
    field_energy = grid.dx * sum(abs2, E) / 2
    return (mass=mass, momentum=momentum, energy=kinetic + field_energy,
            field_l2=sqrt(2field_energy), minimum=minimum(f),
            negative_mass=-cell * sum(min.(f, 0)),
            velocity_edge_max=maximum(abs, f[:, [1, size(f, 2)]]))
end

"""
    vlasov_scarfinder(f0, grid; maxdim, dt, steps_per_iteration,
                     iterations=50, layout=phase_space_layout(size(f0)...),
                     cutoff=0, restore_mass=true, time=0, drive=(x,t)->0.0,
                     advection=:fourier)

Project initially, then repeatedly evolve densely for τ=dt*steps_per_iteration
and truncate to an MPS. Time advances continuously, including for driving.
Scalar mass restoration preserves ranks; negativity is reported without
clipping. No momentum or energy constraint is imposed.

Returns final grid/MPS, initial projection, per-iteration diagnostics, and an
extra UNPROJECTED cycle from the final state to measure PDE recurrence. For
a periodic drive choose τ as an integer number of drive periods.
"""
function vlasov_scarfinder(f0::AbstractMatrix{<:Real}, grid::VlasovGrid;
                           maxdim::Integer, dt::Real, steps_per_iteration::Integer,
                           iterations::Integer=50,
                           layout::PhaseSpaceLayout=phase_space_layout(size(f0)...),
                           cutoff::Real=0, restore_mass::Bool=true,
                           time::Real=0, drive=(x, t) -> 0.0,
                           advection::Symbol=:fourier)
    _check_vlasov_grid(f0, grid)
    _check_vlasov_advection(advection)
    iterations >= 0 && steps_per_iteration >= 1 ||
        throw(ArgumentError("iterations must be non-negative; steps_per_iteration positive"))
    isfinite(dt) && dt > 0 && isfinite(time) ||
        throw(ArgumentError("dt must be positive and time finite"))
    target = restore_mass ? sum(f0) : nothing
    initial = project_phase_space(f0, layout; maxdim=maxdim, cutoff=cutoff, target_sum=target)
    projected = initial
    current = initial.grid
    history = NamedTuple[]
    tau = dt * steps_per_iteration
    for iteration in 1:iterations
        start_time = time + (iteration - 1) * tau
        evolved = evolve_vlasov(current, grid; dt=dt, steps=steps_per_iteration,
                               time=start_time, drive=drive, advection=advection)
        projected = project_phase_space(evolved, layout; maxdim=maxdim,
                                        cutoff=cutoff, target_sum=target)
        step_error = norm(projected.grid - current) / max(norm(current), eps(Float64))
        push!(history, merge((iteration=iteration, time=start_time + tau,
            step_error=step_error, projection_error=projected.relative_error,
            discarded_weight=projected.discarded_weight, mass_scale=projected.mass_scale,
            max_bond=maximum(size(A, 2) for A in projected.state)),
            vlasov_diagnostics(projected.grid, grid)))
        current = projected.grid
    end
    final_time = time + iterations * tau
    probe = evolve_vlasov(current, grid; dt=dt, steps=steps_per_iteration,
                         time=final_time, drive=drive, advection=advection)
    cycle_error = norm(probe - current) / max(norm(current), eps(Float64))
    return (grid=current, state=projected.state, layout=layout, history=history,
            initial_projection=initial, initial_diagnostics=vlasov_diagnostics(initial.grid, grid),
            time=final_time, tau=tau, unprojected_cycle=probe, cycle_error=cycle_error)
end
