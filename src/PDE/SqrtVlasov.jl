# A real, signed amplitude g represents a nonnegative density f=g^2.
# Keep g between steps and projections: resetting it to abs(g) changes the
# spectral discretization and discards the sign information being evolved.

"""
    project_vlasov_amplitude(g, layout; maxdim, cutoff=0, target_sum_squares=nothing)

TT-SVD of the real amplitude g, with an optional scalar L2 rescaling to
restore sum(g²), i.e. density mass on a uniform grid. No positivity clipping.
Returns an MPS for g (`state`), its dense `amplitude`, and density `grid=g²`.
`maxdim` bounds amplitude bonds; density bonds can be as large as maxdim².
The discarded weight measures amplitude SVD loss before rescaling.
"""
function project_vlasov_amplitude(g::AbstractMatrix{<:Real}, layout::PhaseSpaceLayout;
                                  maxdim::Integer, cutoff::Real=0, target_sum_squares=nothing)
    p = phase_space_mps(g, layout; maxdim=maxdim, cutoff=cutoff)
    amplitude = mps_phase_space(p.state, layout)
    scale = 1.0
    if target_sum_squares !== nothing
        isfinite(target_sum_squares) && target_sum_squares > 0 ||
            throw(ArgumentError("target_sum_squares must be finite and positive"))
        retained = sum(abs2, amplitude)
        isfinite(retained) && retained > 0 ||
            throw(ArgumentError("cannot restore the mass of a zero amplitude"))
        scale = sqrt(target_sum_squares / retained)
        p.state[end] .*= scale
        amplitude .*= scale
    end
    f = amplitude .^ 2
    return merge(p, (amplitude=amplitude, grid=f, amplitude_scale=scale,
        amplitude_error=norm(amplitude - g)/max(norm(g), eps(Float64)),
        density_error=norm(f - g.^2)/max(norm(g.^2), eps(Float64))))
end

"""
    vlasov_amplitude_step(g, grid, dt; time=0, drive=(x,t)->0.0, nyquist=:keep)

Strang evolution of ∂t g + v∂x g + (E[g²]+drive)∂v g = 0.
The self field is evaluated from the squared amplitude after the half x step.
The returned real g may have either sign; f=g² is always nonnegative.
With MATLAB's default `nyquist=:keep`, Fourier shifts preserve sum(g²) to
roundoff. `nyquist=:drop` removes the translated direction's Nyquist mode;
that diagnostic option can lose mass and is a separate spectral filter.
Continuum equivalence to density evolution does not imply identical finite
grid discretizations. Check resolution and density Casimirs as well as mass.
"""
function vlasov_amplitude_step(g::AbstractMatrix{<:Real}, grid::VlasovGrid, dt::Real;
                               time::Real=0, drive=(x,t)->0.0, nyquist::Symbol=:keep)
    _check_vlasov_grid(g, grid)
    isfinite(dt) && dt > 0 && isfinite(time) ||
        throw(ArgumentError("dt must be positive and time finite"))
    half = _vlasov_fourier_advect(g, grid.v, dt/2, grid.kx, 1; nyquist=nyquist)
    field = vlasov_field(half.^2, grid) .+ [drive(x, time + dt/2) for x in grid.x]
    all(isfinite, field) || throw(ArgumentError("driving field must be finite"))
    kicked = _vlasov_fourier_advect(half, field, dt, grid.kv, 2; nyquist=nyquist)
    return _vlasov_fourier_advect(kicked, grid.v, dt/2, grid.kx, 1; nyquist=nyquist)
end

"""Evolve a signed real amplitude densely, without MPS truncation or mass restoration."""
function evolve_vlasov_amplitude(g::AbstractMatrix{<:Real}, grid::VlasovGrid;
                                 dt::Real, steps::Integer, time::Real=0,
                                 drive=(x,t)->0.0, nyquist::Symbol=:keep)
    _check_vlasov_grid(g, grid)
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    isfinite(dt) && dt > 0 && isfinite(time) ||
        throw(ArgumentError("dt must be positive and time finite"))
    nyquist in (:keep, :drop) || throw(ArgumentError("nyquist must be :keep or :drop"))
    current = float.(g)
    for step in 1:steps
        current = vlasov_amplitude_step(current, grid, dt;
            time=time + (step-1)*dt, drive=drive, nyquist=nyquist)
    end
    return current
end

"""
    vlasov_sqrt_scarfinder(f0, grid; maxdim, dt, steps_per_iteration,
                         iterations=50, layout=phase_space_layout(size(f0)...),
                         cutoff=0, restore_mass=true, time=0, drive=(x,t)->0.0,
                         nyquist=:keep)

Start from g=sqrt(f0), evolve g with E[g²], truncate g, restore its squared
norm, and repeat. Negative or non-finite initial densities are rejected.
`state` is an MPS for the signed `amplitude`, not for the returned density
`grid`. Positivity holds by squaring; momentum and energy are unconstrained.
Returns an extra unprojected amplitude cycle and its density endpoint.
"""
function vlasov_sqrt_scarfinder(f0::AbstractMatrix{<:Real}, grid::VlasovGrid;
                               maxdim::Integer, dt::Real, steps_per_iteration::Integer,
                               iterations::Integer=50,
                               layout::PhaseSpaceLayout=phase_space_layout(size(f0)...),
                               cutoff::Real=0, restore_mass::Bool=true, time::Real=0,
                               drive=(x,t)->0.0, nyquist::Symbol=:keep)
    _check_vlasov_grid(f0, grid)
    minimum(f0) >= 0 || throw(ArgumentError("initial density must be nonnegative"))
    iterations >= 0 && steps_per_iteration >= 1 ||
        throw(ArgumentError("iterations must be non-negative; steps_per_iteration positive"))
    isfinite(dt) && dt > 0 && isfinite(time) ||
        throw(ArgumentError("dt must be positive and time finite"))
    nyquist in (:keep, :drop) || throw(ArgumentError("nyquist must be :keep or :drop"))
    target = restore_mass ? sum(f0) : nothing
    initial = project_vlasov_amplitude(sqrt.(float.(f0)), layout;
        maxdim=maxdim, cutoff=cutoff, target_sum_squares=target)
    p = initial
    current = initial.amplitude
    history = NamedTuple[]
    tau = dt*steps_per_iteration
    for iteration in 1:iterations
        start_time = time + (iteration-1)*tau
        evolved = evolve_vlasov_amplitude(current, grid; dt=dt, steps=steps_per_iteration,
            time=start_time, drive=drive, nyquist=nyquist)
        p = project_vlasov_amplitude(evolved, layout; maxdim=maxdim, cutoff=cutoff,
                                      target_sum_squares=target)
        push!(history, merge((iteration=iteration, time=start_time+tau,
            step_error=norm(p.grid-current.^2)/max(norm(current.^2),eps(Float64)),
            projection_error=p.density_error, discarded_weight=p.discarded_weight,
            amplitude_scale=p.amplitude_scale,
            evolution_mass_change=sum(abs2,evolved)/max(sum(abs2,current),eps(Float64))-1,
            max_bond=maximum(size(A,2) for A in p.state)), vlasov_diagnostics(p.grid,grid)))
        current = p.amplitude
    end
    final_time = time + iterations*tau
    probe = evolve_vlasov_amplitude(current,grid;dt=dt,steps=steps_per_iteration,
                                     time=final_time,drive=drive,nyquist=nyquist)
    return (grid=p.grid, amplitude=current, state=p.state, layout=layout, history=history,
        initial_projection=initial, time=final_time, tau=tau,
        unprojected_amplitude=probe, unprojected_cycle=probe.^2,
        cycle_error=norm(probe.^2-p.grid)/max(norm(p.grid),eps(Float64)))
end
