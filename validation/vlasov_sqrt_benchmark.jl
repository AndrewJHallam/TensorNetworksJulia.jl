using FFTW
include("vlasov_baseline.jl")

# Positive analytic-in-energy BGK distribution:
# F(H)=C(a+H)exp(-H), rho(phi)=C sqrt(2pi)(a+phi+1/2)exp(-phi).
# Solve phi''=1-rho(phi) on its periodic orbit. a=0.1 and phi_max=0.05
# keep a+H strictly positive, including trapped particles at v=0.
function controlled_bgk(n; a=0.1, phi_max=0.05, vmax=8., ode_step=1e-4)
    C = 1/(sqrt(2pi)*(a+0.5))
    rhs(y) = [y[2], 1-C*sqrt(2pi)*(a+y[1]+0.5)*exp(-y[1])]
    y = [phi_max,0.]
    times, phis = [0.], [phi_max]
    for step in 1:100_000
        k1=rhs(y); k2=rhs(y+ode_step/2*k1)
        k3=rhs(y+ode_step/2*k2); k4=rhs(y+ode_step*k3)
        next=y+ode_step/6*(k1+2k2+2k3+k4)
        if y[2]<0 && next[2]>=0
            fraction=-y[2]/(next[2]-y[2])
            push!(times,(step-1+fraction)*ode_step)
            push!(phis,y[1]+fraction*(next[1]-y[1]))
            break
        end
        push!(times,step*ode_step); push!(phis,next[1]); y=next
        step==100_000 && error("BGK turning point not reached")
    end
    half=last(times); grid=VlasovGrid(n,n;Lx=2half,vmax=vmax)
    phi = map(grid.x) do x
        t=min(x,2half-x)
        i=min(searchsortedlast(times,t),length(times)-1)
        fraction=(t-times[i])/(times[i+1]-times[i])
        (1-fraction)*phis[i]+fraction*phis[i+1]
    end
    minimum(phi)>-a || error("BGK distribution would be negative")
    f=[C*(a+v^2/2+phi[i])*exp(-v^2/2-phi[i]) for i in 1:n,v in grid.v]
    return (grid=grid,density=f,potential=phi .- sum(phi)/n)
end

function sqrt_stationary_metrics(f, grid)
    E=vlasov_field(f,grid)
    dxf=real.(ifft(im.*reshape(grid.kx,:,1).*fft(f,1),1))
    dvf=real.(ifft(im.*reshape(grid.kv,1,:).*fft(f,2),2))
    residual=reshape(grid.v,1,:).*dxf+reshape(E,:,1).*dvf
    rho=vec(sum(f;dims=2)).*grid.dv; rh=fft(rho .- sum(rho)/length(rho))
    ph=zeros(ComplexF64,length(rho))
    for i in eachindex(ph)
        grid.kx[i]!=0 && (ph[i]=rh[i]/grid.kx[i]^2)
    end
    potential=real.(ifft(ph))
    transformed=fft(f,1); n=size(f,1)
    nyquist=iseven(n) ? norm(transformed[n÷2+1,:])/sqrt(n)/norm(f) : 0.
    transformed[1,:].=0
    iseven(n) && (transformed[n÷2+1,:].=0)
    return (stationary_residual=norm(residual)/norm(f),
        potential_range=maximum(potential)-minimum(potential),
        resolved_x_variation=norm(transformed)/sqrt(n)/norm(f),x_nyquist_fraction=nyquist)
end

function amplitude_probe(g,grid;dt,steps_per_iteration,cycles=5,nyquist=:keep)
    current=copy(g); f0=g.^2; d0=vlasov_diagnostics(f0,grid); rows=NamedTuple[]
    for step in 0:(cycles*steps_per_iteration)
        step>0 && (current=vlasov_amplitude_step(current,grid,dt;nyquist=nyquist))
        f=current.^2; d=vlasov_diagnostics(f,grid)
        push!(rows,merge((step=step,time=step*dt,relative_change=norm(f-f0)/norm(f0),
            relative_mass_drift=(d.mass-d0.mass)/d0.mass,
            relative_density_l2_drift=norm(f)/norm(f0)-1),d))
    end
    return (grid=current.^2,history=rows,cycle_error=rows[steps_per_iteration+1].relative_change,
        maximum_motion=maximum(r.relative_change for r in rows[1:steps_per_iteration+1]),
        five_cycle_error=last(rows).relative_change)
end

function run_sqrt_case(name,f0,grid;iterations=100,ranks=(4,8,16),dt=0.01,steps=100,
                       output,nyquist=:keep)
    layout=phase_space_layout(size(f0)...)
    base=vlasov_diagnostics(f0,grid); rows=NamedTuple[]
    # Compile each path before recording elapsed time.
    vlasov_sqrt_scarfinder(f0,grid;maxdim=4,dt=dt,steps_per_iteration=1,iterations=1)
    vlasov_scarfinder(f0,grid;maxdim=4,dt=dt,steps_per_iteration=1,iterations=1)
    states=Dict{String,Matrix{Float64}}()
    for representation in (:density,:sqrt), rank in ranks
        nyquist==:drop && representation==:density && continue
        elapsed=@elapsed result = representation==:sqrt ?
            vlasov_sqrt_scarfinder(f0,grid;maxdim=rank,dt=dt,steps_per_iteration=steps,
                iterations=iterations,nyquist=nyquist) :
            vlasov_scarfinder(f0,grid;maxdim=rank,dt=dt,steps_per_iteration=steps,iterations=iterations)
        label="$(representation)_chi$(rank)_$(nyquist)"
        path=joinpath(output,name,label)
        probe=representation==:sqrt ? amplitude_probe(result.amplitude,grid;dt=dt,
            steps_per_iteration=steps,nyquist=nyquist) :
            unprojected_probe(result.grid,grid;dt=dt,steps_per_iteration=steps)
        # Independent density evolution from every final field, even for g runs.
        density_check=evolve_vlasov(result.grid,grid;dt=dt,steps=steps)
        d=vlasov_diagnostics(result.grid,grid)
        push!(rows,merge((case=name,representation=string(representation),rank=rank,nyquist=string(nyquist),
            stored_parameters=sum(length,result.state),elapsed_seconds=elapsed,
            initial_projection_error=norm(result.initial_projection.grid-f0)/norm(f0),
            final_step_error=last(result.history).step_error,
            relative_initial_difference=norm(result.grid-f0)/norm(f0),
            cycle_error=probe.cycle_error,maximum_motion=probe.maximum_motion,
            five_cycle_error=probe.five_cycle_error,
            independent_density_cycle_error=norm(density_check-result.grid)/norm(result.grid),
            negative_mass_fraction=d.negative_mass/d.mass,
            relative_mass_error=(d.mass-base.mass)/base.mass,
            relative_energy_change=(d.energy-base.energy)/base.energy,
            relative_density_l2_change=norm(result.grid)/norm(f0)-1),sqrt_stationary_metrics(result.grid,grid)))
        write_rows(joinpath(path,"iterations.csv"),result.history)
        write_rows(joinpath(path,"probe.csv"),probe.history)
        write_rows(joinpath(path,"final.csv"),[(x=grid.x[i],v=grid.v[j],f=result.grid[i,j])
            for j in eachindex(grid.v) for i in eachindex(grid.x)])
        states[label]=result.grid
        println(name," ",label,": negative mass=",d.negative_mass/d.mass,
            ", residual=",last(rows).stationary_residual,", density release=",last(rows).independent_density_cycle_error)
    end
    write_rows(joinpath(output,name,"summary_$(nyquist).csv"),rows)
    return (summary=rows,states=states)
end

function benchmark_sqrt(;output=joinpath(@__DIR__,"results","vlasov_sqrt"))
    allrows=NamedTuple[]
    g=VlasovGrid(128,128)
    f0=[(1+0.4cos(x))*(exp(-0.5*((v-2.4)/0.75)^2)+exp(-0.5*((v+2.4)/0.75)^2)) /
        (2sqrt(2pi)*0.75) for x in g.x,v in g.v]
    twostream=run_sqrt_case("two_stream",f0,g;output=output)
    append!(allrows,twostream.summary)
    bgk=controlled_bgk(128)
    write_rows(joinpath(output,"controlled_bgk","construction.csv"),[(Lx=128bgk.grid.dx,
        potential_range=maximum(bgk.potential)-minimum(bgk.potential),
        poisson_potential_error=begin
            rho=vec(sum(bgk.density;dims=2)).*bgk.grid.dv
            ph=fft(rho); ph[1]=0; ph[65]=0
            for i in eachindex(ph); bgk.grid.kx[i]!=0 && (ph[i]/=bgk.grid.kx[i]^2); end
            norm(real.(ifft(ph))-bgk.potential)/norm(bgk.potential)
        end,stationary_residual=sqrt_stationary_metrics(bgk.density,bgk.grid).stationary_residual)])
    equilibrium=run_sqrt_case("controlled_bgk",bgk.density,bgk.grid;output=output,iterations=20)
    append!(allrows,equilibrium.summary)
    # The saved candidate is retained as an approximate candidate, not labelled exact BGK.
    candidate_path=joinpath(output,"matlab_candidate.csv")
    if isfile(candidate_path)
        f=read_numeric_matrix(candidate_path)
        p=vec(read_numeric_matrix(joinpath(output,"matlab_candidate_parameters.csv")))
        cg=VlasovGrid(size(f)...;Lx=p[2],vmax=p[3])
        candidate=run_sqrt_case("matlab_candidate",f,cg;output=output,iterations=20,
                                dt=p[4],steps=round(Int,p[5]))
        append!(allrows,candidate.summary)
    end
    # Isolate the frozen amplitude mode with the same g rank and PDE interval.
    drop=run_sqrt_case("two_stream",f0,g;output=output,nyquist=:drop,ranks=(4,))
    append!(allrows,drop.summary)
    write_rows(joinpath(output,"summary.csv"),allrows)
    return allrows
end

if abspath(PROGRAM_FILE)==@__FILE__
    benchmark_sqrt()
end
