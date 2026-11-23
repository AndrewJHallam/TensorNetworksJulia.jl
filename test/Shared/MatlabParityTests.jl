function matlab_fixture(name; rank=0)
    folder = joinpath(@__DIR__, "..", "fixtures", "matlab")
    shape = parse.(Int, split(strip(read(joinpath(folder, "$name.shape"),String)), ','))
    append!(shape, ones(Int, max(0,rank-length(shape)))) # MATLAB drops trailing singleton axes.
    rows = split(strip(read(joinpath(folder, "$name.csv"), String)), '\n')
    values = map(rows) do row
        parts = parse.(Float64, split(strip(row), ','))
        complex(parts[1],parts[2])
    end
    return reshape(values, Tuple(shape))
end

@testset "Original MATLAB parity fixtures" begin
    psi = [matlab_fixture("psi$n") for n in 1:6]
    H = [matlab_fixture("mpo$n";rank=4) for n in 1:6]
    model = ising_mpo(6; J=-0.83,hx=0.27,hz=-0.61,shift=0.04,interaction=:x)
    @test all(isapprox(a,b; atol=1e-13) for (a,b) in zip(H,model))
    @test expectation(psi,H) ≈ only(matlab_fixture("energy")) atol=1e-11
    Y=ComplexF64[0 -im; im 0]
    # Original measure_o contracts O[input,output]; Julia uses O[output,input].
    @test local_expectations(psi,transpose(Y)) ≈ vec(matlab_fixture("measure_y_transposed")) atol=1e-11
    FL=left_environments(psi,H); FR=right_environments(psi,H)
    @test FL[4] ≈ matlab_fixture("left3") atol=1e-11
    @test FR[3] ≈ matlab_fixture("right3") atol=1e-11
    @test apply_heff1(psi[3],FL[3],H[3],FR[4]) ≈ matlab_fixture("heff3") atol=1e-11
    AA=contract_two_sites(psi[3],psi[4])
    @test apply_heff2(AA,FL[3],H[3],H[4],FR[5]) ≈ matlab_fixture("heff34") atol=1e-11
    evolved=deepcopy(psi)
    tdvp2_step!(evolved,H,0.017; maxdim=8,tol=1e-12,krylovdim=30)
    reference=vec(matlab_fixture("tdvp2_state")); actual=dense_state(evolved)
    @test abs(dot(reference,actual))/(norm(reference)*norm(actual)) ≈ 1 atol=1e-10
    @test norm(actual) ≈ norm(reference) atol=1e-10
    evolved=deepcopy(psi)
    tdvp1_step!(evolved,H,0.017;tol=1e-12,krylovdim=30)
    reference=vec(matlab_fixture("tdvp1_state")); actual=dense_state(evolved)
    @test abs(dot(reference,actual))/(norm(reference)*norm(actual)) ≈ 1 atol=1e-10
    @test norm(actual) ≈ norm(reference) atol=1e-10
    ground=deepcopy(psi); dmrg1_sweep!(ground,H;eig_tol=1e-12)
    reference=vec(matlab_fixture("dmrg1_state")); actual=dense_state(ground)
    @test abs(dot(reference,actual))/(norm(reference)*norm(actual)) ≈ 1 atol=1e-8
    @test expectation(ground,H) ≈ only(matlab_fixture("dmrg1_energy")) atol=1e-8
    ground=deepcopy(psi); dmrg2_sweep!(ground,H; maxdim=8,eig_tol=1e-12)
    reference=vec(matlab_fixture("dmrg2_state")); actual=dense_state(ground)
    @test abs(dot(reference,actual))/(norm(reference)*norm(actual)) ≈ 1 atol=1e-8
    @test expectation(ground,H) ≈ only(matlab_fixture("dmrg2_energy")) atol=1e-8
    state=canonicalize_uniform(matlab_fixture("uniform_input"))
    @test schmidt_values(state.C) ≈ vec(matlab_fixture("uniform_schmidt")) atol=1e-8
    @test reduced_density_matrix(state.AL,state.C) ≈ matlab_fixture("uniform_rho") atol=1e-8
    h=ising_two_site(J=-1,hx=-0.7)
    evolved=infinite_tdvp_step(state.AL,state.AR,state.C,h,0.017)
    @test reduced_density_matrix(evolved.AL,evolved.C) ≈ matlab_fixture("uniform_tdvp_rho") atol=2e-7
    @test schmidt_values(evolved.C) ≈ vec(matlab_fixture("uniform_tdvp_schmidt")) atol=2e-7
    updated=vumps_step(state.AL,state.AR,state.C,h)
    @test reduced_density_matrix(updated.AL,updated.C) ≈ matlab_fixture("vumps_rho") atol=2e-6
    @test schmidt_values(updated.C) ≈ vec(matlab_fixture("vumps_schmidt")) atol=2e-6
    @test uniform_energy(updated.AL,updated.AR,updated.C,h) ≈ only(matlab_fixture("vumps_energy")) atol=2e-6
end
