@testset "Solver-local MPO scratch and output ownership" begin
    TN=TensorNetworksJulia
    rng=MersenneTwister(392)
    # Unequal bonds, physical dimensions, and intermediate MPO widths;
    # complex, non-Hermitian coefficients expose mistaken conjugations.
    for T in (Float64,ComplexF64)
        random(dims...) = randn(rng,T,dims...)
        FL=random(3,2,3); FR=random(4,3,4)
        W1=random(2,2,2,5); W2=random(3,3,5,3)
        A=random(3,4,2,3)
        action=TN._two_site_action(A,FL,W1,W2,FR)
        saved=copy(A)
        first_result=action(A); saved_result=copy(first_result)
        @test first_result ≈ apply_heff2(A,FL,W1,W2,FR) rtol=1e-12
        other=random(size(A)...)
        @test action(other) ≈ apply_heff2(other,FL,W1,W2,FR) rtol=1e-12
        @test first_result == saved_result
        @test A == saved
        first_result .= 0 # Krylov may modify its returned vectors.
        @test action(A) ≈ saved_result rtol=1e-12
        independent=TN._two_site_action(A,FL,W1,W2,FR)
        @test independent(A) ≈ saved_result rtol=1e-12
        @test !Base.mightalias(action.scratch_left,independent.scratch_left)
        W=random(2,2,2,3); site=random(3,4,2)
        op=TN._one_site_action(site,FL,W,FR)
        @test op(site) ≈ apply_heff1(site,FL,W,FR) rtol=1e-12
        output=op(site); preserved=copy(output)
        op(random(size(site)...))
        @test output == preserved
        W .= 0 # The operator context snapshots fixed coefficients.
        @test op(site) ≈ preserved rtol=1e-12
        FRbond=random(4,2,4); C=random(3,4)
        bond=TN._bond_action(C,FL,FRbond)
        @test bond(C) ≈ apply_keff(C,FL,FRbond) rtol=1e-12
        output=bond(C); preserved=copy(output); bond(2C)
        @test output == preserved
        @test bond(2C) ≈ 2preserved rtol=1e-12
        @test_throws DimensionMismatch op(zeros(T,2,2,2))
    end
    # Sparse model operators, rather than only dense random blocks.
    H=xyz_mpo(4;Jz=-1,hx=-0.7,hy=0.1)
    psi=random_mps(4,2;maxdim=4,rng=rng)
    FL=left_environments(psi,H); FR=right_environments(psi,H)
    A=contract_two_sites(psi[2],psi[3])
    op=TN._two_site_action(A,FL[2],H[2],H[3],FR[4])
    @test op(A) ≈ apply_heff2(A,FL[2],H[2],H[3],FR[4]) rtol=1e-12
    # A zero MPO must yield zero, not stale scratch from a previous call.
    zeroop=TN._two_site_action(A,FL[2],zero(H[2]),H[3],FR[4])
    @test norm(zeroop(A)) == 0
    # Per-call allocations must be near the owned output, not scratch size.
    chi=16; A=randn(rng,ComplexF64,chi,chi,2,2)
    L=randn(rng,ComplexF64,chi,5,chi); R=randn(rng,ComplexF64,chi,5,chi)
    W=xyz_mpo(3;Jz=-1,hx=-1)[2]
    op=TN._two_site_action(A,L,W,W,R)
    op(A); op(A)
    @test (@allocated op(A)) < 100_000
    B=A[:,:,:,1]; siteop=TN._one_site_action(B,L,W,R)
    siteop(B); siteop(B)
    @test (@allocated siteop(B)) < 50_000
    C=Matrix{ComplexF64}(I,chi,chi); bondop=TN._bond_action(C,L,R)
    bondop(C); bondop(C)
    @test (@allocated bondop(C)) < 25_000
end
