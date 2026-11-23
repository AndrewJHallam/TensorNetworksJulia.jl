@testset "Environment contraction allocation regression" begin
    # Meaningful bound: a chi^4 intermediate exceeds this by an order of
    # magnitude. Warm up before measuring, with ample room for backend changes.
    rng=MersenneTwister(258)
    chi=16
    A=randn(rng,ComplexF64,chi,chi,2)
    W=xyz_mpo(3;Jz=-1,hx=-1.05)[2]
    FR=randn(rng,ComplexF64,chi,5,chi)
    expected=zeros(ComplexF64,chi,5,chi)
    # Independent block-matrix contraction, no TensorOperations macros.
    for wl in 1:5,wr in 1:5,so in 1:2,si in 1:2
        expected[:,wl,:] .+= W[so,si,wl,wr] .* (A[:,:,si]*FR[:,wr,:]*A[:,:,so]')
    end
    @test right_environment(A,W,FR) ≈ expected rtol=1e-12
    right_environment(A,W,FR); right_environment(A,W,FR)
    @test (@allocated right_environment(A,W,FR)) < 5_000_000
    R=randn(rng,ComplexF64,chi,chi)
    expected_norm=sum(A[:,:,s]*R*A[:,:,s]' for s in 1:2)
    action=TensorNetworksJulia._finite_norm_right_environment
    @test action(A,R) ≈ expected_norm rtol=1e-12
    action(A,R); action(A,R)
    @test (@allocated action(A,R)) < 500_000
    # A physical block must not leave two open virtual bonds when forming rho.
    psi=random_mps(16,2;maxdim=16,rng=rng)
    rho=reduced_density_matrix(psi,5:8)
    @test tr(rho) ≈ 1
    reduced_density_matrix(psi,5:8); reduced_density_matrix(psi,5:8)
    @test (@allocated reduced_density_matrix(psi,5:8)) < 5_000_000
end
