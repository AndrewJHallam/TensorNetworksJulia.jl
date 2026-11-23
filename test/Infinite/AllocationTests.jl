@testset "Infinite contraction allocation regression" begin
    TN=TensorNetworksJulia; rng=MersenneTwister(103); chi=16
    A=randn(rng,ComplexF64,chi,chi,2)
    X=randn(rng,ComplexF64,chi,chi)
    O=randn(rng,ComplexF64,2,2)
    reference=sum(O[so,si]*(A[:,:,si]*X*A[:,:,so]') for si in 1:2,so in 1:2)
    @test TN._mpo_right_block(A,O,X) ≈ reference rtol=1e-12
    TN._mpo_right_block(A,O,X); TN._mpo_right_block(A,O,X)
    @test (@allocated TN._mpo_right_block(A,O,X)) < 500_000
    h=ising_two_site(J=-1,hx=-0.7,hy=0.2)
    reference=zeros(ComplexF64,chi,chi)
    for si in 1:2,ti in 1:2,so in 1:2,to in 1:2
        reference .+= h[so,to,si,ti].*((A[:,:,si]*A[:,:,ti])*(A[:,:,so]*A[:,:,to])')
    end
    @test TN._right_hamiltonian_source(A,h) ≈ reference rtol=1e-12
    TN._right_hamiltonian_source(A,h); TN._right_hamiltonian_source(A,h)
    @test (@allocated TN._right_hamiltonian_source(A,h)) < 1_000_000
end
