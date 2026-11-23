@testset "Blocked periodic unit cells" begin
    W1=ising_mpo(3;J=-0.8,hx=-1.1,hz=0.2,interaction=:z)[2]
    W2=ising_mpo(3;J=-1.2,hx=-0.7,hz=-0.2,interaction=:z)[2]
    blocked=block_mpo([W1,W2])
    # Independent four-site dense Hamiltonian; exposes swapped cell order
    # and omission of interactions across cells.
    X=ComplexF64[0 1;1 0]; Z=ComplexF64[1 0;0 -1]; I2=Matrix{ComplexF64}(I,2,2)
    exact=zeros(ComplexF64,16,16)
    for n in 1:4
        operators=[copy(I2) for _ in 1:4]
        operators[n]=(isodd(n) ? -1.1 : -0.7)*X+(isodd(n) ? 0.2 : -0.2)*Z
        exact .+= reduce(kron,operators)
    end
    for n in 1:3
        operators=[copy(I2) for _ in 1:4]; operators[n]=Z; operators[n+1]=Z
        exact .+= (isodd(n) ? -0.8 : -1.2)*reduce(kron,operators)
    end
    @test mpo_matrix(finite_mpo(blocked,2)) ≈ exact
    @test block_mpo([TriangularMPO(W1),TriangularMPO(W2)]).W ≈ blocked.W
    @test block_mpo([W1]).W ≈ W1
    rng=MersenneTwister(830)
    A=randn(rng,ComplexF64,2,3,2); B=randn(rng,ComplexF64,3,2,3)
    grouped=block_mps([A,B])
    @test size(grouped)==(2,2,6)
    for s in 1:2,t in 1:3
        @test grouped[:,:,(s-1)*3+t] ≈ A[:,:,s]*B[:,:,t]
    end
    # Product state and exactly soluble staggered onsite Hamiltonian.
    plus=ComplexF64[1,1]/sqrt(2); down=ComplexF64[0,1]
    product=block_mps([reshape(plus,1,1,2),reshape(down,1,1,2)])
    @test vec(product) ≈ kron(plus,down)
    state=canonicalize_uniform(product)
    H=block_mpo([ising_mpo(3;J=0,hx=-1,interaction=:z)[2],
                 ising_mpo(3;J=0,hz=0.5,interaction=:z)[2]])
    @test uniform_mpo_energy(state.AL,state.AR,state.C,H) ≈ -1.5
    ground=vumps(H,1;d=4,rng=MersenneTwister(831),maxiter=30,tol=1e-9)
    @test ground.converged
    @test ground.energy ≈ -1.5 atol=1e-9
    evolved=infinite_tdvp_step(ground.AL,ground.AR,ground.C,H,0.02)
    @test reduced_density_matrix(evolved.AL,evolved.C) ≈ reduced_density_matrix(ground.AL,ground.C) atol=1e-9
    @test_throws ArgumentError block_mps([])
    @test_throws ArgumentError block_mpo([])
    @test_throws DimensionMismatch block_mps([A])
    @test_throws ArgumentError block_mps([A,B];max_physicaldim=4)
    @test_throws ArgumentError block_mpo([W1,W2];max_physicaldim=2)
end
