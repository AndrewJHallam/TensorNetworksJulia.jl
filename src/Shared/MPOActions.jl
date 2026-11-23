# Solver-local scratch for repeated effective-Hamiltonian applications.
# Each call returns a fresh output: KrylovKit may retain/mutate those vectors.
# Only scratch is reused. A context belongs to one solve and is not reentrant.

function _packed_environments(FL, FR, ::Type{T}) where {T}
    size(FL,2) == size(FR,2) || throw(DimensionMismatch("MPO environment channels differ"))
    left = [Matrix{T}(transpose(view(FL,:,w,:))) for w in axes(FL,2)]
    right = [Matrix{T}(view(FR,:,w,:)) for w in axes(FR,2)]
    return left, right
end

struct _OneSiteAction{T}
    left::Vector{Matrix{T}}
    right::Vector{Matrix{T}}
    weights::Matrix{T}
    scratch_left::Array{T,4}
    scratch_middle::Array{T,4}
    input_size::NTuple{3,Int}
    output_size::NTuple{3,Int}
end

function _one_site_action(A, FL, W, FR)
    size(A) == (size(FL,1),size(FR,1),size(W,2)) || throw(DimensionMismatch("one-site input dimensions differ"))
    size(FL,2) == size(W,3) && size(FR,2) == size(W,4) || throw(DimensionMismatch("one-site MPO channels differ"))
    T = promote_type(eltype(A),eltype(FL),eltype(W),eltype(FR))
    # Here MPO left/right dimensions can differ; pack them separately.
    left = [Matrix{T}(transpose(view(FL,:,w,:))) for w in axes(FL,2)]
    right = [Matrix{T}(view(FR,:,w,:)) for w in axes(FR,2)]
    # Contract the small physical/MPO axes in one GEMM. A BLAS call per
    # nonzero coefficient is slower for dense MPOs and larger bonds.
    weights = Matrix{T}(reshape(permutedims(W,(3,2,1,4)),
                               size(W,3)*size(W,2),size(W,1)*size(W,4)))
    l,r = size(FL,3),size(FR,1)
    return _OneSiteAction(left,right,weights,
        zeros(T,l,r,size(W,3),size(W,2)),zeros(T,l,r,size(W,1),size(W,4)),
        size(A),(size(FL,3),size(FR,3),size(W,1)))
end

function (action::_OneSiteAction{T})(A) where {T}
    size(A) == action.input_size || throw(DimensionMismatch("one-site workspace input differs"))
    L,M = action.scratch_left,action.scratch_middle
    for si in axes(L,4), wl in axes(L,3)
        mul!(view(L,:,:,wl,si),action.left[wl],view(A,:,:,si))
    end
    rows = size(L,1)*size(L,2)
    mul!(reshape(M,rows,:),reshape(L,rows,:),action.weights)
    output = zeros(T,action.output_size)
    for wr in axes(M,4), so in axes(M,3)
        mul!(view(output,:,:,so),view(M,:,:,so,wr),action.right[wr],one(T),one(T))
    end
    return output
end

struct _TwoSiteAction{T}
    left::Vector{Matrix{T}}
    right::Vector{Matrix{T}}
    weights1::Matrix{T}
    weights2::Matrix{T}
    scratch_left::Array{T,5}
    scratch_middle::Array{T,5}
    scratch_packed::Array{T,5}
    scratch_right::Array{T,5}
    input_size::NTuple{4,Int}
    output_size::NTuple{4,Int}
end

function _two_site_action(A,FL,W1,W2,FR)
    size(A) == (size(FL,1),size(FR,1),size(W1,2),size(W2,2)) || throw(DimensionMismatch("two-site input dimensions differ"))
    size(FL,2) == size(W1,3) && size(W1,4) == size(W2,3) && size(W2,4) == size(FR,2) || throw(DimensionMismatch("two-site MPO channels differ"))
    T = promote_type(eltype(A),eltype(FL),eltype(W1),eltype(W2),eltype(FR))
    left = [Matrix{T}(transpose(view(FL,:,w,:))) for w in axes(FL,2)]
    right = [Matrix{T}(view(FR,:,w,:)) for w in axes(FR,2)]
    weights1 = Matrix{T}(reshape(permutedims(W1,(3,2,1,4)),
                                size(W1,3)*size(W1,2),size(W1,1)*size(W1,4)))
    weights2 = Matrix{T}(reshape(permutedims(W2,(3,2,1,4)),
                                size(W2,3)*size(W2,2),size(W2,1)*size(W2,4)))
    l,r = size(FL,3),size(FR,1)
    return _TwoSiteAction(left,right,weights1,weights2,
        zeros(T,l,r,size(W2,2),size(W1,3),size(W1,2)),
        zeros(T,l,r,size(W2,2),size(W1,1),size(W1,4)),
        zeros(T,l,r,size(W1,1),size(W1,4),size(W2,2)),
        zeros(T,l,r,size(W1,1),size(W2,1),size(W2,4)),
        size(A),(l,size(FR,3),size(W1,1),size(W2,1)))
end

function (action::_TwoSiteAction{T})(A) where {T}
    size(A) == action.input_size || throw(DimensionMismatch("two-site workspace input differs"))
    L,M,P,R = action.scratch_left,action.scratch_middle,
              action.scratch_packed,action.scratch_right
    for s1 in axes(A,3), wl in axes(L,4), s2 in axes(A,4)
        mul!(view(L,:,:,s2,wl,s1),action.left[wl],view(A,:,:,s1,s2))
    end
    rows1 = size(L,1)*size(L,2)*size(L,3)
    mul!(reshape(M,rows1,:),reshape(L,rows1,:),action.weights1)
    # Bring the second site's input and MPO channel next to each other.
    permutedims!(P,M,(1,2,4,5,3))
    rows2 = size(P,1)*size(P,2)*size(P,3)
    mul!(reshape(R,rows2,:),reshape(P,rows2,:),action.weights2)
    output = zeros(T,action.output_size)
    for wr in axes(R,5), s2 in axes(R,4), s1 in axes(R,3)
        mul!(view(output,:,:,s1,s2),view(R,:,:,s1,s2,wr),action.right[wr],one(T),one(T))
    end
    return output
end

struct _BondAction{T}
    left::Vector{Matrix{T}}
    right::Vector{Matrix{T}}
    scratch::Matrix{T}
    input_size::Tuple{Int,Int}
    output_size::Tuple{Int,Int}
end

function _bond_action(C,FL,FR)
    size(C) == (size(FL,1),size(FR,1)) || throw(DimensionMismatch("bond input dimensions differ"))
    T = promote_type(eltype(C),eltype(FL),eltype(FR))
    left,right = _packed_environments(FL,FR,T)
    return _BondAction(left,right,zeros(T,size(FL,3),size(FR,1)),
                       size(C),(size(FL,3),size(FR,3)))
end

function (action::_BondAction{T})(C) where {T}
    size(C) == action.input_size || throw(DimensionMismatch("bond workspace input differs"))
    output = zeros(T,action.output_size)
    for w in eachindex(action.left)
        mul!(action.scratch,action.left[w],C)
        mul!(output,action.scratch,action.right[w],one(T),one(T))
    end
    return output
end
