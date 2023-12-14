# Index manipulations
#---------------------
"""
    permute!(tdst::AbstractTensorMap{S,N₁,N₂}, tsrc::AbstractTensorMap{S},
             (p₁, p₂)::Tuple{IndexTuple{N₁},IndexTuple{N₂}}) where {S,N₁,N₂}
        -> tdst

Write into `tdst` the result of permuting the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
                
See [`permute`](@ref) for creating a new tensor and [`add_permute!`](@ref) for a more general version.
"""
@propagate_inbounds function Base.permute!(tdst::AbstractTensorMap{S,N₁,N₂},
                                           tsrc::AbstractTensorMap{S},
                                           p::Index2Tuple{N₁,N₂}) where {S,N₁,N₂}
    return add_permute!(tdst, tsrc, p, true, false)
end

"""
    permute(tsrc::AbstractTensorMap{S}, (p₁, p₂)::Tuple{IndexTuple{N₁},IndexTuple{N₂}};
            copy::Bool=false) where {S,N₁,N₂}
        -> tdst::TensorMap{S,N₁,N₂}

Return tensor `tdst` obtained by permuting the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To permute into an existing destination, see [permute!](@ref) and [`add_permute!`](@ref)
"""
function permute(t::TensorMap{S}, (p₁, p₂)::Index2Tuple{N₁,N₂};
                 copy::Bool=false) where {S,N₁,N₂}
    cod = ProductSpace{S,N₁}(map(n -> space(t, n), p₁))
    dom = ProductSpace{S,N₂}(map(n -> dual(space(t, n)), p₂))
    # share data if possible
    if (!copy) && (p₁ === codomainind(t) && p₂ === domainind(t)) 
        return t
    end
    # general case
    @inbounds begin
        return permute!(similar(t, cod ← dom), t, (p₁, p₂))
    end
end
function permute(t::AdjointTensorMap{S}, (p₁, p₂)::Index2Tuple;
                 copy::Bool=false) where {S}
    p₁′ = adjointtensorindices(t, p₂)
    p₂′ = adjointtensorindices(t, p₁)
    return adjoint(permute(adjoint(t), (p₁′, p₂′); copy=copy))
end
function permute(t::AbstractTensorMap, p1::IndexTuple, p2::IndexTuple=(); copy::Bool=false)
    return permute(t, (p1, p2); copy=copy)
end

has_shared_permute(t::TensorMap, (p1, p2)::Index2Tuple) = p1 === codomainind(t) && p2 === domainind(t)
function has_shared_permute(t::AdjointTensorMap, (p1, p2)::Index2Tuple)
    p1′ = adjointtensorindices(t, p2)
    p2′ = adjointtensorindices(t, p1)
    return has_shared_permute(t', (p1′, p2′))
end

# Braid
"""
    braid!(tdst::AbstractTensorMap{S,N₁,N₂}, tsrc::AbstractTensorMap{S},
           (p₁, p₂)::Tuple{IndexTuple{N₁},IndexTuple{N₂}}, levels::Tuple) where {S,N₁,N₂}
        -> tdst

Write into `tdst` the result of braiding the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
Here, `levels` is a tuple of length `numind(tsrc)` that assigns a level or height to the indices of `tsrc`,
which determines whether they will braid over or under any other index with which they have to change places.

See [`braid`](@ref) for creating a new tensor and [`add_braid!`](@ref) for a more general version.
"""
@propagate_inbounds function braid!(tdst::AbstractTensorMap{S,N₁,N₂},
                                    tsrc::AbstractTensorMap{S},
                                    (p₁, p₂)::Index2Tuple{N₁,N₂},
                                    levels::IndexTuple) where {S,N₁,N₂}
    return add_braid!(tdst, tsrc, (p₁, p₂), levels, true, false)
end

"""
    braid(tsrc::AbstractTensorMap{S}, (p₁, p₂)::Tuple{IndexTuple{N₁},IndexTuple{N₂}}, levels::Tuple;
          copy::Bool = false) where {S,N₁,N₂}
        -> tdst::TensorMap{S,N₁,N₂}

Return tensor `tdst` obtained by braiding the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
Here, `levels` is a tuple of length `numind(tsrc)` that assigns a level or height to the indices of `tsrc`,
which determines whether they will braid over or under any other index with which they have to change places.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To braid into an existing destination, see [braid!](@ref) and [`add_braid!`](@ref)
"""
function braid(t::TensorMap{S}, (p₁, p₂)::Index2Tuple, levels::IndexTuple;
               copy::Bool=false) where {S}
    @assert length(levels) == numind(t)
    if BraidingStyle(sectortype(S)) isa SymmetricBraiding
        return permute(t, (p₁, p₂); copy=copy)
    end
    if !copy && p₁ == codomainind(t) && p₂ == domainind(t)
        return t
    end
    # general case
    cod = ProductSpace{S}(map(n -> space(t, n), p₁))
    dom = ProductSpace{S}(map(n -> dual(space(t, n)), p₂))
    @inbounds begin
        return braid!(similar(t, cod ← dom), t, (p₁, p₂), levels)
    end
end
# TODO: braid for `AdjointTensorMap`; think about how to map the `levels` argument.

# Transpose
_transpose_indices(t::AbstractTensorMap) = (reverse(domainind(t)), reverse(codomainind(t)))

"""
    transpose!(tdst::AbstractTensorMap{S,N₁,N₂}, tsrc::AbstractTensorMap{S},
               (p₁, p₂)::Tuple{IndexTuple{N₁},IndexTuple{N₂}}) where {S,N₁,N₂}
        -> tdst

Write into `tdst` the result of transposing the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
The new index positions should be attainable without any indices crossing each other, i.e.,
the permutation `(p₁..., reverse(p₂)...)` should constitute a cyclic permutation of `(codomainind(tsrc)..., reverse(domainind(tsrc))...)`.

See [`transpose`](@ref) for creating a new tensor and [`add_transpose!`](@ref) for a more general version.
"""
function LinearAlgebra.transpose!(tdst::AbstractTensorMap,
                                  tsrc::AbstractTensorMap,
                                  (p₁, p₂)::Index2Tuple=_transpose_indices(t))
    return add_transpose!(tdst, tsrc, (p₁, p₂), true, false)
end

"""
    transpose(tsrc::AbstractTensorMap{S}, (p₁, p₂)::Tuple{IndexTuple{N₁},IndexTuple{N₂}};
              copy::Bool=false) where {S,N₁,N₂}
        -> tdst::TensorMap{S,N₁,N₂}

Return tensor `tdst` obtained by transposing the indices of `tsrc`.
The codomain and domain of `tdst` correspond to the indices in `p₁` and `p₂` of `tsrc` respectively.
The new index positions should be attainable without any indices crossing each other, i.e.,
the permutation `(p₁..., reverse(p₂)...)` should constitute a cyclic permutation of `(codomainind(tsrc)..., reverse(domainind(tsrc))...)`.

If `copy=false`, `tdst` might share data with `tsrc` whenever possible. Otherwise, a copy is always made.

To permute into an existing destination, see [permute!](@ref) and [`add_permute!`](@ref)
"""
function LinearAlgebra.transpose(t::TensorMap{S},
                                 (p₁, p₂)::Index2Tuple=_transpose_indices(t);
                                 copy::Bool=false) where {S}
    if !copy && p₁ == codomainind(t) && p₂ == domainind(t)
        return t
    end
    # general case
    cod = ProductSpace{S}(map(n -> space(t, n), p₁))
    dom = ProductSpace{S}(map(n -> dual(space(t, n)), p₂))
    @inbounds begin
        return transpose!(similar(t, cod ← dom), t, (p₁, p₂))
    end
end

function LinearAlgebra.transpose(t::AdjointTensorMap{S},
                                 (p₁, p₂)::Index2Tuple=_transpose_indices(t);
                                 copy::Bool=false) where {S}
    p₁′ = map(n -> adjointtensorindex(t, n), p₂)
    p₂′ = map(n -> adjointtensorindex(t, n), p₁)
    return adjoint(transpose(adjoint(t), (p₁′, p₂′); copy=copy))
end

# Fusing and splitting
# TODO: add functionality for easy fusing and splitting of tensor indices

#-------------------------------------
# Full implementations based on `add`
#-------------------------------------
@propagate_inbounds function add_permute!(tdst::AbstractTensorMap{S,N₁,N₂},
                                          tsrc::AbstractTensorMap,
                                          p::Index2Tuple{N₁,N₂},
                                          α::Number,
                                          β::Number,
                                          backend::Backend...) where {S,N₁,N₂}
    treepermuter(f₁, f₂) = permute(f₁, f₂, p[1], p[2])
    return add_transform!(tdst, tsrc, p, treepermuter, α, β, backend...)
end

@propagate_inbounds function add_braid!(tdst::AbstractTensorMap{S,N₁,N₂},
                                        tsrc::AbstractTensorMap,
                                        p::Index2Tuple{N₁,N₂},
                                        levels::IndexTuple,
                                        α::Number,
                                        β::Number,
                                        backend::Backend...) where {S,N₁,N₂}
    length(levels) == numind(tsrc) ||
        throw(ArgumentError("incorrect levels $levels for tensor map $(codomain(tsrc)) ← $(domain(tsrc))"))

    levels1 = TupleTools.getindices(levels, codomainind(tsrc))
    levels2 = TupleTools.getindices(levels, domainind(tsrc))
    # TODO: arg order for tensormaps is different than for fusiontrees
    treebraider(f₁, f₂) = braid(f₁, f₂, levels1, levels2, p...)
    return add_transform!(tdst, tsrc, p, treebraider, α, β, backend...)
end

@propagate_inbounds function add_transpose!(tdst::AbstractTensorMap{S,N₁,N₂},
                                            tsrc::AbstractTensorMap,
                                            p::Index2Tuple{N₁,N₂},
                                            α::Number,
                                            β::Number,
                                            backend::Backend...) where {S,N₁,N₂}
    treetransposer(f₁, f₂) = transpose(f₁, f₂, p[1], p[2])
    return add_transform!(tdst, tsrc, p, treetransposer, α, β, backend...)
end

function add_transform!(tdst::AbstractTensorMap{S,N₁,N₂},
                        tsrc::AbstractTensorMap,
                        (p₁, p₂)::Index2Tuple{N₁,N₂},
                        fusiontreetransform,
                        α::Number,
                        β::Number,
                        backend::Backend...) where {S,N₁,N₂}
    @boundscheck begin
        all(i -> space(tsrc, p₁[i]) == space(tdst, i), 1:N₁) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(tsrc, p₂[i]) == space(tdst, N₁ + i), 1:N₂) ||
            throw(SpaceMismatch("source = $(codomain(tsrc))←$(domain(tsrc)),
            dest = $(codomain(tdst))←$(domain(tdst)), p₁ = $(p₁), p₂ = $(p₂)"))
    end

    I = sectortype(S)
    if p₁ == codomainind(tsrc) && p₂ == domainind(tsrc)
        axpby!(α, tsrc, β, tdst)
    else
        if FusionStyle(I) isa UniqueFusion
            _add_abelian_kernel!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β, backend...)
        else
            _add_general_kernel!(tdst, tsrc, (p₁, p₂), fusiontreetransform, α, β, backend...)
        end        
    end
    return tdst
end

# internal methods: no argument types
function _add_abelian_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
    if Threads.nthreads() > 1
        Threads.@sync for (f₁, f₂) in fusiontrees(tsrc)
            Threads.@spawn _add_abelian_block!(tdst, tsrc, p, fusiontreetransform,
                                               f₁, f₂, α, β, backend...)
        end
    else
        for (f₁, f₂) in fusiontrees(tsrc)
            _add_abelian_block!(tdst, tsrc, p, fusiontreetransform,
                                f₁, f₂, α, β, backend...)
        end
    end
    return tdst
end

function _add_abelian_block!(tdst, tsrc, p, fusiontreetransform, f₁, f₂, α, β, backend...)
    (f₁′, f₂′), coeff = first(fusiontreetransform(f₁, f₂))
    TO.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, β, backend...)
    return nothing
end

function _add_general_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
    if iszero(β)
        fill!(tdst, β)
    elseif β != 1
        tdst = scale!(tdst, β)
    end

    # TODO: implement multithreading for general symmetries
    # Currently disable multithreading for general symmetries, requires more testing and
    # possibly a different approach. Ideally, we'd loop over output blocks in parallel, to
    # avoid parallel writing, but this requires the inverse of the fusiontreetransform.

    for (f₁, f₂) in fusiontrees(tsrc)
        for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
            TO.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, true,
                          backend...)
        end
    end

    # if Threads.nthreads() > 1
    #     Threads.@sync for s₁ in sectors(codomain(tsrc)), s₂ in sectors(domain(tsrc))
    #         _add_sectors!(tdst, tsrc, fusiontreemap, s₁, s₂, α, β, backend...)
    #     end
    # else
    #     for (f₁, f₂) in fusiontrees(tsrc)
    #         for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
    #             TO.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, true,
    #                           backend...)
    #         end
    #     end
    # end

    return nothing
end

function _add_sectors!(tdst, tsrc, p, fusiontreetransform, s₁, s₂, α, β, backend...)
    for (f₁, f₂) in fusiontrees(tsrc)
        (f₁.outgoing == s₁ && f₂.outgoing == s₂) || continue
        for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
            TO.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, true, backend...)
        end
    end
    return nothing
end
