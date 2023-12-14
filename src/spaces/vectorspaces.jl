# VECTOR SPACES:
#==============================================================================#
"""
    abstract type VectorSpace end

Abstract type at the top of the type hierarchy for denoting vector spaces, or, more
accurately, 𝕜-linear categories. All instances of subtypes of VectorSpace will
represent objects in 𝕜-linear monoidal categories.
"""
abstract type VectorSpace end

# Basic vector space methods
#----------------------------
"""
    space(a) -> VectorSpace

Return the vector space associated to object `a`.
"""
function space end

"""
    dim(V::VectorSpace) -> Int

Return the total dimension of the vector space `V` as an Int.
"""
function dim end

"""
    dual(V::VectorSpace) -> VectorSpace

Return the dual space of `V`; also obtained via `V'`. This should satisfy
`dual(dual(V)) == V`. It is assumed that `typeof(V) == typeof(V')`.
"""
function dual end

# convenience definitions:
Base.adjoint(V::VectorSpace) = dual(V)

"""
    isdual(V::ElementarySpace) -> Bool

Return wether an ElementarySpace `V` is normal or rather a dual space. Always returns
`false` for spaces where `V == dual(V)`.
"""
function isdual end

# Hierarchy of elementary vector spaces
#---------------------------------------
"""
    abstract type ElementarySpace{𝕜} <: InnerProductSpace{𝕜} end

Abstract type for denoting real or complex spaces with a standard Euclidean inner product
(i.e. orthonormal basis, and the metric is identity), such that the dual space is naturally
isomorphic to the conjugate space `dual(V) == conj(V)` (in the complex case) or even to
the space itself `dual(V) == V` (in the real case), also known as the category of
finite-dimensional Hilbert spaces ``FdHilb``. In the language of categories, this subtype
represents dagger or unitary categories, and support an adjoint operation.
"""
abstract type ElementarySpace <: VectorSpace end
const IndexSpace = ElementarySpace

"""
    oneunit(V::S) where {S<:ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the trivial
one-dimensional space, i.e. the space that is isomorphic to the corresponding field. Note
that this is different from `one(V::S)`, which returns the empty product space
`ProductSpace{S,0}(())`.
"""
Base.oneunit(V::ElementarySpace) = oneunit(typeof(V))


"""
    ⊕(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the direct sum sum of the
spaces `V1`, `V2`, ... Note that all the individual spaces should have the same value for
[`isdual`](@ref), as otherwise the direct sum is not defined.
"""
function ⊕ end
⊕(V::VectorSpace) = V
⊕(V1, V2, V3, V4...) = ⊕(⊕(V1, V2), V3, V4...)

"""
    ⊗(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S

Create a [`ProductSpace{S}(V1, V2, V3...)`](@ref) representing the tensor product of several
elementary vector spaces. For convience, Julia's regular multiplication operator `*` applied
to vector spaces has the same effect.

The tensor product structure is preserved, see [`fuse`](@ref) for returning a single
elementary space of type `S` that is isomorphic to this tensor product.
"""
function ⊗ end
⊗(V1, V2, V3, V4...) = ⊗(⊗(V1, V2), V3, V4...)

# convenience definitions:
Base.:*(V1::VectorSpace, V2::VectorSpace) = ⊗(V1, V2)

"""
    fuse(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S
    fuse(P::ProductSpace{S}) where {S<:ElementarySpace} -> S

Return a single vector space of type `S` that is isomorphic to the fusion product of the
individual spaces `V1`, `V2`, ..., or the spaces contained in `P`.
"""
function fuse end
fuse(V::ElementarySpace) = V
fuse(V1::VectorSpace, V2::VectorSpace, V3::VectorSpace...) =
    fuse(fuse(fuse(V1), fuse(V2)), V3...)
    # calling fuse on V1 and V2 will allow these to be `ProductSpace`

"""
    flip(V::S) where {S<:ElementarySpace} -> S

Return a single vector space of type `S` that has the same value of [`isdual`](@ref) as
`dual(V)`, but yet is isomorphic to `V` rather than to `dual(V)`. The spaces `flip(V)` and
`dual(V)` only differ in the case of [`GradedSpace{I}`](@ref).
"""
function flip end

"""
    conj(V::S) where {S<:ElementarySpace} -> S

Return the conjugate space of `V`. This should satisfy `conj(conj(V)) == V`.

For `field(V)==ℝ`, `conj(V) == V`. It is assumed that `typeof(V) == typeof(conj(V))`.
"""
Base.conj(V::ElementarySpace) = error("conj not implemented for spacetype $(typeof(V))")

dual(V::ElementarySpace) = conj(V)
isdual(V::ElementarySpace) = error("isdual not implemented for spacetype $(typeof(V))")

"""
    sectortype(a) -> Type{<:Sector}

Return the type of sector over which object `a` (e.g. a representation space or a tensor) is
defined. Also works in type domain.
"""
sectortype(V::VectorSpace) = sectortype(typeof(V))

"""
    hassector(V::VectorSpace, a::Sector) -> Bool

Return whether a vector space `V` has a subspace corresponding to sector `a` with non-zero
dimension, i.e. `dim(V, a) > 0`.
"""
hassector(V::ElementarySpace, s) = error("hassector not implemented for spacetype $(typeof(V))")
Base.axes(V::ElementarySpace, s) = error("axes not implemented for spacetype $(typeof(V))")


"""
    sectors(V::ElementarySpace)

Return an iterator over the different sectors of `V`.
"""
sectors(V::ElementarySpace) = error("sectors not implemented for spacetype $(typeof(V))")
dim(V::ElementarySpace, s) = error("dim not implemented for spacetype $(typeof(V))")

spacetype(S::Type{<:ElementarySpace}) = S
spacetype(V::ElementarySpace) = typeof(V) # = spacetype(typeof(V))

# make ElementarySpace instances behave similar to ProductSpace instances
blocksectors(V::ElementarySpace) = sectors(V)
blockdim(V::ElementarySpace, c::Sector) = dim(V, c)

# space with internal structure corresponding to the irreducible representations of
# a group, or more generally, the simple objects of a fusion category.
include("gradedspace.jl")

# Specific realizations of CompositeSpace types
#-----------------------------------------------
# a tensor product of N elementary spaces of the same type S
include("productspace.jl")
# deligne tensor product
include("deligne.jl")

# Other examples might include:
# symmetric and antisymmetric subspace of a tensor product of identical vector spaces
# ...

# HomSpace: space of morphisms
#------------------------------
include("homspace.jl")


# Partial order for vector spaces
#---------------------------------
"""
    isisomorphic(V1::VectorSpace, V2::VectorSpace)
    V1 ≅ V2

Return if `V1` and `V2` are isomorphic, meaning that there exists isomorphisms from `V1` to
`V2`, i.e. morphisms with left and right inverses.
"""
function isisomorphic(V1::VectorSpace, V2::VectorSpace)
    spacetype(V1) == spacetype(V2) || return false
    for c in union(blocksectors(V1), blocksectors(V2))
        if blockdim(V1, c) != blockdim(V2, c)
            return false
        end
    end
    return true
end

"""
    ismonomorphic(V1::VectorSpace, V2::VectorSpace)
    V1 ≾ V2

Return whether there exist monomorphisms from `V1` to `V2`, i.e. 'injective' morphisms with
left inverses.
"""
function ismonomorphic(V1::VectorSpace, V2::VectorSpace)
    spacetype(V1) == spacetype(V2) || return false
    for c in blocksectors(V1)
        if blockdim(V1, c) > blockdim(V2, c)
            return false
        end
    end
    return true
end

"""
    isepimorphic(V1::VectorSpace, V2::VectorSpace)
    V1 ≿ V2

Return whether there exist epimorphisms from `V1` to `V2`, i.e. 'surjective' morphisms with
right inverses.
"""
function isepimorphic(V1::VectorSpace, V2::VectorSpace)
    spacetype(V1) == spacetype(V2) || return false
    for c in blocksectors(V2)
        if blockdim(V1, c) < blockdim(V2, c)
            return false
        end
    end
    return true
end

# unicode alternatives
const ≅ = isisomorphic
const ≾ = ismonomorphic
const ≿ = isepimorphic

≺(V1::VectorSpace, V2::VectorSpace) = V1 ≾ V2 && !(V1 ≿ V2)
≻(V1::VectorSpace, V2::VectorSpace) = V1 ≿ V2 && !(V1 ≾ V2)

"""
    infimum(V1::ElementarySpace, V2::ElementarySpace, V3::ElementarySpace...)

Return the infimum of a number of elementary spaces, i.e. an instance `V::ElementarySpace`
such that `V ≾ V1`, `V ≾ V2`, ... and no other `W ≻ V` has this property. This requires
that all arguments have the same value of `isdual( )`, and also the return value `V` will
have the same value.
"""
infimum(V1::ElementarySpace, V2::ElementarySpace, V3::ElementarySpace...) =
    infimum(infimum(V1, V2), V3...)

"""
    supremum(V1::ElementarySpace, V2::ElementarySpace, V3::ElementarySpace...)

Return the supremum of a number of elementary spaces, i.e. an instance `V::ElementarySpace`
such that `V ≿ V1`, `V ≿ V2`, ... and no other `W ≺ V` has this property. This requires
that all arguments have the same value of `isdual( )`, and also the return value `V` will
have the same value.
"""
supremum(V1::ElementarySpace, V2::ElementarySpace, V3::ElementarySpace...) =
    supremum(supremum(V1, V2), V3...)



