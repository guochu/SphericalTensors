"""
    struct GradedSpace{I<:Sector, D} <: ElementarySpace
        dims::D
        dual::Bool
    end

A complex Euclidean space with a direct sum structure corresponding to labels in a set `I`,
the objects of which have the structure of a monoid with respect to a monoidal product `⊗`.
In practice, we restrict the label set to be a set of superselection sectors of type
`I<:Sector`, e.g. the set of distinct irreps of a finite or compact group, or the
isomorphism classes of simple objects of a unitary and pivotal (pre-)fusion category.

Here `dims` represents the degeneracy or multiplicity of every sector.

The data structure `D` of `dims` will depend on the result `Base.IteratorElsize(values(I))`;
if the result is of type `HasLength` or `HasShape`, `dims` will be stored in a
`NTuple{N,Int}` with `N = length(values(I))`. This requires that a sector `s::I` can be
transformed into an index via `s == getindex(values(I), i)` and
`i == findindex(values(I), s)`. If `Base.IteratorElsize(values(I))` results `IsInfinite()`
or `SizeUnknown()`, a `SectorDict{I,Int}` is used to store the non-zero degeneracy
dimensions with the corresponding sector as key. The parameter `D` is hidden from the user
and should typically be of no concern.

The concrete type `GradedSpace{I,D}` with correct `D` can be obtained as `Vect[I]`, or if
`I == Irrep[G]` for some `G<:Group`, as `Rep[G]`.
"""
struct GradedSpace{I<:Sector} <: ElementarySpace
    dims::SectorDict{I, Int}
    dual::Bool
    function GradedSpace{I}(dims; dual::Bool=false) where {I}
        d = SectorDict{I, Int}()
        for (c, dc) in dims
            k = convert(I, c)
            haskey(d, c) && throw(ArgumentError("Sector $k appears multiple times"))
            !iszero(dc) && push!(d, k=>dc)
        end
        new{I}(d, dual)
    end
end
sectortype(::Type{<:GradedSpace{I}}) where {I<:Sector} = I

GradedSpace{I}(; kwargs...) where I = GradedSpace{I}((); kwargs...)
GradedSpace{I}(d1::Pair, dims::Vararg{Pair}; dual::Bool=false) where I = GradedSpace{I}((d1, dims...); dual=dual)

GradedSpace(dims::Tuple{Vararg{Pair{I, <:Integer}}}; dual::Bool=false) where {I<:Sector} = GradedSpace{I}(dims, dual=dual)
GradedSpace(dims::Vararg{Pair{I, <:Integer}}; dual::Bool=false) where {I<:Sector} = GradedSpace{I}(dims; dual=dual)
GradedSpace(dims::AbstractDict{I, <:Integer}; dual::Bool=false) where {I<:Sector} = GradedSpace{I}(dims; dual=dual)

GradedSpace(g::Base.Generator; dual::Bool=false) = GradedSpace(g...; dual=dual)
GradedSpace(g::AbstractDict; dual::Bool=false) = GradedSpace(g...; dual=dual)

Base.hash(V::GradedSpace, h::UInt) = hash(V.dual, hash(V.dims, h))

# Corresponding methods:
# properties
dim(V::GradedSpace) = reduce(+, dim(V, c) * dim(c) for c in sectors(V); init = zero(dim(one(sectortype(V)))))
dim(V::GradedSpace{I}, c::I) where {I<:Sector} = get(V.dims, isdual(V) ? dual(c) : c, 0)

sectors(V::GradedSpace{I}) where {I<:Sector} = SectorSet{I}(s->isdual(V) ? dual(s) : s, keys(V.dims))

hassector(V::GradedSpace{I}, s::I) where {I<:Sector} = dim(V, s) != 0

Base.conj(V::GradedSpace) = GradedSpace(V.dims, dual=!V.dual)
isdual(V::GradedSpace) = V.dual

# equality / comparison
Base.:(==)(V1::GradedSpace, V2::GradedSpace) =
    sectortype(V1) == sectortype(V2) && (V1.dims == V2.dims) && V1.dual == V2.dual

# axes
Base.axes(V::GradedSpace) = Base.OneTo(dim(V))
function Base.axes(V::GradedSpace{I}, c::I) where {I<:Sector}
    offset = 0
    for c′ in sectors(V)
        c′ == c && break
        offset += dim(c′)*dim(V, c′)
    end
    return (offset+1):(offset+dim(c)*dim(V, c))
end

Base.oneunit(S::Type{<:GradedSpace{I}}) where {I<:Sector} = S(one(I)=>1)

# TODO: the following methods can probably be implemented more efficiently for
# `FiniteGradedSpace`, but we don't expect them to be used often in hot loops, so
# these generic definitions (which are still quite efficient) are good for now.
function ⊕(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    dual1 = isdual(V1)
    dual1 == isdual(V2) ||
        throw(SpaceMismatch("Direct sum of a vector space and a dual space does not exist"))
    dims = SectorDict{I, Int}()
    for c in union(sectors(V1), sectors(V2))
        cout = ifelse(dual1, dual(c), c)
        dims[cout] = dim(V1, c) + dim(V2, c)
    end
    return GradedSpace{I}(dims; dual = dual1)
end

function flip(V::GradedSpace{I}) where {I<:Sector}
    if isdual(V)
        GradedSpace{I}(c=>dim(V, c) for c in sectors(V))
    else
        GradedSpace{I}(dual(c)=>dim(V, c) for c in sectors(V))'
    end
end

function fuse(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    dims = SectorDict{I, Int}()
    for a in sectors(V1), b in sectors(V2)
        for c in a ⊗ b
            dims[c] = get(dims, c, 0) + Nsymbol(a, b, c)*dim(V1, a)*dim(V2, b)
        end
    end
    return GradedSpace{I}(dims)
end

function infimum(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    if V1.dual == V2.dual
        GradedSpace{I}(c=>min(dim(V1, c), dim(V2, c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    end
end

function supremum(V1::GradedSpace{I}, V2::GradedSpace{I}) where {I<:Sector}
    if V1.dual == V2.dual
        GradedSpace{I}(c=>max(dim(V1, c), dim(V2, c)) for c in
            union(sectors(V1), sectors(V2)), dual = V1.dual)
    else
        throw(SpaceMismatch("Supremum of space and dual space does not exist"))
    end
end

function Base.show(io::IO, V::GradedSpace{I}) where {I<:Sector}
    print(io, type_repr(typeof(V)), "(")
    seperator = ""
    comma = ", "
    io2 = IOContext(io, :typeinfo => I)
    for c in sectors(V)
        if isdual(V)
            print(io2, seperator, dual(c), "=>", dim(V, c))
        else
            print(io2, seperator, c, "=>", dim(V, c))
        end
        seperator = comma
    end
    print(io, ")")
    V.dual && print(io, "'")
    return nothing
end

struct SpaceTable end
"""
    const Vect

A constant of a singleton type used as `Vect[I]` with `I<:Sector` a type of sector, to
construct or obtain the concrete type `GradedSpace{I,D}` instances without having to
specify `D`.
"""
const Vect = SpaceTable()
Base.getindex(::SpaceTable, I::Type{<:Sector}) = GradedSpace{I}

struct RepTable end
"""
    const Rep

A constant of a singleton type used as `Rep[G]` with `G<:Group` a type of group, to
construct or obtain the concrete type `GradedSpace{Irrep[G],D}` instances without having to
specify `D`. Note that `Rep[G] == Vect[Irrep[G]]`.

See also [`Irrep`](@ref) and [`Vect`](@ref).
"""
const Rep = RepTable()
Base.getindex(::RepTable, G::Type{<:Group}) = Vect[Irrep[G]]

type_repr(::Type{<:GradedSpace{<:AbstractIrrep{G}}}) where {G<:Group} =
    "Rep[" * type_repr(G) * "]"
function type_repr(::Type{<:GradedSpace{ProductSector{T}}}) where
                                                        {T<:Tuple{Vararg{AbstractIrrep}}}
    sectors = T.parameters
    s = "Rep["
    for i in 1:length(sectors)
        if i != 1
            s *= " × "
        end
        s *= type_repr(supertype(sectors[i]).parameters[1])
    end
    s *= "]"
    return s
end

# TODO: Do we still need all of those
# ASCII type aliases
const ZNSpace{N} = GradedSpace{ZNIrrep{N}}
const Z2Space = ZNSpace{2}
const U1Space = Rep[U₁]
const SU2Space = Rep[SU₂]

# Unicode alternatives
const ℤ₂Space = Z2Space
const U₁Space = U1Space
const SU₂Space = SU2Space






