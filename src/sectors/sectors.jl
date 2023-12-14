# Superselection sectors (quantum numbers):
# for defining graded vector spaces and invariant subspaces of tensor products
#==============================================================================#
"""
    abstract type Sector end

Abstract type for representing the (isomorphism classes of) simple objects in (unitary
and pivotal) (pre-)fusion categories, e.g. the irreducible representations of a finite or
compact group. Subtypes `I<:Sector` as the set of labels of a `GradedSpace`.

Every new `I<:Sector` should implement the following methods:
*   `one(::Type{I})`: unit element of `I`
*   `conj(a::I)`: ``a̅``, conjugate or dual label of ``a``
*   `⊗(a::I, b::I)`: iterable with unique fusion outputs of ``a ⊗ b``
    (i.e. don't repeat in case of multiplicities)
*   `Nsymbol(a::I, b::I, c::I)`: number of times `c` appears in `a ⊗ b`, i.e. the
    multiplicity
*   `FusionStyle(::Type{I})`: `UniqueFusion()`, `SimpleFusion()` or
    `GenericFusion()`
*   `BraidingStyle(::Type{I})`: `Bosonic()`, `Fermionic()`, `Anyonic()`, ...
*   `Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I)`: F-symbol: scalar (in case of
    `UniqueFusion`/`SimpleFusion`) or matrix (in case of `GenericFusion`)
*   `Rsymbol(a::I, b::I, c::I)`: R-symbol: scalar (in case of
    `UniqueFusion`/`SimpleFusion`) or matrix (in case of `GenericFusion`)
and optionally
*   `dim(a::I)`: quantum dimension of sector `a`
*   `frobeniusschur(a::I)`: Frobenius-Schur indicator of `a`
*   `Bsymbol(a::I, b::I, c::I)`: B-symbol: scalar (in case of
    `UniqueFusion`/`SimpleFusion`) or matrix (in case of `GenericFusion`)
*   `twist(a::I)` -> twist of sector `a`
and optionally, if `FusionStyle(I) isa GenericFusion`
*   `vertex_ind2label(i::Int, a::I, b::I, c::I)` -> a custom label for the `i`th copy of
    `c` appearing in `a ⊗ b`

Furthermore, `iterate` and `Base.IteratorSize` should be made to work for the singleton type
[`SectorValues{I}`](@ref).
"""
abstract type Sector end


# iterator over the values (i.e., elements of representative set of simple objects)
# in the sector
"""
    struct SectorValues{I<:Sector}

Singleton type to represent an iterator over the possible values of type `I`, whose
instance is obtained as `values(I)`. For a new `I::Sector`, the following should be defined
*   `Base.iterate(::SectorValues{I}[, state])`: iterate over the values
*   `Base.IteratorSize(::Type{SectorValues{I}})`: `HasLenght()`, `SizeUnkown()`
    or `IsInfinite()` depending on whether the number of values of type `I` is finite
    (and sufficiently small) or infinite; for a large number of values, `SizeUnknown()` is
    recommend because this will trigger the use of `GenericGradedSpace`.
If `IteratorSize(I) == HasLength()`, also the following must be implemented:
*   `Base.length(::SectorValues{I})`: the number of different values
*   `Base.getindex(::SectorValues{I}, i::Int)`: a mapping between an index `i` and an
    instance of `I`
*   `findindex(::SectorValues{I}, c::I)`: reverse mapping between a value `c::I` and an
    index `i::Integer ∈ 1:length(values(I))`
"""
struct SectorValues{I<:Sector} end
Base.IteratorEltype(::Type{<:SectorValues}) = HasEltype()
Base.eltype(::Type{SectorValues{I}}) where {I<:Sector} = I
Base.values(::Type{I}) where {I<:Sector} = SectorValues{I}()


"""
    one(::Sector) -> Sector
    one(::Type{<:Sector}) -> Sector

Return the unit element within this type of sector.
"""
Base.one(a::Sector) = one(typeof(a))

"""
    dual(a::Sector) -> Sector

Return the conjugate label `conj(a)`.
"""
dual(a::Sector) = conj(a)



"""
    isreal(::Type{<:Sector}) -> Bool

Return whether the topological data (Fsymbol, Rsymbol) of the sector is real or not (in
which case it is complex).
"""
function Base.isreal(I::Type{<:Sector})
    u = one(I)
    return (eltype(Fsymbol(u, u, u, u, u, u))<:Real) && (eltype(Rsymbol(u, u, u))<:Real)
end
"""
    eltype(I::Type{<:Sector}) -> Scalar

Return the scalar type of the Fsymbol
"""
function Base.eltype(I::Type{<:Sector})
    u = one(I)
    # return promote_type(eltype(Fsymbol(u, u, u, u, u, u)), eltype(Rsymbol(u, u, u)))
    return eltype(Fsymbol(u, u, u, u, u, u))
end

"""
    Nsymbol(a::I, b::I, c::I) where {I<:Sector} -> Integer

Return an `Integer` representing the number of times `c` appears in the fusion product
`a ⊗ b`. Could be a `Bool` if `FusionStyle(I) == UniqueFusion()` or `SimpleFusion()`.
"""
function Nsymbol end

# trait to describe the fusion of superselection sectors
abstract type FusionStyle end
struct UniqueFusion <: FusionStyle # unique fusion output when fusion two sectors
end
abstract type MultipleFusion <: FusionStyle end
struct SimpleFusion <: MultipleFusion # multiple fusion but multiplicity free
end
struct GenericFusion <: MultipleFusion # multiple fusion with multiplicities
end
const MultiplicityFreeFusion = Union{UniqueFusion, SimpleFusion}

"""
    FusionStyle(a::Sector) -> ::FusionStyle
    FusionStyle(I::Type{<:Sector}) -> ::FusionStyle

Return the type of fusion behavior of sectors of type I, which can be either
*   `UniqueFusion()`: single fusion output when fusing two sectors;
*   `SimpleFusion()`: multiple outputs, but every output occurs at most one,
    also known as multiplicity free (e.g. irreps of ``SU(2)``);
*   `GenericFusion()`: multiple outputs that can occur more than once (e.g. irreps
    of ``SU(3)``).
There is an abstract supertype `MultipleFusion` of which both `SimpleFusion` and
`GenericFusion` are subtypes. Furthermore, there is a type alias `MultiplicityFreeFusion` for those fusion types which do not require muliplicity labels, i.e.
`MultiplicityFreeFusion = Union{UniqueFusion,SimpleFusion}`.
"""
FusionStyle(a::Sector) = FusionStyle(typeof(a))

# combine fusion properties of tensor products of sectors
Base.:&(f::F, ::F) where {F<:FusionStyle} = f
Base.:&(f1::FusionStyle, f2::FusionStyle) = f2 & f1

Base.:&(::SimpleFusion, ::UniqueFusion) = SimpleFusion()
Base.:&(::GenericFusion, ::UniqueFusion) = GenericFusion()
Base.:&(::GenericFusion, ::SimpleFusion) = GenericFusion()


"""
	⊗(a::I, b::I, c::I, rest::Vararg{I}) where {I<:Sector}

Return tensor product of sectors.
"""
⊗(::Type{I}) where {I<:Sector} = (one(I),)
⊗(::Type{I}, a::I, rest::Vararg{I}) where {I<:Sector} = ⊗(a, rest...)
⊗(I::Sector) = (I,)
@inline function ⊗(a::I, b::I, c::I, rest::Vararg{I}) where {I<:Sector}
    if FusionStyle(I) isa UniqueFusion
        return a ⊗ first(⊗(b, c, rest...))
    else
        s = Set{I}()
        for d in ⊗(b, c, rest...)
            for e in a ⊗ d
                push!(s, e)
            end
        end
        return s
    end
end


"""
    Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:Sector}

Return the F-symbol ``F^{abc}_d`` that associates the two different fusion orders of sectors
`a`, `b` and `c` into an ouput sector `d`, using either an intermediate sector ``a ⊗ b → e``
or ``b ⊗ c → f``:
```
a-<-μ-<-e-<-ν-<-d                                     a-<-λ-<-d
    ∨       ∨       -> Fsymbol(a,b,c,d,e,f)[μ,ν,κ,λ]      ∨
    b       c                                             f
                                                          v
                                                      b-<-κ
                                                          ∨
                                                          c
```
If `FusionStyle(I)` is `UniqueFusion` or `SimpleFusion`, the F-symbol is a number. Otherwise
it is a rank 4 array of size
`(Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f), Nsymbol(a, f, d))`.
"""
function Fsymbol end

"""
    Rsymbol(a::I, b::I, c::I) where {I<:Sector}

Returns the R-symbol ``R^{ab}_c`` that maps between ``c → a ⊗ b`` and ``c → b ⊗ a`` as in
```
a -<-μ-<- c                                 b -<-ν-<- c
     ∨          -> Rsymbol(a,b,c)[μ,ν]           v
     b                                           a
```
If `FusionStyle(I)` is `UniqueFusion()` or `SimpleFusion()`, the R-symbol is a
number. Otherwise it is a square matrix with row and column size
`Nsymbol(a,b,c) == Nsymbol(b,a,c)`.
"""
function Rsymbol end

"""
    dim(a::Sector)

Return the (quantum) dimension of the sector `a`.
"""
dim(a::Sector) = error("dim not implemented for sectortype $(typeof(a))")
sqrtdim(a::Sector) = sqrt(dim(a))
isqrtdim(a::Sector) = inv(sqrt(dim(a)))

"""
    frobeniusschur(a::Sector)

Return the Frobenius-Schur indicator of a sector `a`.
"""
frobeniusschur(a::Sector) = sign(Fsymbol(a, conj(a), a, a, one(a), one(a)))

# """
#     twist(a::Sector)

# Return the twist of a sector `a`
# """
# twist(a::Sector) = sum(dim(b)/dim(a)*tr(Rsymbol(a,a,b)) for b in a ⊗ a)


"""
    Bsymbol(a::I, b::I, c::I) where {I<:Sector}

Return the value of ``B^{ab}_c`` which appears in transforming a splitting vertex
into a fusion vertex using the transformation
```
a -<-μ-<- c                                                    a -<-ν-<- c
     ∨          -> √(dim(c)/dim(a)) * Bsymbol(a,b,c)[μ,ν]           ∧
     b                                                            dual(b)
```
If `FusionStyle(I)` is `UniqueFusion()` or `SimpleFusion()`, the B-symbol is a
number. Otherwise it is a square matrix with row and column size
`Nsymbol(a, b, c) == Nsymbol(c, dual(b), a)`.
"""
Bsymbol(a::I, b::I, c::I) where {I<:Sector} = sqrtdim(a)*sqrtdim(b)*isqrtdim(c)*Fsymbol(a, b, dual(b), a, c, one(a))


# SectorSet:
#-------------------------------------------------------------------------------
# Custum generator to represent sets of sectors with type inference
struct SectorSet{I<:Sector, F, S}
    f::F
    set::S
end
SectorSet{I}(::Type{F}, set::S) where {I<:Sector, F, S} = SectorSet{I, Type{F}, S}(F, set)
SectorSet{I}(f::F, set::S) where {I<:Sector, F, S} = SectorSet{I, F, S}(f, set)
SectorSet{I}(set) where {I<:Sector} = SectorSet{I}(identity, set)

Base.IteratorEltype(::Type{<:SectorSet}) = HasEltype()
Base.IteratorSize(::Type{SectorSet{I, F, S}}) where {I<:Sector, F, S} = Base.IteratorSize(S)

Base.eltype(::SectorSet{I}) where {I<:Sector} = I
Base.length(s::SectorSet) = length(s.set)
Base.size(s::SectorSet) = size(s.set)

function Base.iterate(s::SectorSet{I}, args...) where {I<:Sector}
    next = iterate(s.set, args...)
    next === nothing && return nothing
    val, state = next
    return convert(I, s.f(val)), state
end


include("groups.jl")
include("irreps.jl") # irreps of symmetry groups, with bosonic braiding
include("product.jl") # direct product of different sectors


