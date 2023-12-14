# Fusion trees:
#==============================================================================#
"""
    struct FusionTree{I, N, M}

Represents a fusion tree of sectors of type `I<:Sector`, fusing (or splitting) `N` uncoupled
sectors to a coupled sector. (It actually represents a splitting tree, but fusion tree
is a more common term). The `isdual` field indicates whether an isomorphism is present
(if the corresponding value is true) or not. The field `uncoupled` contains the sectors
coming out of the splitting trees, before the possible 𝑍 isomorphism. This fusion tree
has `M=max(0, N-2)` inner lines. Furthermore, for `FusionStyle(I) isa GenericFusion`,
the `L=max(0, N-1)` corresponding vertices carry a label of type `T`. If `FusionStyle(I)
isa MultiplicityFreeFusion, `T = Nothing`.
"""
struct FusionTree{I<:Sector, N, M}
    uncoupled::NTuple{N, I}
    coupled::I
    isdual::NTuple{N, Bool}
    innerlines::NTuple{M, I} # M = N-2
end

function FusionTree{I}(uncoupled::NTuple{N, Any},
                        coupled,
                        isdual::NTuple{N, Bool},
                        innerlines,
                        ) where {I<:Sector, N}
        fusiontreetype(I, N)(map(s->convert(I, s), uncoupled),
                convert(I, coupled), isdual, map(s->convert(I, s), innerlines))
end

function FusionTree(uncoupled::NTuple{N, I}, coupled::I, isdual::NTuple{N, Bool}) where {I<:Sector, N}
    (FusionStyle(I) isa UniqueFusion) || error("fusion tree requires inner lines if `FusionStyle(I) <: MultipleFusion`")
    FusionTree{I}(uncoupled, coupled, isdual, _abelianinner((uncoupled..., dual(coupled))))
end

# Properties
sectortype(::Type{<:FusionTree{I}}) where {I<:Sector} = I
FusionStyle(::Type{<:FusionTree{I}}) where {I<:Sector} = FusionStyle(I)
Base.length(::Type{<:FusionTree{<:Sector, N}}) where {N} = N
sectortype(f::FusionTree) = sectortype(typeof(f))
FusionStyle(f::FusionTree) = FusionStyle(typeof(f))
Base.length(f::FusionTree) = length(typeof(f))

# Hashing, important for using fusion trees as key in a dictionary
function Base.hash(f::FusionTree{I}, h::UInt) where {I}
    h = hash(f.isdual, hash(f.coupled, hash(f.uncoupled, h)))
    h = hash(f.innerlines, h)
    return h
end
function Base.isequal(f1::FusionTree{I, N}, f2::FusionTree{I, N}) where {I<:Sector, N}
    f1.coupled == f2.coupled || return false
    @inbounds for i = 1:N
        f1.uncoupled[i] == f2.uncoupled[i] || return false
        f1.isdual[i] == f2.isdual[i] || return false
    end
    @inbounds for i=1:N-2
        f1.innerlines[i] == f2.innerlines[i] || return false
    end
    return true
end
Base.isequal(f1::FusionTree, f2::FusionTree) = false

# converting to actual array
function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I, 0}) where {I}
    X = convert(A, fusiontensor(one(I), one(I), one(I)))[1, 1, :]
    return X
end
function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I, 1}) where {I}
    c = f.coupled
    if f.isdual[1]
        sqrtdc = sqrtdim(c)
        Zcbartranspose = sqrtdc * convert(A, fusiontensor(conj(c), c, one(c)))[:, :, 1]
        X = conj!(Zcbartranspose) # we want Zcbar^†
    else
        X = convert(A, fusiontensor(c, one(c), c))[:, 1, :]
    end
    return X
end

function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I, 2}) where {I}
    a, b = f.uncoupled
    isduala, isdualb = f.isdual
    c = f.coupled
    da, db, dc = dim.((a, b, c))
    C = convert(A, fusiontensor(a, b, c))[:, :, :]
    X = C
    if isduala
        Xtemp = X
        X = similar(Xtemp)
        Za = convert(A, FusionTree((a,), a, (isduala,), ()))
        # TO.contract!(1, Za, :N, Xtemp, :N, 0, X, (1,), (2,), (2, 3), (1,), (1,2,3))
        TO.tensorcontract!(X, ((1,2,3), ()), Za, ((1,), (2,)), :N, Xtemp, ((1,), (2,3)), :N, true, false)
    end
    if isdualb
        Xtemp = X
        X = similar(Xtemp)
        Zb = convert(A, FusionTree((b,), b, (isdualb,), ()))
        # TO.contract!(1, Zb, :N, Xtemp, :N, 0, X, (1,), (2,), (1, 3), (2,), (2,1,3))
        TO.tensorcontract!(X, ((2,1,3), ()), Zb, ((1,), (2,)), :N, Xtemp, ((2,), (1,3)), :N, true, false)
    end
    return X
end

# Facilitate getting correct fusion tree types
function fusiontreetype(::Type{I}, N::Int) where {I<:Sector}
    if N === 0
        FusionTree{I, 0, 0}
    elseif N === 1
        FusionTree{I, 1, 0}
    else
        FusionTree{I, N, N-2}
    end
end


function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I,N}) where {I,N}
    tailout = (f.innerlines[1], TupleTools.tail2(f.uncoupled)...)
    isdualout = (false, TupleTools.tail2(f.isdual)...)
    ftail = FusionTree(tailout, f.coupled, isdualout,
                        Base.tail(f.innerlines))
    Ctail = convert(A, ftail)
    f1 = FusionTree((f.uncoupled[1], f.uncoupled[2]), f.innerlines[1],
                    (f.isdual[1], f.isdual[2]), ())
    C1 = convert(A, f1)
    dtail = size(Ctail)
    d1 = size(C1)
    X = similar(C1, (d1[1], d1[2], Base.tail(dtail)...))
    trivialtuple = ntuple(identity, Val(N))
    # TO.contract!(1, C1, :N, Ctail, :N, 0, X,
    #                 (1,2), (3,), Base.tail(trivialtuple), (1,), (trivialtuple..., N+1))
    TO.tensorcontract!(X, ((trivialtuple..., N+1), ()), C1, ((1,2), (3,)), :N, Ctail, ((1,), 
                        Base.tail(trivialtuple)), :N, true, false)
    return X
end

# Show methods
function Base.show(io::IO, t::FusionTree{I}) where {I<:Sector}
    print(IOContext(io, :typeinfo => I), "FusionTree{", type_repr(I), "}(",
        t.uncoupled, ", ", t.coupled, ", ", t.isdual, ",",
        t.innerlines, ")")
end

# Manipulate fusion trees
include("manipulations.jl")

# Fusion tree iterators
include("iterator.jl")

# auxiliary routines
# _abelianinner: generate the inner indices for given outer indices in the abelian case
_abelianinner(outer::Tuple{}) = ()
_abelianinner(outer::Tuple{I}) where {I<:Sector} =
    outer[1] == one(I) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{I, I}) where {I<:Sector} =
    outer[1] == dual(outer[2]) ? () : throw(SectorMismatch())
_abelianinner(outer::Tuple{I, I, I}) where {I<:Sector} =
    first(⊗(outer...)) == one(I) ? () : throw(SectorMismatch())
function _abelianinner(outer::NTuple{N, I}) where {I<:Sector, N}
    c = first(outer[1] ⊗ outer[2])
    return (c, _abelianinner((c, TupleTools.tail2(outer)...))...)
end