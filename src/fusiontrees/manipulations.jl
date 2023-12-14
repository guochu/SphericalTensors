# BASIC MANIPULATIONS:
#----------------------------------------------
# -> rewrite generic fusion tree in basis of fusion trees in standard form
# -> only depend on Fsymbol

"""
    insertat(f::FusionTree{I, N₁}, i::Int, f2::FusionTree{I, N₂})
    -> <:AbstractDict{<:FusionTree{I, N₁+N₂-1}, <:Number}

Attach a fusion tree `f2` to the uncoupled leg `i` of the fusion tree `f1` and bring it
into a linear combination of fusion trees in standard form. This requires that
`f2.coupled == f1.uncoupled[i]` and `f1.isdual[i] == false`.
"""
function insertat(f1::FusionTree{I}, i::Int, f2::FusionTree{I, 0}) where {I}
    # this actually removes uncoupled line i, which should be trivial
    (f1.uncoupled[i] == f2.coupled && !f1.isdual[i]) ||
        throw(SectorMismatch("cannot connect $(f2.uncoupled) to $(f1.uncoupled[i])"))
    coeff = _trivial_Fsymbol(I)

    uncoupled = TupleTools.deleteat(f1.uncoupled, i)
    coupled = f1.coupled
    isdual = TupleTools.deleteat(f1.isdual, i)
    if length(uncoupled) <= 2
        inner = ()
    else
        inner = TupleTools.deleteat(f1.innerlines, max(1, i-2))
    end
    f = FusionTree(uncoupled, coupled, isdual, inner)
    return FusionTreeDict(f => coeff)
end
function insertat(f1::FusionTree{I}, i::Int, f2::FusionTree{I, 1}) where {I}
    # identity operation
    (f1.uncoupled[i] == f2.coupled && !f1.isdual[i]) ||
        throw(SectorMismatch("cannot connect $(f2.uncoupled) to $(f1.uncoupled[i])"))
    coeff = _trivial_Fsymbol(I)
    isdual′ = TupleTools.setindex(f1.isdual, f2.isdual[1], i)
    f = FusionTree{I}(f1.uncoupled, f1.coupled, isdual′, f1.innerlines)
    return FusionTreeDict(f => coeff)
end
function insertat(f1::FusionTree{I, N}, i::Int, f2::FusionTree{I, 2}) where {I, N}
    # elementary building block,
    (f1.uncoupled[i] == f2.coupled && !f1.isdual[i]) ||
        throw(SectorMismatch("cannot connect $(f2.uncoupled) to $(f1.uncoupled[i])"))
    uncoupled = f1.uncoupled
    coupled = f1.coupled
    inner = f1.innerlines
    b, c = f2.uncoupled
    isdual = f1.isdual
    isdualb, isdualc = f2.isdual
    if i == 1
        uncoupled′ = (b, c, tail(uncoupled)...)
        isdual′ = (isdualb, isdualc, tail(isdual)...)
        inner′ = (uncoupled[1], inner...)
        coeff = Fsymbol(one(I), one(I), one(I), one(I), one(I), one(I))
        f′ = FusionTree(uncoupled′, coupled, isdual′, inner′)
        return FusionTreeDict(f′ => coeff)
    end
    uncoupled′ = TupleTools.insertafter(TupleTools.setindex(uncoupled, b, i), i, (c,))
    isdual′ = TupleTools.insertafter(TupleTools.setindex(isdual, isdualb, i), i, (isdualc,))
    inner_extended = (uncoupled[1], inner..., coupled)
    a = inner_extended[i-1]
    d = inner_extended[i]
    e′ = uncoupled[i]

    newtrees = FusionTreeDict{fusiontreetype(I, N+1), eltype(I)}()
    for e in a ⊗ b
        coeff = conj(Fsymbol(a, b, c, d, e, e′))
        iszero(coeff) && continue
        inner′ = TupleTools.insertafter(inner, i-2, (e,))
        f′ = FusionTree(uncoupled′, coupled, isdual′, inner′)
        push!(newtrees, f′=> coeff)
    end
    return newtrees
end
function insertat(f1::FusionTree{I,N₁}, i::Int, f2::FusionTree{I,N₂}) where {I,N₁,N₂}
    F = fusiontreetype(I, N₁ + N₂ - 1)
    (f1.uncoupled[i] == f2.coupled && !f1.isdual[i]) ||
        throw(SectorMismatch("cannot connect $(f2.uncoupled) to $(f1.uncoupled[i])"))
    coeff = _trivial_Fsymbol(I)
    T = typeof(coeff)
    if length(f1) == 1
        return FusionTreeDict{F,T}(f2 => coeff)
    end
    if i == 1
        uncoupled = (f2.uncoupled..., tail(f1.uncoupled)...)
        isdual = (f2.isdual..., tail(f1.isdual)...)
        inner = (f2.innerlines..., f2.coupled, f1.innerlines...)
        coupled = f1.coupled
        f′ = FusionTree(uncoupled, coupled, isdual, inner)
        return FusionTreeDict{F,T}(f′ => coeff)
    else # recursive definition
        N2 = length(f2)
        f2′, f2′′ = split(f2, N2 - 1)
        newtrees = FusionTreeDict{F,T}()
        for (f, coeff) in insertat(f1, i, f2′′)
            for (f′, coeff′) in insertat(f, i, f2′)
                coeff′′ = coeff*coeff′
                newtrees[f′] = get(newtrees, f′, zero(coeff′′)) + coeff′′
            end
        end
        return newtrees
    end
end
function _trivial_Fsymbol(I::Type{<:Sector})
    u = one(I)
    Fsymbol(u, u, u, u, u, u)
end

"""
    split(f::FusionTree{I, N}, M::Int)
    -> (::FusionTree{I, M}, ::FusionTree{I, N-M+1})

Split a fusion tree into two. The first tree has as uncoupled sectors the first `M`
uncoupled sectors of the input tree `f`, whereas its coupled sector corresponds to the
internal sector between uncoupled sectors `M` and `M+1` of the original tree `f`. The
second tree has as first uncoupled sector that same internal sector of `f`, followed by
remaining `N-M` uncoupled sectors of `f`. It couples to the same sector as `f`. This
operation is the inverse of `insertat` in the sense that if
`f1, f2 = split(t, M) ⇒ f == insertat(f2, 1, f1)`.
"""
@inline function split(f::FusionTree{I, N}, M::Int) where {I, N}
    if M > N || M < 0
        throw(ArgumentError("M should be between 0 and N = $N"))
    elseif M === N
        (f, FusionTree{I}((f.coupled,), f.coupled, (false,), ()))
    elseif M === 1
        isdual1 = (f.isdual[1],)
        isdual2 = Base.setindex(f.isdual, false, 1)
        f1 = FusionTree{I}((f.uncoupled[1],), f.uncoupled[1], isdual1, ())
        f2 = FusionTree{I}(f.uncoupled, f.coupled, isdual2, f.innerlines)
        return f1, f2
    elseif M === 0
        f1 = FusionTree{I}((), one(I), (), ())
        uncoupled2 = (one(I), f.uncoupled...)
        coupled2 = f.coupled
        isdual2 = (false, f.isdual...)
        innerlines2 = N >= 2 ? (f.uncoupled[1], f.innerlines...) : ()
        return f1, FusionTree{I}(uncoupled2, coupled2, isdual2, innerlines2)
    else
        uncoupled1 = ntuple(n->f.uncoupled[n], M)
        isdual1 = ntuple(n->f.isdual[n], M)
        innerlines1 = ntuple(n->f.innerlines[n], max(0, M-2))
        coupled1 = f.innerlines[M-1]

        uncoupled2 = ntuple(N - M + 1) do n
            n == 1 ? f.innerlines[M - 1] : f.uncoupled[M + n - 1]
        end
        isdual2 = ntuple(N - M + 1) do n
            n == 1 ? false : f.isdual[M + n - 1]
        end
        innerlines2 = ntuple(n->f.innerlines[M-1+n], N-M-1)
        coupled2 = f.coupled

        f1 = FusionTree{I}(uncoupled1, coupled1, isdual1, innerlines1)
        f2 = FusionTree{I}(uncoupled2, coupled2, isdual2, innerlines2)
        return f1, f2
    end
end


"""
    merge(f1::FusionTree{I, N₁}, f2::FusionTree{I, N₂}, c::I, μ = nothing)
    -> <:AbstractDict{<:FusionTree{I, N₁+N₂}, <:Number}

Merge two fusion trees together to a linear combination of fusion trees whose uncoupled
sectors are those of `f1` followed by those of `f2`, and where the two coupled sectors of
`f1` and `f2` are further fused to `c`. In case of
`FusionStyle(I) == GenericFusion()`, also a degeneracy label `μ` for the fusion of
the coupled sectors of `f1` and `f2` to `c` needs to be specified.
"""
function merge(f1::FusionTree{I, N₁}, f2::FusionTree{I, N₂}, c::I) where {I, N₁, N₂}
    if !(c in f1.coupled ⊗ f2.coupled)
        throw(SectorMismatch("cannot fuse sectors $(f1.coupled) and $(f2.coupled) to $c"))
    end
    f0 = FusionTree((f1.coupled, f2.coupled), c, (false, false), ())
    f, coeff = first(insertat(f0, 1, f1)) # takes fast path, single output
    @assert coeff == one(coeff)
    return insertat(f, N₁+1, f2)
end
function merge(f1::FusionTree{I, 0}, f2::FusionTree{I, 0}, c::I) where {I}
    c == one(I) || throw(SectorMismatch("cannot fuse sectors $(f1.coupled) and $(f2.coupled) to $c"))
    return FusionTreeDict(f1=>Fsymbol(c, c, c, c, c, c))
end

# ELEMENTARY DUALITY MANIPULATIONS: A- and B-moves
#---------------------------------------------------------
# -> elementary manipulations that depend on the duality (rigidity) and pivotal structure
# -> planar manipulations that do not require braiding, everything is in Fsymbol (A/Bsymbol)
# -> B-move (bendleft, bendright) is simple in standard basis
# -> A-move (foldleft, foldright) is complicated, needs to be reexpressed in standard form

# change to N₁ - 1, N₂ + 1
function bendright(f1::FusionTree{I, N₁}, f2::FusionTree{I, N₂}) where {I<:Sector, N₁, N₂}
    # map final splitting vertex (a, b)<-c to fusion vertex a<-(c, dual(b))
    @assert N₁ > 0
    c = f1.coupled
    a = N₁ == 1 ? one(I) : (N₁ == 2 ? f1.uncoupled[1] : f1.innerlines[end])
    b = f1.uncoupled[N₁]

    uncoupled1 = Base.front(f1.uncoupled)
    isdual1 = Base.front(f1.isdual)
    inner1 = N₁ > 2 ? Base.front(f1.innerlines) : ()
    f1′ = FusionTree(uncoupled1, a, isdual1, inner1)

    uncoupled2 = (f2.uncoupled..., dual(b))
    isdual2 = (f2.isdual..., !(f1.isdual[N₁]))
    inner2 = N₂ > 1 ? (f2.innerlines..., c) : ()

    coeff = sqrtdim(c) * isqrtdim(a) * Bsymbol(a, b, c)
    if f1.isdual[N₁]
        coeff *= conj(frobeniusschur(dual(b)))
    end
    f2′ = FusionTree(uncoupled2, a, isdual2, inner2)
    return FusionTreeDict( (f1′, f2′) => coeff )
end

# change to N₁ + 1, N₂ - 1
function bendleft(f1::FusionTree{I}, f2::FusionTree{I}) where I
    # map final fusion vertex c<-(a, b) to splitting vertex (c, dual(b))<-a
    return FusionTreeDict((f1′, f2′) => conj(coeff) for
                                ((f2′, f1′), coeff) in bendright(f2, f1))
end

# change to N₁ - 1, N₂ + 1
function foldright(f1::FusionTree{I, N₁}, f2::FusionTree{I, N₂}) where {I<:Sector, N₁, N₂}
    # map first splitting vertex (a, b)<-c to fusion vertex b<-(dual(a), c)
    @assert N₁ > 0
    a = f1.uncoupled[1]
    isduala = f1.isdual[1]
    factor = sqrtdim(a)
    if !isduala
        factor *= frobeniusschur(a)
    end
    c1 = dual(a)
    c2 = f1.coupled
    uncoupled = Base.tail(f1.uncoupled)
    isdual = Base.tail(f1.isdual)

    if N₁ == 1
        cset = (one(c1),)
    elseif N₁ == 2
        cset = (f1.uncoupled[2],)
    else
        cset = ⊗(Base.tail(f1.uncoupled)...)
    end
    newtrees = FusionTreeDict{Tuple{fusiontreetype(I, N₁-1), fusiontreetype(I, N₂+1)}, eltype(I)}()
    for c in c1 ⊗ c2
        c ∈ cset || continue
        fc = FusionTree((c1, c2), c, (!isduala, false), ())
        for (fl′, coeff1) in insertat(fc, 2, f1)
            N₁ > 1 && fl′.innerlines[1] != one(I) && continue
            coupled = fl′.coupled
            uncoupled = Base.tail(Base.tail(fl′.uncoupled))
            isdual = Base.tail(Base.tail(fl′.isdual))
            inner = N₁ <= 3 ? () : Base.tail(Base.tail(fl′.innerlines))
            fl = FusionTree{I}(uncoupled, coupled, isdual, inner)
            for (fr, coeff2) in insertat(fc, 2, f2)
                coeff = factor * coeff1 * coeff2
                newtrees[(fl,fr)] = get(newtrees, (fl, fr), zero(coeff)) + coeff
            end
        end
    end
    return newtrees
end
# change to N₁ + 1, N₂ - 1
function foldleft(f1::FusionTree{I}, f2::FusionTree{I}) where I
    # map first fusion vertex c<-(a, b) to splitting vertex (dual(a), c)<-b
    return FusionTreeDict((f1′, f2′) => conj(coeff) for
                                    ((f2′, f1′), coeff) in foldright(f2, f1))
end


# COMPOSITE DUALITY MANIPULATIONS PART 1: Repartition and transpose
#-------------------------------------------------------------------
# -> composite manipulations that depend on the duality (rigidity) and pivotal structure
# -> planar manipulations that do not require braiding, everything is in Fsymbol (A/Bsymbol)
# -> transpose expressed as cyclic permutation
# one-argument version: check whether `p` is a cyclic permutation (of `1:length(p)`)
function iscyclicpermutation(p)
    N = length(p)
    @inbounds for i = 1:N
        p[mod1(i+1, N)] == mod1(p[i] + 1, N) || return false
    end
    return true
end
# two-argument version: check whether `v1` is a cyclic permutation of `v2`
function iscyclicpermutation(v1, v2)
    length(v1) == length(v2) || return false
    return iscyclicpermutation(indexin(v1, v2))
end

# clockwise cyclic permutation while preserving (N₁, N₂): foldright & bendleft
function cycleclockwise(f1::FusionTree{I, N₁}, f2::FusionTree{I, N₂}) where {I<:Sector, N₁, N₂}
    newtrees = FusionTreeDict{Tuple{fusiontreetype(I, N₁), fusiontreetype(I, N₂)}, eltype(I)}()
    if length(f1) > 0
        for ((f1a, f2a), coeffa) in foldright(f1, f2)
            for ((f1b, f2b), coeffb) in bendleft(f1a, f2a)
                coeff = coeffa * coeffb
                newtrees[(f1b,f2b)] = get(newtrees, (f1b, f2b), zero(coeff)) + coeff
            end
        end
    else
        for ((f1a, f2a), coeffa) in bendleft(f1, f2)
            for ((f1b, f2b), coeffb) in foldright(f1a, f2a)
                coeff = coeffa * coeffb
                newtrees[(f1b,f2b)] = get(newtrees, (f1b, f2b), zero(coeff)) + coeff
            end
        end
    end
    return newtrees
end

# anticlockwise cyclic permutation while preserving (N₁, N₂): foldleft & bendright
function cycleanticlockwise(f1::FusionTree{I, N₁}, f2::FusionTree{I, N₂}) where {I<:Sector, N₁, N₂}
    newtrees = FusionTreeDict{Tuple{fusiontreetype(I, N₁), fusiontreetype(I, N₂)}, eltype(I)}()
    if length(f2) > 0
        for ((f1a, f2a), coeffa) in foldleft(f1, f2)
            for ((f1b, f2b), coeffb) in bendright(f1a, f2a)
                coeff = coeffa * coeffb
                newtrees[(f1b,f2b)] = get(newtrees, (f1b, f2b), zero(coeff)) + coeff
            end
        end
    else
        for ((f1a, f2a), coeffa) in bendright(f1, f2)
            for ((f1b, f2b), coeffb) in foldleft(f1a, f2a)
                coeff = coeffa * coeffb
                newtrees[(f1b,f2b)] = get(newtrees, (f1b, f2b), zero(coeff)) + coeff
            end
        end
    end
    return newtrees
end

# repartition double fusion tree
"""
    repartition(f1::FusionTree{I, N₁}, f2::FusionTree{I, N₂}, N::Int) where {I, N₁, N₂}
    -> <:AbstractDict{Tuple{FusionTree{I, N}, FusionTree{I, N₁+N₂-N}}, <:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`f1`) and incoming sectors (`f2`) respectively (with identical coupled sector
`f1.coupled == f2.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning the tree by bending incoming to outgoing sectors (or vice versa) in order to
have `N` outgoing sectors.
"""
@inline function repartition(f1::FusionTree{I, N₁},
                        f2::FusionTree{I, N₂},
                        N::Int) where {I<:Sector, N₁, N₂}
    f1.coupled == f2.coupled || throw(SectorMismatch())
    @assert 0 <= N <= N₁+N₂
    return _recursive_repartition(f1, f2, Val(N))
end

function _recursive_repartition(f1::FusionTree{I, N₁},
                                f2::FusionTree{I, N₂},
                                ::Val{N}) where {I<:Sector, N₁, N₂, N}
    # recursive definition is only way to get correct number of loops for
    # GenericFusion, but is too complex for type inference to handle, so we
    # precompute the parameters of the return type
    F1 = fusiontreetype(I, N)
    F2 = fusiontreetype(I, N₁ + N₂ - N)
    coeff = @inbounds Fsymbol(one(I), one(I), one(I), one(I), one(I), one(I))
    T = typeof(coeff)
    if N == N₁
        return FusionTreeDict{Tuple{F1, F2}, T}( (f1, f2) => coeff)
    else
        newtrees = FusionTreeDict{Tuple{F1, F2}, T}()
        for ((f1′, f2′), coeff1) in (N < N₁ ? bendright(f1, f2) : bendleft(f1, f2))
            for ((f1′′, f2′′), coeff2) in _recursive_repartition(f1′, f2′, Val(N))
                push!(newtrees, (f1′′, f2′′) => coeff1*coeff2)
            end
        end
        return newtrees
    end
end

"""
    transpose(f1::FusionTree{I}, f2::FusionTree{I},
            p1::NTuple{N₁, Int}, p2::NTuple{N₂, Int}) where {I, N₁, N₂}
    -> <:AbstractDict{Tuple{FusionTree{I, N₁}, FusionTree{I, N₂}}, <:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`t1`) and incoming sectors (`t2`) respectively (with identical coupled sector
`t1.coupled == t2.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning and permuting the tree such that sectors `p1` become outgoing and sectors
`p2` become incoming.
"""
function Base.transpose(f1::FusionTree{I}, f2::FusionTree{I},
                    p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {I<:Sector, N₁, N₂}
    N = N₁ + N₂
    @assert length(f1) + length(f2) == N
    p = linearizepermutation(p1, p2, length(f1), length(f2))
    @assert iscyclicpermutation(p)
    return _transpose((f1, f2, p1, p2))
end

const TransposeKey{I<:Sector, N₁, N₂} = Tuple{<:FusionTree{I}, <:FusionTree{I},
                                                IndexTuple{N₁}, IndexTuple{N₂}}

function _transpose((f1, f2, p1, p2)::TransposeKey{I,N₁,N₂}) where {I<:Sector, N₁, N₂}
    N = N₁ + N₂
    p = linearizepermutation(p1, p2, length(f1), length(f2))
    newtrees = repartition(f1, f2, N₁)
    length(p) == 0 && return newtrees
    i1 = findfirst(==(1), p)
    @assert i1 !== nothing
    i1 == 1 && return newtrees
    Nhalf = N >> 1
    while 1 < i1 <= Nhalf
        newtrees′ = typeof(newtrees)()
        for ((f1a, f2a), coeffa) in newtrees
            for ((f1b, f2b), coeffb) in cycleanticlockwise(f1a, f2a)
                coeff = coeffa * coeffb
                newtrees′[(f1b, f2b)] = get(newtrees′, (f1b, f2b), zero(coeff)) + coeff
            end
        end
        newtrees = newtrees′
        i1 -= 1
    end
    while Nhalf < i1
        newtrees′ = typeof(newtrees)()
        for ((f1a, f2a), coeffa) in newtrees
            for ((f1b, f2b), coeffb) in cycleclockwise(f1a, f2a)
                coeff = coeffa * coeffb
                newtrees′[(f1b, f2b)] = get(newtrees′, (f1b, f2b), zero(coeff)) + coeff
            end
        end
        newtrees = newtrees′
        i1 = mod1(i1 + 1, N)
    end
    return newtrees
end

# BRAIDING MANIPULATIONS:
#-----------------------------------------------
# -> manipulations that depend on a braiding
# -> requires both Fsymbol and Rsymbol
"""
    artin_braid(f::FusionTree, i; inv::Bool = false) -> <:AbstractDict{typeof(t), <:Number}

Perform an elementary braid (Artin generator) of neighbouring uncoupled indices `i` and
`i+1` on a fusion tree `f`, and returns the result as a dictionary of output trees and
corresponding coefficients.

The keyword `inv` determines whether index `i` will braid above or below index `i+1`, i.e.
applying `artin_braid(f′, i; inv = true)` to all the outputs `f′` of
`artin_braid(f, i; inv = false)` and collecting the results should yield a single fusion
tree with non-zero coefficient, namely `f` with coefficient `1`. This keyword has no effect
if  `BraidingStyle(sectortype(f)) isa SymmetricBraiding`.
"""
function artin_braid(f::FusionTree{I, N}, i::Integer; inv::Bool = false) where {I<:Sector, N}
    1 <= i < N ||
        throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    uncoupled = f.uncoupled
    a, b = uncoupled[i], uncoupled[i+1]
    uncoupled′ = TupleTools.setindex(uncoupled, b, i)
    uncoupled′ = TupleTools.setindex(uncoupled′, a, i+1)
    coupled′ = f.coupled
    isdual′ = TupleTools.setindex(f.isdual, f.isdual[i], i+1)
    isdual′ = TupleTools.setindex(isdual′, f.isdual[i+1], i)
    inner = f.innerlines
    inner_extended = (uncoupled[1], inner..., coupled′)
    u = one(I)
    oneT = Rsymbol(u,u,u) * Fsymbol(u,u,u,u,u,u)

    if u in (uncoupled[i], uncoupled[i+1])
        # braiding with trivial sector: simple and always possible
        inner′ = inner
        if i > 1 # we also need to alter innerlines and vertices
            inner′ = TupleTools.setindex(inner, inner_extended[a == u ? (i+1) : (i-1)], i-1)
        end
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′)
        return FusionTreeDict(f′ => oneT)
    end

    if i == 1
        c = N > 2 ? inner[1] : coupled′
        R = oftype(oneT, (inv ? conj(Rsymbol(b, a, c)) : Rsymbol(a, b, c)))
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner)
        return FusionTreeDict(f′ => R)
    end
    # case i > 1: other naming convention
    b = uncoupled[i]
    d = uncoupled[i+1]
    a = inner_extended[i-1]
    c = inner_extended[i]
    e = inner_extended[i+1]

    newtrees = FusionTreeDict{fusiontreetype(I, N), eltype(oneT)}()
    for c′ in intersect(a ⊗ d, e ⊗ conj(b))
        coeff = oftype(oneT, if inv
                conj(Rsymbol(d, c, e)*Fsymbol(d, a, b, e, c′, c))*Rsymbol(d, a, c′)
            else
                Rsymbol(c, d, e)*conj(Fsymbol(d, a, b, e, c′, c)*Rsymbol(a, d, c′))
            end)
        iszero(coeff) && continue
        inner′ = TupleTools.setindex(inner, c′, i-1)
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′)
        push!(newtrees, f′ => coeff)
    end
    return newtrees
end

# braid fusion tree
"""
    braid(f::FusionTree{<:Sector, N}, levels::NTuple{N, Int}, p::NTuple{N, Int})
    -> <:AbstractDict{typeof(t), <:Number}

Perform a braiding of the uncoupled indices of the fusion tree `f` and return the result as
a `<:AbstractDict` of output trees and corresponding coefficients. The braiding is
determined by specifying that the new sector at position `k` corresponds to the sector that
was originally at the position `i = p[k]`, and assigning to every index `i` of the original
fusion tree a distinct level or depth `levels[i]`. This permutation is then decomposed into
elementary swaps between neighbouring indices, where the swaps are applied as braids such
that if `i` and `j` cross, ``τ_{i,j}`` is applied if `levels[i] < levels[j]` and
``τ_{j,i}^{-1}`` if `levels[i] > levels[j]`. This does not allow to encode the most general
braid, but a general braid can be obtained by combining such operations.
"""
function braid(f::FusionTree{I, N},
                levels::NTuple{N, Int},
                p::NTuple{N, Int}) where {I<:Sector, N}
    TupleTools.isperm(p) || throw(ArgumentError("not a valid permutation: $p"))
    coeff = Rsymbol(one(I), one(I), one(I))
    trees = FusionTreeDict(f => coeff)
    newtrees = empty(trees)
    for s in permutation2swaps(p)
        inv = levels[s] > levels[s+1]
        for (f, c) in trees
            for (f′, c′) in artin_braid(f, s; inv = inv)
                newtrees[f′] = get(newtrees, f′, zero(coeff)) + c*c′
            end
        end
        l = levels[s]
        levels = TupleTools.setindex(levels, levels[s+1], s)
        levels = TupleTools.setindex(levels, l, s+1)
        trees, newtrees = newtrees, trees
        empty!(newtrees)
    end
    return trees
end

# permute fusion tree
"""
    permute(f::FusionTree, p::NTuple{N, Int}) -> <:AbstractDict{typeof(t), <:Number}

Perform a permutation of the uncoupled indices of the fusion tree `f` and returns the result
as a `<:AbstractDict` of output trees and corresponding coefficients; this requires that
`BraidingStyle(sectortype(f)) isa SymmetricBraiding`.
"""
function permute(f::FusionTree{I, N}, p::NTuple{N, Int}) where {I<:Sector, N}
    return braid(f, ntuple(identity, Val(N)), p)
end

"""
    braid(f1::FusionTree{I}, f2::FusionTree{I},
            levels1::IndexTuple, levels2::IndexTuple,
            p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {I<:Sector, N₁, N₂}
    -> <:AbstractDict{Tuple{FusionTree{I, N₁}, FusionTree{I, N₂}}, <:Number}

Input is a fusion-splitting tree pair that describes the fusion of a set of incoming
uncoupled sectors to a set of outgoing uncoupled sectors, represented using the splitting
tree `f1` and fusion tree `f2`, such that the incoming sectors `f2.uncoupled` are fused to
`f1.coupled == f2.coupled` and then to the outgoing sectors `f1.uncoupled`. Compute new
trees and corresponding coefficients obtained from repartitioning and braiding the tree such
that sectors `p1` become outgoing and sectors `p2` become incoming. The uncoupled indices in
splitting tree `f1` and fusion tree `f2` have levels (or depths) `levels1` and `levels2`
respectively, which determines how indices braid. In particular, if `i` and `j` cross,
``τ_{i,j}`` is applied if `levels[i] < levels[j]` and ``τ_{j,i}^{-1}`` if `levels[i] >
levels[j]`. This does not allow to encode the most general braid, but a general braid can
be obtained by combining such operations.
"""
function braid(f1::FusionTree{I}, f2::FusionTree{I},
                levels1::IndexTuple, levels2::IndexTuple,
                p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {I<:Sector, N₁, N₂}
    @assert length(f1) + length(f2) == N₁ + N₂
    @assert length(f1) == length(levels1) && length(f2) == length(levels2)
    @assert TupleTools.isperm((p1..., p2...))
    if FusionStyle(I) isa UniqueFusion
        return _abelian_braid((f1, f2, levels1, levels2, p1, p2))
    else
        return _braid((f1, f2, levels1, levels2, p1, p2))
    end
    # return _braid((f1, f2, levels1, levels2, p1, p2))
end

const BraidKey{I<:Sector, N₁, N₂} = Tuple{<:FusionTree{I}, <:FusionTree{I},
                                        IndexTuple, IndexTuple,
                                        IndexTuple{N₁}, IndexTuple{N₂}}

# this function is extensively used and is very costy for nonabelian symmetry
function _braid((f1, f2, l1, l2, p1, p2)::BraidKey{I, N₁, N₂}) where {I<:Sector, N₁, N₂}
    p = linearizepermutation(p1, p2, length(f1), length(f2))
    levels = (l1..., reverse(l2)...)
    newtrees = FusionTreeDict{Tuple{fusiontreetype(I, N₁), fusiontreetype(I, N₂)}, eltype(I)}()
    for ((f, f0), coeff1) in repartition(f1, f2, N₁ + N₂)
        for (f′, coeff2) in braid(f, levels, p)
            for ((f1′, f2′), coeff3) in repartition(f′, f0, N₁)
                newtrees[(f1′, f2′)] = get(newtrees, (f1′, f2′), zero(coeff3)) +
                    coeff1*coeff2*coeff3
            end
        end
    end
    return newtrees
end
# shortcut for abelian symmetry
function _abelian_braid((f1, f2, l1, l2, p1, p2)::BraidKey{I, N₁, N₂}) where {I<:Sector, N₁, N₂}
    isdual = (f1.isdual..., (!).(f2.isdual)...)
    uncoupled = (f1.uncoupled..., dual.(f2.uncoupled)...)
    isdual1′, isdual2′ = TupleTools.getindices(isdual, p1), TupleTools.getindices(isdual, p2)
    uncoupled1′, uncoupled2′ = TupleTools.getindices(uncoupled, p1), TupleTools.getindices(uncoupled, p2)
    uncoupled2′ = ntuple(i->dual(uncoupled2′[i]), Val(N₂))
    isdual2′ = (!).(isdual2′)
    coupled1′ = first(⊗(I, uncoupled1′...))
    coupled2′ = first(⊗(I, uncoupled2′...))
    f1′ = FusionTree(uncoupled1′, coupled1′, isdual1′)
    f2′ = FusionTree(uncoupled2′, coupled2′, isdual2′)
    # newtrees = FusionTreeDict{Tuple{fusiontreetype(I, N₁), fusiontreetype(I, N₂)}, eltype(I)}()
    # coeff = (coupled1′ == coupled2′) ? 1 : 0
    # newtrees[(f1′, f2′)] = (coupled1′ == coupled2′) ? 1 : 0
    coeff = (coupled1′ == coupled2′) ? 1 : 0
    return SingletonDict((f1′, f2′)=>coeff)
end

"""
    permute(f1::FusionTree{I}, f2::FusionTree{I},
            p1::NTuple{N₁, Int}, p2::NTuple{N₂, Int}) where {I, N₁, N₂}
    -> <:AbstractDict{Tuple{FusionTree{I, N₁}, FusionTree{I, N₂}}, <:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`t1`) and incoming sectors (`t2`) respectively (with identical coupled sector
`t1.coupled == t2.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning and permuting the tree such that sectors `p1` become outgoing and sectors
`p2` become incoming.
"""
function permute(f1::FusionTree{I}, f2::FusionTree{I},
                    p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {I<:Sector, N₁, N₂}
    levels1 = ntuple(identity, length(f1))
    levels2 = length(f1) .+ ntuple(identity, length(f2))
    return braid(f1, f2, levels1, levels2, p1, p2)
end