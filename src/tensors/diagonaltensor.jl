

struct DiagonalMap{S<:IndexSpace, I<:Sector, A<:SectorDict{I,<:Diagonal}, F₁, F₂} <: AbstractTensorMap{S, 1, 1}
    data::A
    codom::ProductSpace{S,1}
    dom::ProductSpace{S,1}
    rowr::SectorDict{I,FusionTreeDict{F₁,UnitRange{Int}}}
    colr::SectorDict{I,FusionTreeDict{F₂,UnitRange{Int}}}
end

# Basic methods for characterising a tensor:
#--------------------------------------------
codomain(t::DiagonalMap) = t.codom
domain(t::DiagonalMap) = t.dom
storagetype(::Type{<:DiagonalMap{<:IndexSpace,I,<:SectorDict{I,A}}}) where {I<:Sector,A<:Diagonal} = Matrix{eltype(A)}

blocksectors(t::DiagonalMap) = keys(t.data)
hasblock(t::DiagonalMap, s::Sector) = haskey(t.data, s)

blocks(t::DiagonalMap) = t.data
fusiontrees(t::DiagonalMap) = TensorKeyIterator(t.rowr, t.colr)

# Getting and setting the data
#------------------------------
function block(t::DiagonalMap, s::Sector)
    sectortype(t) == typeof(s) || throw(SectorMismatch())
    if haskey(t.data, s)
        return t.data[s]
    else # at least one of the two matrix dimensions will be zero
        return storagetype(t)(undef, (blockdim(codomain(t),s), blockdim(domain(t), s)))
    end
end


@inline function Base.getindex(t::DiagonalMap{<:IndexSpace,I},
                                sectors::NTuple{2, I}) where {I<:Sector}

    (FusionStyle(I) isa UniqueFusion) || throw(SectorMismatch("Indexing with sectors only possible if unique fusion"))
    s1 = TupleTools.getindices(sectors, codomainind(t))
    s2 = map(dual, TupleTools.getindices(sectors, domainind(t)))
    c1 = length(s1) == 0 ? one(I) : (length(s1) == 1 ? s1[1] : first(⊗(s1...)))
    @boundscheck begin
        c2 = length(s2) == 0 ? one(I) : (length(s2) == 1 ? s2[1] : first(⊗(s2...)))
        c2 == c1 || throw(SectorMismatch("Not a valid sector for this tensor"))
        hassector(codomain(t), s1) && hassector(domain(t), s2)
    end
    f1 = FusionTree(s1, c1, map(isdual, tuple(codomain(t)...)))
    f2 = FusionTree(s2, c1, map(isdual, tuple(domain(t)...)))
    @inbounds begin
        return t[f1,f2]
    end
end

@inline function Base.getindex(t::DiagonalMap{<:IndexSpace,I},
        f1::FusionTree{I,1}, f2::FusionTree{I,1}) where {I<:Sector}

    c = f1.coupled
    @boundscheck begin
        c == f2.coupled || throw(SectorMismatch())
        haskey(t.rowr[c], f1) || throw(SectorMismatch())
        haskey(t.colr[c], f2) || throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(codomain(t), f1.uncoupled)..., dims(domain(t), f2.uncoupled)...)
        # return sreshape(StridedView(t.data[c])[t.rowr[c][f1], t.colr[c][f2]], d)
        r = view(t.data[c], t.rowr[c][f1], t.colr[c][f2])
        (size(r) == d) || throw(ArgumentError("should not be here"))
        return r
    end
end
@propagate_inbounds Base.setindex!(t::DiagonalMap{<:IndexSpace,I},
                                    v,
                                    f1::FusionTree{I,1},
                                    f2::FusionTree{I,1}) where {I<:Sector} =
                                    copy!(getindex(t, f1, f2), v)


function diagonalmaptype(::Type{S}, ::Type{T}) where {S<:IndexSpace, T<:Number}
    I = sectortype(S)
    F₁ = fusiontreetype(I, 1)
    M = Diagonal{T, Vector{T}}
    return DiagonalMap{S, I, SectorDict{I, M}, F₁, F₁}
end

dim(t::DiagonalMap) = mapreduce(x->length(x[2]), +, blocks(t); init = 0)

# General TensorMap constructors
#--------------------------------
function DiagonalMap(data::AbstractDict{<:Sector,<:Diagonal}, codom::ProductSpace{S,1}, dom::ProductSpace{S,1}) where {S<:IndexSpace}
    I = sectortype(S)
    I == keytype(data) || throw(SectorMismatch())
    F₁ = fusiontreetype(I, 1)
    F₂ = fusiontreetype(I, 1)
    rowr = SectorDict{I, FusionTreeDict{F₁, UnitRange{Int}}}()
    colr = SectorDict{I, FusionTreeDict{F₂, UnitRange{Int}}}()
    rowdims = SectorDict{I, Int}()
    coldims = SectorDict{I, Int}()
    blocksectoriterator = blocksectors(dom)
    for s1 in sectors(codom)
        for c in blocksectoriterator
            offset1 = get!(rowdims, c, 0)
            rowrc = get!(rowr, c) do
                FusionTreeDict{F₁, UnitRange{Int}}()
            end
            for f1 in fusiontrees(s1, c, map(isdual, codom.spaces))
                r = (offset1 + 1):(offset1 + dim(codom, s1))
                push!(rowrc, f1 => r)
                offset1 = last(r)
            end
            rowdims[c] = offset1
        end
    end
    for s2 in sectors(dom)
        for c in blocksectoriterator
            offset2 = get!(coldims, c, 0)
            colrc = get!(colr, c) do
                FusionTreeDict{F₂, UnitRange{Int}}()
            end
            for f2 in fusiontrees(s2, c, map(isdual, dom.spaces))
                r = (offset2 + 1):(offset2 + dim(dom, s2))
                push!(colrc, f2 => r)
                offset2 = last(r)
            end
            coldims[c] = offset2
        end
    end
    for c in blocksectoriterator
        dim1 = get!(rowdims, c, 0)
        dim2 = get!(coldims, c, 0)
        if dim1 == 0 || dim2 == 0
            delete!(rowr, c)
            delete!(colr, c)
        else
            (haskey(data, c) && size(data[c]) == (dim1, dim2)) ||
            throw(DimensionMismatch())
        end
    end
    if !isreal(I) && eltype(valtype(data)) <: Real
        b = valtype(data)(undef, (0,0))
        V = typeof(complex(b))
        K = keytype(data)
        data2 = SectorDict{K,V}((c=>complex(data[c])) for c in keys(rowr))
        A = typeof(data2)
        return DiagonalMap{S, I, A, F₁, F₂}(data2, codom, dom, rowr, colr)
    else
        V = valtype(data)
        K = keytype(data)
        data2 = SectorDict{K,V}((c=>data[c]) for c in keys(rowr))
        A = typeof(data2)
        return DiagonalMap{S, I, A, F₁, F₂}(data2, codom, dom, rowr, colr)
    end
end
DiagonalMap(dataorf, codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:IndexSpace} =
    DiagonalMap(dataorf, convert(ProductSpace, codom), convert(ProductSpace, dom))
DiagonalMap(dataorf, P::TensorMapSpace{S}) where {S<:IndexSpace} =
    DiagonalMap(dataorf, codomain(P), domain(P))

# Similar
#---------
Base.similar(t::DiagonalMap, T::Type, P::TensorMapSpace) = TensorMap(undef, T, P)
Base.copy(b::DiagonalMap) = copy!(similar(b), b)
function Base.copy!(t::TensorMap, t2::DiagonalMap)
    space(t) == space(t2) || throw(SectorMismatch())
    # fill!(t, zero(scalartype(t)))
    for (c, b) in blocks(t)
        copy!(b, block(t2, c))
    end
    return t
end
Base.convert(A::Type{DiagonalMap}, t::TensorMap{<:IndexSpace, 1, 1}) = Diagonal(t)

function Base.one(t::DiagonalMap)
    domain(t) == codomain(t) ||
        throw(SectorMismatch("no identity if domain and codomain are different"))
    data = SectorDict(c=>one(bc) for (c, bc) in blocks(t))
    return DiagonalMap(data, codomain(t), domain(t), t.rowr, t.colr)
end

LinearAlgebra.Diagonal(t::TensorMap{<:IndexSpace, 1, 1}) = DiagonalMap(SectorDict(c=>Diagonal(b) for (c, b) in blocks(t)), codomain(t), domain(t), t.rowr, t.colr)	
Strided.StridedView(t::Diagonal) = t
Base.adjoint(t::DiagonalMap) = DiagonalMap(SectorDict(c=>bc' for (c, bc) in blocks(t)), domain(t), codomain(t), t.colr, t.rowr)

function Base.inv(t::DiagonalMap)
    cod = codomain(t)
    dom = domain(t)
    isisomorphic(cod, dom) || throw(SpaceMismatch("codomain $cod and domain $dom are not isomorphic: no inverse"))

    data = empty(t.data)
    for (c, b) in blocks(t)
        data[c] = inv(b)
    end
    return DiagonalMap(data, domain(t)←codomain(t))
end
function LinearAlgebra.pinv(t::DiagonalMap; kwargs...)
    data = empty(t.data)
    for (c, b) in blocks(t)
        data[c] = pinv(b; kwargs...)
    end
    return DiagonalMap(data, domain(t)←codomain(t))
end

# Show
#------
function Base.summary(t::DiagonalMap)
    print("DiagonalMap(", codomain(t), " ← ", domain(t), ")")
end
function Base.show(io::IO, t::DiagonalMap{S}) where {S<:IndexSpace}
    if get(io, :compact, false)
        print(io, "DiagonalMap(", codomain(t), " ← ", domain(t), ")")
        return
    end
    println(io, "DiagonalMap(", codomain(t), " ← ", domain(t), "):")
    if FusionStyle(sectortype(S)) isa UniqueFusion
        for (f1,f2) in fusiontrees(t)
            println(io, "* Data for sector ", f1.uncoupled, " ← ", f2.uncoupled, ":")
            Base.print_array(io, t[f1,f2])
            println(io)
        end
    else
        for (f1,f2) in fusiontrees(t)
            println(io, "* Data for fusiontree ", f1, " ← ", f2, ":")
            Base.print_array(io, t[f1,f2])
            println(io)
        end
    end
end






