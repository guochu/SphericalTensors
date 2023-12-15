# TensorMap & Tensor:
# general tensor implementation with arbitrary symmetries
#==========================================================#
"""
    struct TensorMap{S<:IndexSpace, N₁, N₂, ...} <: AbstractTensorMap{S, N₁, N₂}

Specific subtype of [`AbstractTensorMap`](@ref) for representing tensor maps (morphisms in
a tensor category) whose data is stored in blocks of some subtype of `DenseMatrix`.
"""
struct TensorMap{S<:IndexSpace, N₁, N₂, I<:Sector, A<:SectorDict{I,<:DenseMatrix}, F₁, F₂} <: AbstractTensorMap{S, N₁, N₂}
    data::A
    codom::ProductSpace{S,N₁}
    dom::ProductSpace{S,N₂}
    rowr::SectorDict{I,FusionTreeDict{F₁,UnitRange{Int}}}
    colr::SectorDict{I,FusionTreeDict{F₂,UnitRange{Int}}}
end

function tensormaptype(::Type{S}, N₁::Int, N₂::Int, ::Type{T}) where {S,T}
    I = sectortype(S)
    if T <: DenseMatrix
        M = T
    elseif T <: Number
        M = Matrix{T}
    else
        throw(ArgumentError("the final argument of `tensormaptype` should either be the scalar or the storage type, i.e. a subtype of `Number` or of `DenseMatrix`"))
    end

    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    return TensorMap{S,N₁,N₂,I,SectorDict{I,M},F₁,F₂}
end
tensormaptype(S, N₁, N₂ = 0) = tensormaptype(S, N₁, N₂, Float64)

# Basic methods for characterising a tensor:
#--------------------------------------------
codomain(t::TensorMap) = t.codom
domain(t::TensorMap) = t.dom

blocksectors(t::TensorMap) = keys(t.data)

storagetype(::Type{<:TensorMap{<:IndexSpace,N₁,N₂,I,<:SectorDict{I,A}}}) where
    {N₁,N₂,I<:Sector,A<:DenseMatrix} = A

dim(t::TensorMap) = mapreduce(x->length(x[2]), +, blocks(t); init = 0)

# General TensorMap constructors
#--------------------------------
function TensorMap(data::AbstractDict{<:Sector,<:DenseMatrix}, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S<:IndexSpace, N₁, N₂}
    I = sectortype(S)
    I == keytype(data) || throw(SectorMismatch())

    blocksectoriterator = blocksectors(codom ← dom)
    for c in blocksectoriterator
        haskey(data, c) || throw(SectorMismatch("no data for block sector $c"))
    end
    rowr, rowdims = _buildblockstructure(codom, blocksectoriterator)
    colr, coldims = _buildblockstructure(dom, blocksectoriterator)
    for (c, b) in data
        c in blocksectoriterator || isempty(b) ||
            throw(SectorMismatch("data for block sector $c not expected"))
        isempty(b) || size(b) == (rowdims[c], coldims[c]) ||
            throw(DimensionMismatch("wrong size of block for sector $c"))
    end
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    if !isreal(I)
        data2 = SectorDict(c => complex(data[c]) for c in blocksectoriterator)
        A = typeof(data2)
        return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data2, codom, dom, rowr, colr)
    else
        data2 = SectorDict(c => data[c] for c in blocksectoriterator)
        A = typeof(data2)
        return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data2, codom, dom, rowr, colr)
    end
end

# without data: generic constructor from callable:
function TensorMap(f, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂}) where {S<:IndexSpace, N₁, N₂}
    I = sectortype(S)
    blocksectoriterator = blocksectors(codom ← dom)
    rowr, rowdims = _buildblockstructure(codom, blocksectoriterator)
    colr, coldims = _buildblockstructure(dom, blocksectoriterator)
    if !isreal(I)
        data = SectorDict(c => complex(f((rowdims[c], coldims[c])))
                          for c in blocksectoriterator)
    else
        data = SectorDict(c => f((rowdims[c], coldims[c])) for c in blocksectoriterator)
    end
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    A = typeof(data)
    return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data, codom, dom, rowr, colr)
end

# auxiliary function
function _buildblockstructure(P::ProductSpace{S,N}, blocksectors) where {S<:IndexSpace,N}
    I = sectortype(S)
    F = fusiontreetype(I, N)
    treeranges = SectorDict{I,FusionTreeDict{F,UnitRange{Int}}}()
    blockdims = SectorDict{I,Int}()
    for s in sectors(P)
        for c in blocksectors
            offset = get!(blockdims, c, 0)
            treerangesc = get!(treeranges, c) do
                return FusionTreeDict{F,UnitRange{Int}}()
            end
            for f in fusiontrees(s, c, map(isdual, P.spaces))
                r = (offset + 1):(offset + dim(P, s))
                push!(treerangesc, f => r)
                offset = last(r)
            end
            blockdims[c] = offset
        end
    end
    return treeranges, blockdims
end

# constructor starting from a dense array
function TensorMap(data::DenseArray, codom::ProductSpace{S,N₁}, dom::ProductSpace{S,N₂};
                   tol=sqrt(eps(real(float(eltype(data)))))) where {S<:IndexSpace,N₁,N₂}
    (d1, d2) = (dim(codom), dim(dom))
    if !(length(data) == d1 * d2 || size(data) == (d1, d2) ||
         size(data) == (dims(codom)..., dims(dom)...))
        throw(DimensionMismatch())
    end

    t = TensorMap(zeros, eltype(data), codom, dom)
    ta = convert(Array, t)
    l = length(ta)
    dimt = dim(t)
    basis = zeros(eltype(ta), (l, dimt))
    qdims = zeros(real(eltype(ta)), (dimt,))
    i = 1
    for (c, b) in blocks(t)
        for k in 1:length(b)
            b[k] = 1
            copy!(view(basis, :, i), reshape(convert(Array, t), (l,)))
            qdims[i] = dim(c)
            b[k] = 0
            i += 1
        end
    end
    rhs = reshape(data, (l,))
    if FusionStyle(sectortype(t)) isa UniqueFusion
        lhs = basis' * rhs
    else
        lhs = Diagonal(qdims) \ (basis' * rhs)
    end
    if norm(basis * lhs - rhs) > tol
        throw(ArgumentError("Data has non-zero elements at incompatible positions"))
    end
    if eltype(lhs) != scalartype(t)
        t2 = TensorMap(zeros, promote_type(eltype(lhs), scalartype(t)), codom, dom)
    else
        t2 = t
    end
    i = 1
    for (c, b) in blocks(t2)
        for k in 1:length(b)
            b[k] = lhs[i]
            i += 1
        end
    end
    return t2

end

TensorMap(f,
            ::Type{T},
            codom::ProductSpace{S},
            dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->f(T, d), codom, dom)

TensorMap(::Type{T},
            codom::ProductSpace{S},
            dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->Array{T}(undef, d), codom, dom)

TensorMap(::UndefInitializer,
            ::Type{T},
            codom::ProductSpace{S},
            dom::ProductSpace{S}) where {S<:IndexSpace, T<:Number} =
    TensorMap(d->Array{T}(undef, d), codom, dom)

TensorMap(::UndefInitializer,
            codom::ProductSpace{S},
            dom::ProductSpace{S}) where {S<:IndexSpace} =
    TensorMap(undef, Float64, codom, dom)

TensorMap(::Type{T},
            codom::TensorSpace{S},
            dom::TensorSpace{S}) where {T<:Number, S<:IndexSpace} =
    TensorMap(T, convert(ProductSpace, codom), convert(ProductSpace, dom))

TensorMap(dataorf, codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:IndexSpace} =
    TensorMap(dataorf, convert(ProductSpace, codom), convert(ProductSpace, dom))

TensorMap(dataorf, ::Type{T},
            codom::TensorSpace{S},
            dom::TensorSpace{S}) where {T<:Number, S<:IndexSpace} =
    TensorMap(dataorf, T, convert(ProductSpace, codom), convert(ProductSpace, dom))

TensorMap(codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:IndexSpace} =
    TensorMap(Float64, convert(ProductSpace, codom), convert(ProductSpace, dom))

TensorMap(dataorf, T::Type{<:Number}, P::TensorMapSpace{S}) where {S<:IndexSpace} =
    TensorMap(dataorf, T, codomain(P), domain(P))

TensorMap(dataorf, P::TensorMapSpace{S}) where {S<:IndexSpace} =
    TensorMap(dataorf, codomain(P), domain(P))

TensorMap(T::Type{<:Number}, P::TensorMapSpace{S}) where {S<:IndexSpace} =
    TensorMap(T, codomain(P), domain(P))

TensorMap(P::TensorMapSpace{S}) where {S<:IndexSpace} = TensorMap(codomain(P), domain(P))

TensorMap(dataorf, T::Type{<:Number}, P::TensorSpace{S}) where {S<:IndexSpace} =
    TensorMap(dataorf, T, P, one(P))

# Efficient copy constructors
#-----------------------------
function Base.copy(t::TensorMap{S, N₁, N₂, I, A, F₁, F₂}) where {S, N₁, N₂, I, A, F₁, F₂}
    return TensorMap{S, N₁, N₂, I, A, F₁, F₂}(deepcopy(t.data), t.codom, t.dom, t.rowr, t.colr)
end

# Similar
#---------
# 4 arguments
function Base.similar(t::AbstractTensorMap, T::Type, codomain::VectorSpace,
                      domain::VectorSpace)
    return similar(t, T, codomain ← domain)
end
# 3 arguments
function Base.similar(t::AbstractTensorMap, codomain::VectorSpace, domain::VectorSpace)
    return similar(t, scalartype(t), codomain ← domain)
end
function Base.similar(t::AbstractTensorMap, T::Type, codomain::VectorSpace)
    return similar(t, T, codomain ← one(codomain))
end
# 2 arguments
function Base.similar(t::AbstractTensorMap, codomain::VectorSpace)
    return similar(t, scalartype(t), codomain ← one(codomain))
end
Base.similar(t::AbstractTensorMap, P::TensorMapSpace) = similar(t, scalartype(t), P)
Base.similar(t::AbstractTensorMap, T::Type) = similar(t, T, space(t))
# 1 argument
Base.similar(t::AbstractTensorMap) = similar(t, scalartype(t), space(t))

# actual implementation
function Base.similar(t::TensorMap{S}, ::Type{T}, P::TensorMapSpace{S}) where {T,S}
    N₁ = length(codomain(P))
    N₂ = length(domain(P))
    I = sectortype(S)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    if space(t) == P
        data = SectorDict(c => similar(b, T) for (c, b) in blocks(t))
        A = typeof(data)
        return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data, codomain(P), domain(P), t.rowr, t.colr)
    end

    blocksectoriterator = blocksectors(P)
    # try to recycle rowr
    if codomain(P) == codomain(t) && all(c -> haskey(t.rowr, c), blocksectoriterator)
        if length(t.rowr) == length(blocksectoriterator)
            rowr = t.rowr
        else
            rowr = SectorDict(c => t.rowr[c] for c in blocksectoriterator)
        end
        rowdims = SectorDict(c => size(block(t, c), 1) for c in blocksectoriterator)
    elseif codomain(P) == domain(t) && all(c -> haskey(t.colr, c), blocksectoriterator)
        if length(t.colr) == length(blocksectoriterator)
            rowr = t.colr
        else
            rowr = SectorDict(c => t.colr[c] for c in blocksectoriterator)
        end
        rowdims = SectorDict(c => size(block(t, c), 2) for c in blocksectoriterator)
    else
        rowr, rowdims = _buildblockstructure(codomain(P), blocksectoriterator)
    end
    # try to recylce colr
    if domain(P) == codomain(t) && all(c -> haskey(t.rowr, c), blocksectoriterator)
        if length(t.rowr) == length(blocksectoriterator)
            colr = t.rowr
        else
            colr = SectorDict(c => t.rowr[c] for c in blocksectoriterator)
        end
        coldims = SectorDict(c => size(block(t, c), 1) for c in blocksectoriterator)
    elseif domain(P) == domain(t) && all(c -> haskey(t.colr, c), blocksectoriterator)
        if length(t.colr) == length(blocksectoriterator)
            colr = t.colr
        else
            colr = SectorDict(c => t.colr[c] for c in blocksectoriterator)
        end
        coldims = SectorDict(c => size(block(t, c), 2) for c in blocksectoriterator)
    else
        colr, coldims = _buildblockstructure(domain(P), blocksectoriterator)
    end
    M = similarstoragetype(t, T)
    data = SectorDict{I,M}(c => M(undef, (rowdims[c], coldims[c]))
                           for c in blocksectoriterator)
    A = typeof(data)
    return TensorMap{S,N₁,N₂,I,A,F₁,F₂}(data, codomain(P), domain(P), rowr, colr)
end

function Base.complex(t::AbstractTensorMap)
    if scalartype(t) <: Complex
        return t
    else
        return copy!(similar(t, complex(scalartype(t))), t)
    end
end

# Conversion between TensorMap and Dict, for read and write purpose
#------------------------------------------------------------------
function Base.convert(::Type{Dict}, t::AbstractTensorMap)
    d = Dict{Symbol,Any}()
    d[:codomain] = repr(codomain(t))
    d[:domain] = repr(domain(t))
    data = Dict{String,Any}()
    for (c,b) in blocks(t)
        data[repr(c)] = Array(b)
    end
    d[:data] = data
    return d
end
function Base.convert(::Type{TensorMap}, d::Dict{Symbol,Any})
    try
        codomain = eval(Meta.parse(d[:codomain]))
        domain = eval(Meta.parse(d[:domain]))
        data = SectorDict(eval(Meta.parse(c))=>b for (c,b) in d[:data])
        return TensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(Base.eval(Main, Meta.parse(c))=>b for (c,b) in d[:data])
        return TensorMap(data, codomain, domain)
    end
end

# Getting and setting the data
#------------------------------
hasblock(t::TensorMap, s::Sector) = haskey(t.data, s)

function block(t::TensorMap, s::Sector)
    sectortype(t) == typeof(s) || throw(SectorMismatch())
    if haskey(t.data, s)
        return t.data[s]
    else # at least one of the two matrix dimensions will be zero
        return storagetype(t)(undef, (blockdim(codomain(t),s), blockdim(domain(t), s)))
    end
end

blocks(t::TensorMap) = t.data
fusiontrees(t::TensorMap) = TensorKeyIterator(t.rowr, t.colr)

@inline function Base.getindex(t::TensorMap{<:IndexSpace,N₁,N₂,I},
                                sectors::NTuple{N, I}) where {N₁,N₂,I<:Sector, N}

    (FusionStyle(I) isa UniqueFusion) || throw(SectorMismatch("Indexing with sectors only possible if unique fusion"))
    @assert N == N₁ + N₂
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

@inline function Base.getindex(t::TensorMap{<:IndexSpace,N₁,N₂,I},
        f1::FusionTree{I,N₁}, f2::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector}

    c = f1.coupled
    @boundscheck begin
        c == f2.coupled || throw(SectorMismatch())
        haskey(t.rowr[c], f1) || throw(SectorMismatch())
        haskey(t.colr[c], f2) || throw(SectorMismatch())
    end
    @inbounds begin
        d = (dims(codomain(t), f1.uncoupled)..., dims(domain(t), f2.uncoupled)...)
        return sreshape(StridedView(t.data[c])[t.rowr[c][f1], t.colr[c][f2]], d)
    end
end
function Base.get(t::TensorMap{<:IndexSpace,N₁,N₂,I}, f::Tuple{FusionTree{I,N₁}, FusionTree{I,N₂}}, default=nothing) where {N₁,N₂,I<:Sector}
    f1, f2 = f
    c = f1.coupled
    c == f2.coupled || return default
    tmp1 = get(t.rowr, c, nothing)
    isnothing(tmp1) && return default
    r1 = get(tmp1, f1, nothing)
    isnothing(r1) && return default
    tmp2 = get(t.colr, c, nothing)
    isnothing(tmp2) && return default
    r2 = get(tmp2, f2, nothing)
    isnothing(r2) && return default
    d = (dims(codomain(t), f1.uncoupled)..., dims(domain(t), f2.uncoupled)...)
    return sreshape(StridedView(t.data[c])[r1, r2], d)
end
function hasfusiontree(t::TensorMap{<:IndexSpace,N₁,N₂,I}, f1::FusionTree{I,N₁}, f2::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector}
    c = f1.coupled
    c == f2.coupled || throw(SectorMismatch())
    (haskey(t.rowr, c) &&  haskey(t.colr, c)) || return false
    return haskey(t.rowr[c], f1) && haskey(t.colr[c], f2)
end
@propagate_inbounds Base.setindex!(t::TensorMap{<:IndexSpace,N₁,N₂,I},
                                    v,
                                    f1::FusionTree{I,N₁},
                                    f2::FusionTree{I,N₂}) where {N₁,N₂,I<:Sector} =
                                    copy!(getindex(t, f1, f2), v)

# Show
#------
function Base.summary(t::TensorMap)
    print("TensorMap(", codomain(t), " ← ", domain(t), ")")
end
function Base.show(io::IO, t::TensorMap{S}) where {S<:IndexSpace}
    if get(io, :compact, false)
        print(io, "TensorMap(", codomain(t), " ← ", domain(t), ")")
        return
    end
    println(io, "TensorMap(", codomain(t), " ← ", domain(t), "):")
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

# Real and imaginary parts
#---------------------------
function Base.real(t::AbstractTensorMap{S}) where {S}
    # `isreal` for a `Sector` returns true iff the F and R symbols are real. This guarantees
    # that the real/imaginary part of a tensor `t` can be obtained by just taking
    # real/imaginary part of the degeneracy data.
    if isreal(sectortype(S))
        realdata = Dict(k => real(v) for (k, v) in blocks(t))
        return TensorMap(realdata, codomain(t), domain(t))
    else
        msg = "`real` has not been implemented for `AbstractTensorMap{$(S)}`."
        throw(ArgumentError(msg))
    end
end

function Base.imag(t::AbstractTensorMap{S}) where {S}
    # `isreal` for a `Sector` returns true iff the F and R symbols are real. This guarantees
    # that the real/imaginary part of a tensor `t` can be obtained by just taking
    # real/imaginary part of the degeneracy data.
    if isreal(sectortype(S))
        imagdata = Dict(k => imag(v) for (k, v) in blocks(t))
        return TensorMap(imagdata, codomain(t), domain(t))
    else
        msg = "`imag` has not been implemented for `AbstractTensorMap{$(S)}`."
        throw(ArgumentError(msg))
    end
end

# Conversion and promotion:
#---------------------------
Base.convert(::Type{TensorMap}, t::TensorMap) = t
Base.convert(::Type{TensorMap}, t::AbstractTensorMap) =
    copy!(TensorMap(undef, scalartype(t), codomain(t), domain(t)), t)

function Base.convert(T::Type{TensorMap{S,N₁,N₂,I,A,F1,F2}},
                        t::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂,I,A,F1,F2}
    if typeof(t) == T
        return t
    else
        data = Dict(c=>convert(storagetype(T), b) for (c,b) in blocks(t))
        return TensorMap(data, codomain(t), domain(t))
    end
end
