using Test
using TestExtras
using Random
# using TensorKit
using LinearAlgebra
using TensorOperations
using Combinatorics
using VectorInterface

push!(LOAD_PATH, "../src")
using SphericalTensors
using SphericalTensors: type_repr, findindex

const TK = SphericalTensors

Random.seed!(1234)

smallset(::Type{I}) where {I<:Sector} = Iterators.take(values(I), 5)
function smallset(::Type{ProductSector{Tuple{I1,I2}}}) where {I1,I2}
    iter = Iterators.product(smallset(I1), smallset(I2))
    s = collect(i ⊠ j for (i,j) in iter if dim(i)*dim(j) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function smallset(::Type{ProductSector{Tuple{I1,I2,I3}}}) where {I1,I2,I3}
    iter = Iterators.product(smallset(I1), smallset(I2), smallset(I3))
    s = collect(i ⊠ j ⊠ k for (i,j,k) in iter if dim(i)*dim(j)*dim(k) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function randsector(::Type{I}) where {I<:Sector}
    s = collect(smallset(I))
    a = rand(s)
    while a == one(a) # don't use trivial label
        a = rand(s)
    end
    return a
end


sectorlist = (Z2Irrep, ZNIrrep{3}, Irrep[ℤ{4}], U1Irrep, CU1Irrep, SU2Irrep,
              ZNIrrep{3} ⊠ ZNIrrep{4}, U1Irrep ⊠ SU2Irrep, U1Irrep ⊠ U1Irrep, U1Irrep ⊠ Z2Irrep ⊠ U1Irrep)

include("sectors.jl")
include("fusiontrees.jl")
include("spaces.jl")
include("tensors.jl")
