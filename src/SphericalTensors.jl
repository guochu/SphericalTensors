module SphericalTensors

export SpaceMismatch, SectorMismatch, IndexError

export Group, AbelianGroup, ProductGroup, ℤ, ℤ₂, U₁, SU₂, CU₁
export FusionStyle, UniqueFusion, SimpleFusion, MultiplicityFreeFusion
export Sector, AbstractIrrep, AbelianIrrep, ProductSector

export FusionTree, fusiontrees

export Z2Irrep, ZNIrrep, U1Irrep, SU2Irrep, CU1Irrep
export VectorSpace, ElementarySpace, GradedSpace, ProductSpace, HomSpace
export ZNSpace, Z2Space, U1Space, ℤ₂Space, U₁Space, SU₂Space, SU2Space, Irrep, Rep, Vect


export dual, isdual, Nsymbol, sectortype, spacetype, space, dim, dims, dual, flip, fuse
export sectors, hassector, blocksectors, blockdim, insertunit, infimum, supremum

# methods for sectors and properties thereof
export Fsymbol, Rsymbol, Bsymbol, frobeniusschur, fusiontensor
export braid, permute, transpose

# some unicode
export ×, ⊕, ⊗, ⊠, ≾, ≿, ≅, ≺, ≻, ←, →

# tensor
export TensorMap, TensorSpace, TensorMapSpace, AbstractTensorMap
export blocks, block, id, isomorphism, isometry, tensormaptype, domain, codomain, hasfusiontree
export storagetype, numind, numout, numin, domainind, codomainind, allind, scalartype

# tensor algebra and factorizations
export dot, norm, normalize, normalize!, tr
export mul!, lmul!, rmul!, adjoint!, pinv, axpy!, axpby!
export leftorth, rightorth, leftnull, rightnull,
        leftorth!, rightorth!, leftnull!, rightnull!,
        tsvd!, tsvd, eigen, eigen!, eig, eig!, eigh, eigh!, exp, exp!,
        isposdef, isposdef!, ishermitian, sylvester
export braid!, permute!, transpose!
export catdomain, catcodomain

# rank-2 tensor
export DiagonalMap, Diagonal, diagonalmaptype

export OrthogonalFactorizationAlgorithm, QR, QRpos, QL, QLpos, LQ, LQpos, RQ, RQpos,
        SVD, SDD, Polar

# tensor operations
export @tensor
export scalar, add!, contract!

# truncation schemes
export TruncationScheme, NoTruncation, TruncationError, TruncationDimension, TruncationSpace, TruncationCutoff, TruncationDimCutoff
export notrunc, truncerr, truncdim, truncspace, truncbelow, truncdimcutoff

using TupleTools
using HalfIntegers
using WignerSymbols

using VectorInterface

# using LRUCache
using Strided

using TensorOperations: TensorOperations, @tensor
using TensorOperations: IndexTuple, Index2Tuple, linearize, Backend
const TO = TensorOperations

using Base:  @boundscheck, @propagate_inbounds, OneTo, tail, front, tuple_type_head, 
            tuple_type_tail, tuple_type_cons, SizeUnknown, HasLength, HasShape, IsInfinite,
            EltypeUnknown, HasEltype


import LinearAlgebra: ×
using LinearAlgebra: LinearAlgebra
using LinearAlgebra: norm, dot, normalize, normalize!, tr,
                        axpy!, axpby!, lmul!, rmul!, mul!,
                        adjoint, adjoint!, transpose, transpose!,
                        pinv, sylvester, triu!,
                        eigen, eigen!, svd, svd!,
                        isposdef, isposdef!, ishermitian,
                        Diagonal, Hermitian

# const IndexTuple{N} = NTuple{N, Int}


# auxiliary
include("auxiliary/dicts.jl")
include("auxiliary/linalg.jl")
include("auxiliary/misc.jl")


#--------------------------------------------------------------------
# experiment with different dictionaries
const SectorDict{K, V} = SortedVectorDict{K, V}
const FusionTreeDict{K, V} = Dict{K, V}
#--------------------------------------------------------------------

# Exception types:
#------------------
abstract type TensorException <: Exception end

# Exception type for all errors related to sector mismatch
struct SectorMismatch{S<:Union{Nothing, String}} <: TensorException
    message::S
end
SectorMismatch()=SectorMismatch{Nothing}(nothing)
Base.show(io::IO, ::SectorMismatch{Nothing}) = print(io, "SectorMismatch()")
Base.show(io::IO, e::SectorMismatch) = print(io, "SectorMismatch(", e.message, ")")

# Exception type for all errors related to vector space mismatch
struct SpaceMismatch{S<:Union{Nothing, String}} <: TensorException
    message::S
end
SpaceMismatch()=SpaceMismatch{Nothing}(nothing)
Base.show(io::IO, ::SpaceMismatch{Nothing}) = print(io, "SpaceMismatch()")
Base.show(io::IO, e::SpaceMismatch) = print(io, "SpaceMismatch(", e.message, ")")

# Exception type for all errors related to invalid tensor index specification.
struct IndexError{S<:Union{Nothing, String}} <: TensorException
    message::S
end
IndexError()=IndexError{Nothing}(nothing)
Base.show(io::IO, ::IndexError{Nothing}) = print(io, "IndexError()")
Base.show(io::IO, e::IndexError) = print(io, "IndexError(", e.message, ")")

# typerepr
type_repr(T::Type) = repr(T)

# Definitions and methods for superselection sectors (quantum numbers)
#----------------------------------------------------------------------
include("sectors/sectors.jl")

# Constructing and manipulating fusion trees and iterators thereof
#------------------------------------------------------------------
include("fusiontrees/fusiontrees.jl")

# Definitions and methods for vector spaces
#-------------------------------------------
include("spaces/vectorspaces.jl")

# # Definitions and methods for tensors
# #-------------------------------------
# # general definitions
include("tensors/abstracttensor.jl")
include("tensors/tensortreeiterator.jl")
include("tensors/tensor.jl")
include("tensors/diagonaltensor.jl") # specialization for rank-2 tensor
include("tensors/adjoint.jl")
include("tensors/linalg.jl")
include("tensors/vectorinterface.jl")
include("tensors/tensoroperations.jl")
include("tensors/indexmanipulations.jl")
include("tensors/truncation.jl")
include("tensors/factorizations.jl")


end