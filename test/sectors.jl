println("------------------------------------")
println("Sectors")
println("------------------------------------")
ti = time()
@timedtestset "Sector properties of $(type_repr(I))" for I in sectorlist
    Istr = type_repr(I)

    @testset "Sector $Istr: Basic properties" begin
        s = (randsector(I), randsector(I), randsector(I))
        @test eval(Meta.parse(sprint(show, I))) == I
        @test eval(Meta.parse(type_repr(I))) == I
        @test eval(Meta.parse(sprint(show,s[1]))) == s[1]
        @test @constinferred(hash(s[1])) == hash(deepcopy(s[1]))
        @test @constinferred(one(s[1])) == @constinferred(one(I))
        @constinferred dual(s[1])
        @constinferred dim(s[1])
        @constinferred frobeniusschur(s[1])
        @constinferred Nsymbol(s...)
        @constinferred Rsymbol(s...)
        @constinferred Bsymbol(s...)
        @constinferred Fsymbol(s..., s...)
        it = @constinferred s[1] ⊗ s[2]
        @constinferred ⊗(s..., s...)
    end
    @testset "Sector $Istr: Value iterator" begin
        @test eltype(values(I)) == I
        sprev = one(I)
        for (i, s) in enumerate(values(I))
            @test !isless(s, sprev) # confirm compatibility with sort order
            if Base.IteratorSize(values(I)) == Base.IsInfinite() && I <: ProductSector
                @test_throws ArgumentError values(I)[i]
                @test_throws ArgumentError findindex(values(I), s)
            elseif hasmethod(Base.getindex, Tuple{typeof(values(I)), Int})
                @test s == @constinferred (values(I)[i])
                @test findindex(values(I), s) == i
            end
            sprev = s
            i >= 10 && break
        end
        @test one(I) == first(values(I))
        if Base.IteratorSize(values(I)) == Base.IsInfinite() && I <: ProductSector
            @test_throws ArgumentError findindex(values(I), one(I))
        elseif hasmethod(Base.getindex, Tuple{typeof(values(I)), Int})
            @test (@constinferred findindex(values(I), one(I))) == 1
            for s in smallset(I)
                @test (@constinferred values(I)[findindex(values(I), s)]) == s
            end
        end
    end
    
    @testset "Sector $I: fusion tensor and F-move and R-move" begin
        for a in smallset(I), b in smallset(I)
            for c in ⊗(a,b)
                X1 = permutedims(fusiontensor(a, b, c), (2, 1, 3))
                X2 = fusiontensor(b, a, c)
                l =dim(a)*dim(b)*dim(c)
                R = LinearAlgebra.transpose(Rsymbol(a, b, c))
                sz = (l, convert(Int, Nsymbol(a, b, c)))
                @test reshape(X1, sz) ≈ reshape(X2, sz) * R
            end
        end
        for a in smallset(I), b in smallset(I), c in smallset(I)
            for e in ⊗(a, b), f in ⊗(b, c)
                for d in intersect(⊗(e, c), ⊗(a, f))
                    X1 = fusiontensor(a, b, e)
                    X2 = fusiontensor(e, c, d)
                    Y1 = fusiontensor(b, c, f)
                    Y2 = fusiontensor(a, f, d)
                    @tensor f1 = conj(Y2[a,f,d])*conj(Y1[b,c,f])*
                                    X1[a,b,e] * X2[e,c,d]
                    f2 = Fsymbol(a,b,c,d,e,f)*dim(d)
                    @test isapprox(f1, f2; atol = 1e-12, rtol = 1e-12)
                end
            end
        end
    end

    @testset "Sector $Istr: Unitarity of F-move" begin
        for a in smallset(I), b in smallset(I), c in smallset(I)
            for d in ⊗(a,b,c)
                es = collect(intersect(⊗(a,b), map(dual, ⊗(c,dual(d)))))
                fs = collect(intersect(⊗(b,c), map(dual, ⊗(dual(d),a))))
                @test length(es) == length(fs)
                F = [Fsymbol(a,b,c,d,e,f) for e in es, f in fs]
                @test isapprox(F'*F, one(F); atol = 1e-12, rtol = 1e-12)
            end
        end
    end
    @testset "Sector $Istr: Pentagon equation" begin
        for a in smallset(I), b in smallset(I), c in smallset(I), d in smallset(I)
            for f in ⊗(a,b), h in ⊗(c,d)
                for g in ⊗(f,c), i in ⊗(b,h)
                    for e in intersect(⊗(g,d), ⊗(a,i))
                        p1 = Fsymbol(f,c,d,e,g,h) * Fsymbol(a,b,h,e,f,i)
                        p2 = zero(p1)
                        for j in ⊗(b,c)
                            p2 += Fsymbol(a,b,c,g,f,j) *
                                    Fsymbol(a,j,d,e,g,i) *
                                    Fsymbol(b,c,d,i,j,h)
                        end
                        @test isapprox(p1, p2; atol = 1e-12, rtol = 1e-12)
                    end
                end
            end
        end
    end
    @testset "Sector $Istr: Hexagon equation" begin
        for a in smallset(I), b in smallset(I), c in smallset(I)
            for e in ⊗(c,a), g in ⊗(c,b)
                for d in intersect(⊗(e,b), ⊗(a,g))
                    p1 = Rsymbol(c,a,e)*Fsymbol(a,c,b,d,e,g)*Rsymbol(b,c,g)
                    p2 = zero(p1)
                    for f in ⊗(a,b)
                        p2 += Fsymbol(c,a,b,d,e,f)*Rsymbol(c,f,d)*Fsymbol(a,b,c,d,f,g)
                    end
                    @test isapprox(p1, p2; atol = 1e-12, rtol = 1e-12)
                end
            end
        end
    end
    
end
tf = time()
printstyled("Finished sector tests in ",
            string(round(tf-ti; sigdigits=3)),
            " seconds."; bold = true, color = Base.info_color())
println()
