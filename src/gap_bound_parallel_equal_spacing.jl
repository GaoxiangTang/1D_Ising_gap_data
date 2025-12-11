#=
Compute upper bound of energy gap for random-bond Ising model 
    in parity subspace
@author: Yukai Wu
=#
using LinearAlgebra
using Random
using FLoops
using Plots
include("GapBound.jl")
using .GapBound: lambdaND2, lambdaDD, gap_bound

function lambdaND_joint(n::Vector{Int}, J::Vector{Float64}, h::Float64
                    )::Matrix{Float64}
    sort!(n, rev=true)
    if length(n) > 1
        @assert all(isequal.(n[1:end-1] .- n[2:end], n[1] - n[2]))
    end
    # Return an upper-triangular matrix containing lambdaND2
    # for all subchains between n[1]-n[i]+1 and n[1]-n[i]+n[j] (i<=j)
    @assert length(J) == n[1] - 1
    xi = log.(abs.(J)) .- log(abs(h))
    u = cumsum(xi)
    pushfirst!(u, 0)

    umin = Matrix{Float64}(undef, n[1], n[1])
    umax = Matrix{Float64}(undef, n[1], n[1])
    @floop for i = 1 : n[1]
        umin[i, i] = u[i]
        umax[i, i] = u[i]
        for j = i + 1 : n[1]
            umin[i, j] = min(umin[i, j - 1], u[j])
            umax[i, j] = max(umax[i, j - 1], u[j])
        end
    end
    
    l = zeros(Float64, length(n), length(n))
    @floop for i = 1 : length(n), j = i : length(n)
        umin_sub = @view(umin[n[1]-n[i]+1, n[1]-n[i]+1:n[1]-n[i]+n[j]])
        umax_sub = @view(umax[n[1]-n[i]+1:n[1]-n[i]+n[j], n[1]-n[i]+n[j]])
        l[i, j] = h^2 * exp(-2 * maximum(umax_sub .- umin_sub))
    end
    return(l)
end

function test_lambdaND_joint(n::Int=100)
    J = randn(n - 1)
    h = exp(sum(log.(abs.(J))) / length(J))
    data1 = lambdaND_joint(collect(n:-1:1), J, h)
    for i = 1 : n
        data2 = lambdaND2(collect(n-i+1:-1:1), J[i:end], h)
        if !all(abs.(data1[i, i:end] .- data2) .< 1e-8)
            println("Error detected")
        end
    end
end

function gap_bound_equal_spacing(n::Int, d::Int, 
    J::Vector{Float64}, h::Float64)
    # Return a sequence of gaps for J[1:n-1], J[1:n-1-d], J[1:n-1-2d], etc
    @assert length(J) == n - 1
    J = @view(J[end:-1:1]) # reverse order to reuse the slower DD part
    r = collect(1 : d : n - 1) # equal spacing of d
    gap_list = zeros(Float64, length(r))
    lL = lambdaND_joint(r[end:-1:1], J[1:r[end]-1], h)
    lR = lambdaDD(n .- r, J[r[1]:end], h)
    for i = 1 : length(r)
        lmin = 2 * minimum(max.(lL[i, i:end], lR[end:-1:i]))
        e2 = 2 * sqrt(lmin)
        e1 = 2 * sqrt(lambdaND2([n - (i - 1) * d], J[1+(i-1)*d:end], h)[1])
        gap_list[i] = e1 + e2
    end
    return(gap_list)
end

function test_gap_equal_spacing(n_list::AbstractRange{Int}=10:10:500)
    J = randn(n_list[end] - 1)
    h = exp(sum(log.(abs.(J))) / length(J))
    gap1 = gap_bound_equal_spacing(n_list[end], n_list.step, 
                                J, h)[length(n_list):-1:1]
    for i = 1 : length(n_list)
        gap2 = gap_bound(n_list[i], J[n_list[i]-1:-1:1], h,
                        mode=0, k=1, dmax=10)
        if abs(gap1[i] - gap2) > 1e-8
            println("Error detected")
        end
    end
end

function main(n_list::AbstractRange{Int};
            coupling::String="deterministic", 
            filename::Union{String, Nothing}=nothing,
            visualize::Bool=true, sigma::Float64=1.0, checksmall::Int=5000)
    # checksmall can be set to compute small systems more accurately
    @assert coupling in ["deterministic" "uniform" "normal"]
    if coupling == "deterministic"
        x = collect(1 : n_list[end] * 10)
        J = sin.(x .* sin.(x))
    elseif coupling == "uniform"
        J = 2 .* rand(n_list[end] * 10) .- 1
    else
        J = 1 .+ sigma * randn(n_list[end] * 10)
    end
    # fix h at expectation value for 10 times system size
    h = exp(sum(log.(abs.(J))) / length(J))
    gap_list = gap_bound_equal_spacing(n_list[end], n_list.step, 
                                        J[1:n_list[end]-1], h
                                        )[length(n_list):-1:1]
    for i = 1 : length(n_list)
        if n_list[i] <= checksmall
            gap2 = gap_bound(n_list[i], J[n_list[i]-1:-1:1], h, mode=1)
            if abs(gap_list[i] / gap2 - 1) > 0.1
                println("Relative error above 0.1 detected at n=",
                        n_list[i], " with gap1=", gap_list[i],
                        " and gap2=", gap2)
                flush(stdout)
                gap_list[i] = gap2
            end
        end
    end
    if visualize
        fig = plot(sqrt.(n_list), log.(gap_list), 
            xlabel="\$\\sqrt{n}\$", ylabel="\$\\ln\\Delta E\$", 
            label="bound", linewitdh=2, markershape=:rect, color=:blue)
        display(fig)
    end
    if !isnothing(filename)
        f = open(filename * ".txt", "w")
        for i = 1 : length(n_list)
            print(f, n_list[i], ' ')
        end
        println(f)
        for i = 1 : length(n_list)
            print(f, gap_list[i], ' ')
        end
        println(f)
        close(f)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    n_list = eval(Meta.parse(ARGS[1])) # vector or range
    coupling = ARGS[2] # deterministic or random
    filename = ARGS[3]
    if coupling in ["normal" "uniform"]
        rseed = parse(Int, ARGS[4]) # random seed
        Random.seed!(rseed)
        suffix = "_random_" * ARGS[4]
    else
        suffix = ""
    end
    if coupling == "normal"
        sigma = parse(Float64, ARGS[5])
        suffix = "_sigma_" * ARGS[5] * suffix
    else
        sigma = 1.0
    end
    filename = filename * suffix
    main(n_list, coupling=coupling, filename=filename,
        visualize=false, sigma=sigma)
end