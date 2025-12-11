#=
Compute upper bound of energy gap for random-bond Ising model 
    in parity subspace
@author: Yukai Wu
=#
using LinearAlgebra
using Random
using FLoops
using Plots

function lambdaND(n::Vector{Int}, J::Vector{Float64}, h::Float64
                    )::Vector{Float64}
    sort!(n, rev=true)
    @assert length(J) == n[1] - 1
    xi = log.(abs.(J)) .- log(abs(h))
    u = cumsum(xi)
    pushfirst!(u, 0)
    #expu = exp.(-2 * u)
    #tn = (1 ./ expu) .* cumsum(expu) / h^2
    #cumtn = @view(cumsum(@view(tn[end:-1:1]))[end:-1:1])
    ## For large n, u may become too large, leading to expu to be Inf
    ## To handle this, we work with log of tn and cumtn
    function logsum(lnA, lnB)
        # compute log(A+B)=log(exp(lnA)+exp(lnB))
        if lnA > lnB
            return(lnA + log(1 + exp(lnB - lnA)))
        else
            return(lnB + log(1 + exp(lnA - lnB)))
        end
    end
    lntn = 2 * u + accumulate(logsum, -2 * u) .- 2 * log(h)
    lncumtn = @view(accumulate(logsum, @view(lntn[end:-1:1]))[end:-1:1])
    l = zeros(Float64, length(n))
    l[1] = exp(reduce(logsum, -2 * u) - reduce(logsum, -2 * u + lncumtn))
    for i = 2 : length(n)
        u_new = @view(u[1:n[i]])
        lntn_new = @view(lntn[1:n[i]])
        lncumtn = @view(accumulate(logsum, @view(lntn_new[end:-1:1])
                                    )[end:-1:1])
        l[i] = exp(reduce(logsum, -2 * u_new) 
                    - reduce(logsum, -2 * u_new + lncumtn))
    end
    return(l)
end

function lambdaND_original(n::Int, J::Vector{Float64}, h::Float64
                    )::Float64
    # original formula from Phys. Rev. B 106, 064204 (2022)
    s = 0.0
    t = 0.0
    for i = 1 : n
        r = 1.0
        for j = 1 : i - 1
            r *= (h / J[j])^2
        end
        s += r
        u = 0
        for l = i : n
            for m = 1 : l
                v = 1 / h^2
                for j = m : l - 1
                    v *= (J[j] / h)^2
                end
                u += v
            end
        end
        t += r * u
    end
    return(s / t)
end

function lambdaND2(n::Vector{Int}, J::Vector{Float64}, h::Float64
                    )::Vector{Float64}
    sort!(n, rev=true)
    @assert length(J) == n[1] - 1
    xi = log.(abs.(J)) .- log(abs(h))
    u = cumsum(xi)
    pushfirst!(u, 0)
    
    emin = accumulate(min, u)
    l = zeros(Float64, length(n))
    @floop for i = 1 : length(n)
        u_new = @view(u[1:n[i]])
        emin_new = @view(emin[1:n[i]])
        emax = accumulate(max, @view(u_new[end:-1:1]))[end:-1:1]
        l[i] = h^2 * exp(-2 * maximum(emax .- emin_new))
    end
    return(l)
end

function test_lambdaND(n_list::Union{Vector{Int}, AbstractRange{Int}}
                       =[10; 20; 30; 40; 50])
    function exact(n, J, h)
        T = Tridiagonal(J.^2, -h^2 .- [0; J].^2, h^2 * ones(n - 1))
        l = minimum(eigvals(-T))
        return(l)
    end
    data1 = zeros(length(n_list))
    data2 = zeros(length(n_list))
    data3 = zeros(length(n_list))
    for i = 1 : length(n_list)
        J = randn(n_list[i] - 1)
        h = exp(sum(log.(abs.(J))) / (n_list[i] - 1))
        data1[i] = exact(n_list[i], J, h)
        data2[i] = lambdaND([n_list[i]], J, h)[1]
        data3[i] = lambdaND2([n_list[i]], J, h)[1]
    end
    fig = plot(n_list, [data1 data2 data3], 
               xlabel="\$n\$", ylabel="\$\\lambda\$",
               label=["exact" "bound_orignal" "bound2"], yaxis=:log,
               linewitdh=2, markershape=[:circle :rect :diamond], color=[:red :blue :green])
    display(fig)
end

function lambdaDD(n::Vector{Int}, J::Vector{Float64}, h::Float64;
                    )::Vector{Float64}
    sort!(n, rev=true)
    @assert length(J) == n[1]
    xi = log.(abs.(J[2:end])) .- log(abs(h))
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
    e1 = Matrix{Float64}(undef, n[1], n[1])
    @floop for i = 1 : n[1], j = i : n[1]
        e1[i, j] = exp(-2 * (umax[j, end] - umin[i, j]))
    end
    l = zeros(Float64, length(n))
    for i = 1 : length(n)
        umax_new = @view(umax[n[1]-n[i]+1:n[1], n[1]-n[i]+1:n[1]])
        umin_new = @view(umin[n[1]-n[i]+1:n[1], n[1]-n[i]+1:n[1]])
        e1_new = @view(e1[n[1]-n[i]+1:n[1], n[1]-n[i]+1:n[1]])
        tmp = [umax_new[1, 1]; umax_new[1, 1:end-1]]
        @floop for j = 1 : n[i], k = j : n[i]
            tmp2 = e1_new[j, k] + exp(-2 * (tmp[j] - umin_new[j, k]))
            @reduce() do (v=Inf; tmp2)
                if tmp2 < v
                    v = tmp2
                end
            end
        end
        l[i] = h^2 * v
    end
    return(l)
end

function test_lambdaDD(n_list::Union{Vector{Int}, AbstractRange{Int}}
                       =[10; 20; 30; 40; 50])
    function exact(n, J, h)
        T = Tridiagonal(J[2:end].^2, -h^2 .- J.^2, h^2 * ones(n - 1))
        l = minimum(eigvals(-T))
        return(l)
    end
    data1 = zeros(length(n_list))
    data2 = zeros(length(n_list))
    for i = 1 : length(n_list)
        J = randn(n_list[i])
        h = exp(sum(log.(abs.(J))) / length(J))
        data1[i] = exact(n_list[i], J, h)
        data2[i] = lambdaDD([n_list[i]], J, h)[1]
    end
    fig = plot(n_list, [data1 data2], 
               xlabel="\$n\$", ylabel="\$\\lambda\$",
               label=["exact" "bound"], yaxis=:log,
               linewitdh=2, markershape=[:circle :rect], color=[:red :blue])
    display(fig)
end

function gap_bound(n::Int, J::Vector{Float64}, h::Float64; 
                    mode::Int=1, k::Int=10, dmax::Int=100)
    @assert length(J) == n - 1
    @assert mode in [0 1]
    if mode == 0
        d = max(div(n - 2, k), 1) # divided into about k segments
        if dmax > 0 # set an upper bound on d
            d = min(d, dmax)
        end
        r = collect(1 : d : n - 1)
    else
        r = collect(1 : n - 1)
    end
    lL = lambdaND2(r[end:-1:1], J[1:r[end]-1], h)
    lR = lambdaDD(n .- r, J[r[1]:end], h)
    lmin = 2 * minimum(max.(lL, lR[end:-1:1]))
    e2 = 2 * sqrt(lmin)
    e1 = 2 * sqrt(lambdaND2([n], J, h)[1])
    return(e1 + e2)
end

function gap_exact(n, J, h)
    M = diagm(0 => ones(n) * h, -1 => J)
    e = svdvals(M)[end-1:end]
    return(sum(e))
end

function test_gap_rand(n_list::Union{Vector{Int}, AbstractRange{Int}}
                       =[10; 20; 30; 40; 50])
    data1 = zeros(length(n_list))
    data2 = zeros(length(n_list))
    for i = 1 : length(n_list)
        J = randn(n_list[i] - 1)
        h = exp(sum(log.(abs.(J))) / length(J))
        data1[i] = gap_exact(n_list[i], J, h)
        data2[i] = gap_bound(n_list[i], J, h)
    end
    fig = plot(n_list, [data1 data2], 
               xlabel="\$n\$", ylabel="\$\\Delta E\$",
               label=["exact" "bound"], yaxis=:log,
               linewitdh=2, markershape=[:circle :rect], color=[:red :blue])
    display(fig)
end

function test_gap_deterministic(n_list::Union{Vector{Int}, AbstractRange{Int}}
                                =[10; 20; 30; 40; 50])
    data1 = zeros(length(n_list))
    data2 = zeros(length(n_list))
    for i = 1 : length(n_list)
        x = collect(1 : n_list[i] - 1)
        J = sin.(x .* sin.(x))
        h = exp(sum(log.(abs.(J))) / length(J))
        data1[i] = gap_exact(n_list[i], J, h)
        data2[i] = gap_bound(n_list[i], J, h)
    end
    fig = plot(n_list, [data1 data2], 
               xlabel="\$n\$", ylabel="\$\\Delta E\$",
               label=["exact" "bound"], yaxis=:log,
               linewitdh=2, markershape=[:circle :rect], color=[:red :blue])
    display(fig)
end

function main(n_list::Union{Vector{Int}, AbstractRange{Int}};
            coupling::String="deterministic", 
            filename::Union{String, Nothing}=nothing,
            mode::Int=1, k::Int=0, dmax::Int=0, visualize::Bool=true, 
            sigma::Float64=1.0)
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

    gap_list = zeros(Float64, length(n_list))
    for i = 1 : length(n_list)
        gap_list[i] = gap_bound(n_list[i], J[1:n_list[i]-1], h, 
                                mode=mode, k=k, dmax=dmax)
        println(n_list[i], ' ', gap_list[i])
        flush(stdout)
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
        mode=0, k=100, dmax=100, visualize=false, sigma=sigma)
end