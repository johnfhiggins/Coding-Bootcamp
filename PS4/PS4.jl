##problem 1

function matrix_fill(N::Int64, coins::Array{Int64})
    #initialize an empty array of size N+1 x the amount of coins + 1
    #note: because I need a row of ones at the beginning (so that successful ways of making change get counted), the indices of everything will be shifted up by 1. It's not pretty, but it works
    res = zeros(N+1, length(coins)+1)
    #create the aforementioned row of ones
    res[1, :] .= 1
    #sort the list of coins we have, because the way I have written it depends on the list of coins being ordered
    coins = sort(coins)
    #iteratively fill in the rows of the matrix
    for val=1:N
        #iterate over possible coins to use
        for (i, coin) in enumerate(coins)
            if coin > val #if the coin is larger than the remaining balance
                res[val+1,i+1] = res[val+1, i] #we cannot make change using this coin, so we use the previous count for the first i-1 coins
            elseif coin == val #if the coin is exactly the same as the remaining balance
                res[val+1,i+1] = 1 + res[val+1, i] #use this coin to make exact change and thus we need to increment the current count by 1
            else
                #can always make change with the lower denomination coins and ignore the new coin
                res[val+1,i+1] = res[val+1,i]
                #find the highest quantity of coin i that can be used
                j_range = Int(floor(val/coin))
                #loop over possible multiples of the current coin, up to j_range
                for j=1:j_range
                    #add the number of ways to make change for the resulting balance with the first i-1 coins and add to the current entry
                    res[val+1, i+1] += res[val + 1  - j*coin, i]
                end
            end
        end
    end
    #return the resulting matrix, as well as the total number of ways (it will be the entry in the bottom right corner of the matrix)
    res, res[N+1, length(coins)+1]
end

#find the total amount of ways
full_matrix, total_ways = matrix_fill(10, [2,3,5,6])
total_ways

##problem 2
#create bellman function for problem: takes rod length N and price vector P
function rod_bellman(N::Int64, P::Vector{Int64})
    #assign empty policy and value arrays
    value_mat = zeros(N)
    pol_func = zeros(N)
    #loop over state variables n
    for n=1:N
        #start with the candidate choice being to just sell the remaining rod and the candidate max being the value of the remaining
        cand_pol = n
        cand_max = P[n]
        #loop over all rod lengths to cut off; since n is the default, it is not included here
        for i=1:n-1
            #value of choosing length i to cut off is the price of a rod of length i plus the continuation value at n-i
            val = P[i] + value_mat[n-i]
            #if we get a value higher than the candidate max (note: it is possible that there is a tie between two different policies - this doesn't really matter though since we are interested in maximizing the value)
            if val >= cand_max
                #update the max and argmax candidates
                cand_max = val
                cand_pol = i
            end
        end
        #find the value of having a rod of length n by choosing the above candidate
        value_mat[n] = cand_max
        pol_func[n] = cand_pol
    end
    #return the value and policy vectors
    value_mat, pol_func
end

function rod_solver(P::Vector{Int64})
    #start with price vector corresponding to rod of length N
    N = length(P)
    #initialize empty policy sequence
    pol_seq = []
    #fill in the value function and policy vectors
    val, pol = rod_bellman(N, P)
    #use while loop to iteratively find the policy choices required to achieve the max value
    while N >= 1
        #find the policy function for a rod of length N
        cur_pol = Int(pol[N])
        #add the current policy to the policy sequence
        append!(pol_seq, cur_pol)
        #decrease N by the amount of the current policy
        N = N - cur_pol
    end
    #return the overall value as well as the sequence required to attain it
    val[length(P)],pol_seq
end

value_n, cuts = rod_solver( [1,5,8,9,10,17,17,20])
value_n
cuts

value_n, cuts = rod_solver( [1,5,45,9,10,17,17,20])
value_n
cuts

##problem 3
#note: I assume item array is ordered by ascending weight
function knap_bellman(values::Vector{Int64}, weights::Vector{Int64}, C::Int64)
    N = length(values)
    #create empty value array
    v_mat = zeros(N+1, C+1)
    #create empty weight array associated with the values in the value array
    w_mat = zeros(N+1,C+1)
    #create policy array which attains the values in the value array
    p_mat = fill([], N+1, C+1)
    for c=1:C+1
        for (i,item) in enumerate(values)
            #for convenience, define the weight and value of item i
            w_i = weights[i]
            v_i = values[i]
            if w_i > c #if i is not feasible, ignore it and set arrays to previous choice
                v_mat[i+1,c] = v_mat[i, c]
                w_mat[i+1,c] = w_mat[i,c]
                w_mat[i+1, c] = w_mat[i, c]
            elseif w_i ==  c #if item i is exactly the weight limit
                if values[i] >= v_mat[i,c] #if i is better than the previous optimal choice
                    if weights[i] <= w_mat[i,c] #if i weighs less than the previous optimal choice, set values, weights, and policy so that i is chosen at weight c
                        p_mat = [i]
                        v_mat[i+1, c] = v_i
                        w_mat[i+1, c] = w_i
                    end
                end
            else
                #start with i being the candidate optimal choice
                cand_max = v_i
                cand_weight = w_i
                cand_pol = [i]
                for j=1:i-1 #loop over possible choices of smaller subsets
                    v_ij = v_i + v_mat[j+1, c-w_i] #the value of choosing i and the optimal choice of items (1,...,j) with weight at most c - w_i
                    w_ij = w_i+ w_mat[j+1, c-w_i]#weight of the above
                    #println(v_ij, " ", w_ij, " ", values[i], " ", v_mat[j+1, c-w_i+1], w_ij, c)
                    if w_ij <= c
                        if v_ij >= cand_max #if the value of using i with the optimal bundle of items (1,...,j) of weight less than c-w_i is greater than the candidate maximum, set the candidates equal to this new combination
                            cand_max = v_ij
                            cand_weight = w_ij
                            cand_pol = p_mat[j+1, c-w_i]
                        end
                    end
                end
                #set value and weight for optimal bundle of weight less than c which potentially includes up to item i
                v_mat[i+1, c] = cand_max
                w_mat[i+1, c] = cand_weight

                if cand_pol ==[i] #if we are sticking with policy i
                    p_mat[i+1, c] = cand_pol #set the policy to i
                else #if we are combining i with an existing policy vector
                    new_pol = copy(cand_pol)
                    append!(new_pol, i) #append to previous policy vector
                    p_mat[i+1, c] = new_pol #create entry for new policy vector
                end
            end
        end
    end
    #return value and optimal policy vector
    v_mat[N+1, C+1], p_mat[N+1, C+1]
end

#determine value and optimal policies
knap_bellman([4,3,8], [1,1,2], 3)
knap_bellman([60,100,120], [10,20,30], 50)

##problem 4
using Parameters, Plots

@with_kw struct Primitives
    β::Float64 = 0.6 #discount rate
    a::Float64 = 20.0 #demand intercept
    b::Float64 = 2.0 ##demand slope

    alph::Float64 = 0.95 #research parameter, factor by which marginal cost is reduced

    c0::Float64 = 20.0 #max cost    
    c_grid::Vector{Float64} = collect(range(0.01, length=500, stop=c0)) #cost grid
    nc::Int64 = length(c_grid) #number of grid points
    #grid_gen(c_0, alph, nc)
end
mutable struct Results
    val_func::Array{Float64} #value function
    pol_func::Array{Float64} #policy function
end

function adj_cost(ct::Float64, ct1::Float64, alph::Float64)
    rat = ct1/ct #find ratio of future c to current c
    if rat >= 1 #if future cost greater than current
        val = 0 #no adjustment cost (no investment needed to produce at previous cost)
    else
        val = log(alph, rat) #find the amount which needs to be invested to achieve future cost
    end
    val #return adjustment cost
end

function obj_func(ct::Float64, ct1::Float64, a::Float64, b::Float64, alph::Float64)
    q = ((a-ct)/(2b)) #optimal quantity in each period
    rev = q*(a - b*q - ct) #profit at optimal quantity
    cost = adj_cost(ct, ct1, alph) #cost of new investment to get to next period cost ct1
    val = rev - cost #profit net investment cost
    if val < 0 ## impose condition that investment must be lower than current revenue by giving very low value when it is violated
        val = -10000
    end
    val
end
    #@unpack 

function Bellman(prim::Primitives, res::Results, alph::Float64)
    @unpack β, a, b,  c0, c_grid, nc = prim
    v_new = zeros(nc) #new policy function array
    for ci=1:nc #loop over cost index 
        c = c_grid[ci] #find corresponding cost
        cand_max = -50.0 #start with really bad value for max
        for i=1:nc #loop over index of next period's cost
            ct1 = c_grid[i] #find corresponding next period cost
            val = obj_func(c, ct1, a, b, alph) + β*res.val_func[i] #find value of choosing ct1 
            if val >= cand_max #check if this is better than the previous one; if so, update max candidates and policy function
                cand_max = val
                res.pol_func[ci] = ct1
            end
        end
        v_new[ci] = cand_max #fill in value function
    end
    v_new #return new value function
end


function model_solver(alpha::Float64)
    prim = Primitives()
    @unpack nc, c_grid, c0 = prim
    val_func, pol_func = zeros(nc), zeros(nc) #init blank value and policy function arrays 
    res = Results(val_func, pol_func) 
    tol = 0.0001 #tolerance param for convergence
    N = 1000 #max iterations
    n=0 
    error = 100 #starting error
    while error > tol && n < N #loop until convergence or max iterations reached
        n +=1
        v_new = Bellman(prim, res, alpha) #find new value function
        error = maximum(abs.(v_new - res.val_func)) #max difference between old value function and new value function
        println("Iteration ", n, ", error = ", error)
        res.val_func = v_new #update value function
    end
    println("Convergence!")

    #vfplot = plot(c_grid, res.val_func, title="Value") #plot value function
    #pfplot = plot(c_grid, [c_grid, res.pol_func], labels=[ "45 degree line" "Policy function"],title="Policy", ylims=(0,Int(c0))) #plot policy function and 45 degree line
    #display(vfplot)
    #display(pfplot)
    res.val_func, res.pol_func
end



function multiple_plots()
    @unpack c_grid , β = Primitives()
    alpha_list = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    #vf_arr = fill([], 5)
    #pf_arr = fill([], 5)
    vf_plot = plot()
    pf_plot = plot(c_grid, c_grid, title="Value function, selected α", xlabel="Cost", ylabel="Next period cost", labels="45 degree line", legend=:topleft)
    for (i,al) in enumerate(alpha_list)
        vfa, pfa = model_solver(al)
        plot!(vf_plot, c_grid, vfa, title="Value function, selected α, β = $(β)", xlabel="Cost", ylabel="V(c)", labels=al, legend=:bottomleft)
        plot!(pf_plot, c_grid,  pfa, title="Policy function, selected α, β = $(β)", xlabel="Cost", ylabel="Next period cost (c')", labels = al, legend=:topleft)
        #vf_arr[i] = vfa
        #vf_arr[]
    end
    display(vf_plot)
    display(pf_plot)
    cd(dirname(@__FILE__())) 
    savefig(vf_plot, "vfplot beta=$(β).png")
    savefig(pf_plot, "pfplot beta=$(β).png")
end

function cost_evol()
    #cost evolution
    alph = 0.95
    @unpack c_grid, β = Primitives()
    vf, pf = model_solver(alph)
    
    ci = 450
    cs = [c_grid[ci]]
    vs = [vf[ci]]
    error = 100
    while error > 0.01
        c = c_grid[ci]
        ct1 = pf[ci]
        error = c - ct1
        push!(cs, ct1)
        ci = findfirst(isequal(ct1), c_grid)
        push!(vs, vf[ci])
    end
    plot_evol = plot(cs, title="Cost evolution with c_0 = 18, α=$(alph), β=$(β)", xlabel="Time", ylim=(0,20), ylabel="Cost")
    savefig(plot_evol, "costevol$(β).png")
    #plot(vs)
end



##stochastic version of previous model

using Parameters, Plots, Distributions, Interpolations

@with_kw struct Primitives_stoch
    β::Float64 = 0.95 #discount rate
    a::Float64 = 20.0 #demand intercept
    b::Float64 = 1.0 ##demand slope

    alph::Float64 = 0.4 #research parameter, factor by which marginal cost is reduced
    p::Float64 = 0.5 #binomial parameter, probability that unit of research funding is successful

    c0::Float64 = 20.0 #max cost    
    c_grid::Vector{Float64} = collect(range(0.0, length=100, stop=c0)) #cost grid
    x_grid::Vector{Float64} = collect(range(0.0, length = 100, stop= 5.0)) #investment grid
    nc::Int64 = length(c_grid) #number of grid points
    nx::Int64 = length(x_grid)
end
mutable struct Results_stoch
    val_func::Array{Float64} #value function
    pol_func::Array{Float64} #policy function
end


function obj_func_st(ct::Float64, x::Float64, a::Float64, b::Float64)
    q = ((a-ct)/(2b))
    rev = q*(a - b*q - ct) 
    val = rev - x
    #if val < 0 ## impose condition that investment must be lower than current revenue
    #    val = -10000
    #end
    val
end

function succ_prob(x_i::Int64, x_s::Int64, p::Float64)
    dist = Binomial(x_i, p)
    val = pdf(dist, x_s)
    #val = binomial(x_i, x_s)*p^(x_s)*(1-p)^(x_i - x_s)
    val
end

function cost_reduce(c::Float64, x_new::Float64, alph::Float64)
    ct1 = alph^(x_new)*c
    ct1
end

function next_highest(x_arr::Vector{Float64}, x::Float64)
    #create boolean array where 1 indicates x is less than y and 0 if y is less than x
    compare_arr = [isless(x,y) for y in x_arr]
    #find index of first y such that x < y
    i_first = findfirst(compare_arr)
    #find the corresponding value in x_arr
    x_first = x_arr[i_first]
    x_first, Int(i_first)
end

function Bellman(prim::Primitives_stoch, res::Results_stoch)
    @unpack β, a, b, alph, p, c_grid, x_grid, nc, nx = prim
    v_new = zeros(nc)
    interp_val_func = extrapolate(interpolate(res.val_func, BSpline(Linear())), Flat()) #interpolate value function
    #need to search over x grid and take expectation of value at each point
    for ci=1:nc #loop over current cost index
        c = c_grid[ci]
        #x_grid_f = x_filter(xg, c) #filter 
        cand_max = -50.0
        for x_i=1:nx
            x = x_grid[x_i] #find corresponding x to invest
            val = obj_func_st(c, x, a, b)
            for x_s0=1:x_i #loop over number of successes; in x_i research draws, this will be x_s = 0:x_i, need to add to play nicely with array indices
                x_s = x_s0 - 1 #shift down by 1
                prob = succ_prob(x_i, x_s, p) #find probability of each x_s
                x_new = x_grid[min(x_s+1, nx)]
                ct1 = c - cost_reduce(c, x_new, alph) #find the new cost given x_new units of research were successful
                #ct1_i = next_highest(c_grid, ct1) #find index of ct1 
                val += prob*β*interp_val_func(ct1) #res.val_func[ct1_i]
            end
            if val >= cand_max
                cand_max = val
                res.pol_func[ci] = x
            end
        end
        v_new[ci] = cand_max
    end
    v_new
end


function model_solver()
    prim = Primitives_stoch()
    @unpack nc, c_grid, c0 = prim
    val_func, pol_func = zeros(nc), zeros(nc)
    res = Results_stoch(val_func, pol_func)
    tol = 0.001
    N = 1000
    n=0
    error = 100
    while error > tol && n < N
        n +=1
        print("Bellman ", n)
        v_new = Bellman(prim, res)
        error = maximum(abs.(v_new - res.val_func))
        println("Iteration ", n, ", error = ", error)
        res.val_func = v_new
    end
    println("Convergence!")

    vfplot = plot(c_grid, res.val_func, title="Value")
    pfplot = plot(c_grid, [c_grid, res.pol_func], labels=[ "45 degree line" "Policy function"],title="Policy", ylims=(0,5))
    display(vfplot)
    display(pfplot)
    #convert policy to cost
    res.val_func, res.pol_func
end

vf, pf = model_solver()


