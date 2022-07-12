cd(dirname(@__FILE__())) 
##Problem 1
using CSV
using DataFrames

lwage_pre = DataFrame(CSV.File("lwage.csv", header=0, type=Float64))
lwage_mat = Matrix(lwage_pre)
exp2 = lwage_mat[:,3].^2
lwage = hcat(lwage_mat, exp2)

using Distributed

addprocs(19)
@everywhere using Optim, NLSolversBase, Random
@everywhere using LinearAlgebra: diag
Random.seed!(0);

n = length(lwage[:,1])
X = hcat(ones(n), lwage[1:end, 1:end .!= 1])
Y = lwage[:, 1]
nvar = 4

function log_like(X, Y, beta, log_sigma)
    sig = exp(log_sigma)
    llike = -n/2*log(2π) - n/2* log(sig^2) - (sum((Y - X * beta).^2) / (2*sig^2))
    llike = -llike
end

func = TwiceDifferentiable(vars -> log_like(X, Y, vars[1:nvar], vars[nvar + 1]),
                           ones(nvar+1); autodiff=:forward);

opt = optimize(func, ones(nvar+1))

parameters = Optim.minimizer(opt)

parameters[nvar+1] = exp(parameters[nvar+1])

numerical_hessian = hessian!(func,parameters)

var_cov_matrix = inv(numerical_hessian)

β = parameters[1:nvar]

temp = diag(var_cov_matrix)
temp1 = temp[1:nvar]

t_stats = β./sqrt.(temp1)


function estimator(X_d::Array{Float64}, Y_d::Vector{Float64})
    n = length(Y_d)
    nvar = 4
    func = TwiceDifferentiable(vars -> log_like(X_d, Y_d, vars[1:nvar], vars[nvar + 1]),
                           ones(nvar+1); autodiff=:forward);

    opt = optimize(func, ones(nvar+1))

    parameters = Optim.minimizer(opt)

    parameters[nvar+1] = exp(parameters[nvar+1])

    β = parameters[1:nvar]
    β
end

function bootstrapper(X_d::Array{Float64}, Y_d::Array{Float64}, sims::Int64)    
    n = length(Y_d)
    n2 = Int(floor(n/2))
    b0, b1, b2, b3 = zeros(sims), zeros(sims), zeros(sims), zeros(sims)
    for j=1:sims
        Random.seed!(j);
        selected = rand(1:n, n2)
        X_b, Y_b = zeros(n2,4), zeros(n2)
        for i=1:n2
            ind = selected[i]
            X_b[i,:] = X_d[ind,:]
            Y_b[i,:] = Y_d[ind,:]
        end
        β_b = estimator(X_b, Y_b)
        b0[j], b1[j], b2[j], b3[j] = β_b[1], β_b[2], β_b[3], β_b[4]
        println(j/sims*100, "% complete!")
    end
    b0, b1, b2, b3
end

function bootstrapper_par(X_d::Array{Float64}, Y_d::Array{Float64}, sims::Int64)    
    n = length(Y_d)
    n2 = Int(floor(n/2))
    b0, b1, b2, b3 = zeros(sims), zeros(sims), zeros(sims), zeros(sims)
    for j=1:sims
        Random.seed!(j);
        selected = rand(1:n, n2)
        X_b, Y_b = zeros(n2,4), zeros(n2)
        for i=1:n2
            ind = selected[i]
            X_b[i,:] = X_d[ind,:]
            Y_b[i,:] = Y_d[ind,:]
        end
        β_b = estimator(X_b, Y_b)
        b0[j], b1[j], b2[j], b3[j] = β_b[1], β_b[2], β_b[3], β_b[4]
        println(j/sims*100, "% complete!")
    end
    b0, b1, b2, b3
end

b_0, b_1, b_2, b_3 = bootstrapper(X,Y, 100)
@elapsed bootstrapper(X,Y, 100)



#problem 2
using Plots
using Random

function draw_sim(peeps::Vector{Int64})
    #find the total number of slips to generate
    n = length(peeps)
    #since people draw at random without replacement, the process of drawing slips is exactly a random permutation of the range from 1 to n. This line finds one such permutation  
    slips_drawn = randperm(n)
    #this next line creates a vector whose components are 1 if that player's number matches their drawn slip and 0 otherwise
    match = peeps - slips_drawn .==0
    #take the sum of all successes to get the total number of successes
    successes = sum(match)
    #return the total number of successes in this simulation
    successes
end

function hist_gen(n::Int64, sim::Int64)
    #initialize array of people
    people = collect(1:1:n)
    #initialize a blank array for successes
    sim_data = zeros(sim)
    #iterate simulations
    for i=1:sim
        #store the number of successes in iteration i 
        sim_data[i] = draw_sim(people)
    end
    #return both the histogram and the mean number of matches (I included the mean because I was curious)
    hist,mean = histogram(sim_data), sum(sim_data)/sim
end

#generate histograms and means for n = 10 and 20 and save the histograms
hist_10,mean_10 = hist_gen(10, 10000)
savefig(hist_10, "hist10.png")
hist_20,mean_20 = hist_gen(20, 10000)
savefig(hist_20, "hist20.png")

mean_10
mean_20

##problem 3
using Distributed, SharedArrays, Distributions
addprocs(5)

@everywhere using SharedArrays, Distributions, Random

Random.seed!(0);

#function which takes initial earnings, savings, and saving rule as well as a sequence of returns and raises and outputs the amount of savings and whether the goal was satisfied
@everywhere function growth_path(earn::Float64, saved::Float64, prop::Float64, return_seq::Vector{Float64}, raise_seq::Vector{Float64})
    val = saved
    for i=1:37
        #Finds the total amount of current earnings that are saved in this period. I am assuming earnings that are not saved are consumed
        inv = prop*earn
        #Add amount invested in period t to the value of the portfolio. I assume earnings are invested at the start of the period - the wording in the question is ambiguous as to whether it is invested at the start or end of the period
        val += inv
        #find the value of the portfolio in the next period based on this period's return 
        val = val*return_seq[i]
        #find next period's earnings based on this period's raise percent
        earn = earn*raise_seq[i]
    end
    #create boolean variable which determines whether the goal of having 10x of earnings in savings by retirement is attained. I assume that this means 10x of the agent's earnings when they are 67, not 10x of their earnings when they are 30.
    success = val >= 10*earn
    val, success
end


@everywhere function tester(determ::Int64, sims::Int64, earn::Float64, saved::Float64, prop::Float64)
    #create empty arrays for the wealth levels and success variable
    vals = zeros(sims)
    successes = zeros(sims)
    if determ==1
        vals, successes = growth_path(earn, saved, prop, fill(1.06, 37), fill(1.03, 37))
        #convert boolean to rate 
        rate = Float64(successes)
        return rate,vals
    else
        return_dist = Normal(0.06, 0.06)
        raise_dist = Uniform(0.0, 0.06)
        #loop over simulation count
        for i=1:sims
            #draw new return and raise sequences following their specified distributions
            returns =1 .+ rand(return_dist, 37)
            raises = 1 .+ rand(raise_dist, 37)
            #store wealth level and outcome in their respective locations
            vals[i], successes[i] = growth_path(earn, saved, prop, returns, raises)
        end
        #sum outcomes and divide by number of simulations to get success rate
        rate = sum(successes)/sims
        return rate, vals
    end
end

#parallelized version of the above function, significant speed improvement for large number of simulations
@everywhere function tester2(determ::Int64, sims::Int64, earn::Float64, saved::Float64, prop::Float64)
    #check for whether we want the deterministic process or the random one
    if determ==1 #if we choose the deterministic process
        #find the ending wealth level and outcome variable
        vals, successes = growth_path(earn, saved, prop, fill(1.06, 37), fill(1.03, 37))
        #convert boolean to float 
        rate = Float64(successes)
        return rate,vals
    elseif determ==0
        #initialize empty shared arrays 
        vals = SharedArray{Float64}(sims,1)
        successes = SharedArray{Float64}(sims,1)
        #loop over simulation count
        @sync @distributed for i=1:sims
            #draw new return and raise sequences following their specified distributions
            returns =1 .+ rand(return_dist, 37)
            raises = 1 .+ rand(raise_dist, 37)
            #store wealth level and outcome in their respective locations
            vals[i], successes[i] = growth_path(earn, saved, prop, returns, raises)
        end
        #sum outcomes and divide by number of simulations to get success rate
        rate = sum(successes)/sims
        return rate, vals
    end
end

#@elapsed tester(10000000, 100.0, 100.0, 0.1, return_dist, raise_dist)
#@elapsed tester2(10000000, 100.0, 100.0, 0.1, return_dist, raise_dist)

@everywhere function binary_searcher(determ::Int64, target::Float64, tol::Float64, sims::Int64, earn::Float64, saved::Float64)
    p_guess = 0.5
    p_low = 0.0
    p_high = 1.0
    #continue until difference between max and guess is lower than the tolerance parameter 
    while abs(p_high - p_guess) >= tol
        #find the success rate given the current guessed p 
        succ_rate, val_data = tester2(determ, sims, earn, saved, p_guess)
        if succ_rate >= target #if success rate is too high, revise guess downward
            println("Too high!", p_guess, succ_rate)
            p_high = p_guess
            p_guess = (p_low + p_guess)/2 
        elseif succ_rate < target #if success rate is too low, revise guess upward
            println("Too low!", p_guess, succ_rate)
            p_low = p_guess
            p_guess = (p_high + p_guess)/2
        end
    end
    #return the guessed p
    p_guess
end

#create distributions
return_dist = Normal(0.06, 0.06)
raise_dist = Uniform(0.0, 0.06)

#find savings rate required to achieve the goal in a deterministic setting 
binary_searcher(1, 1.0, 0.0000001, 1, 100.0, 100.0)

#find probability of attaining goal in random setting using the answer to part a (p = 0.1062)
tester2(100000000,100.0, 100.0, 0.10626, return_dist, raise_dist)

#find savings rate required to achieve the goal when returns and raises are random
binary_searcher(0, 0.9, 0.0000001, 1000000, 100.0, 100.0)


#plot histogram of wealth using optimal saving rule 
using Plots
wealth_hist = histogram(val_data_wrong, title = "Histogram of wealth using P = 0.16957")
savefig(wealth_hist, "wealthhist.png")

#find probability of attaining goal in random setting using the correct savings level - will be very close to 90% 
succ_opt, val_data_opt = tester2(0,1000000,100.0, 100.0, 0.16957)

#creates plot of success rate vs p 
p_grid = collect(0.0:0.01:1.0)
np = length(p_grid)
succ_p = zeros(np)
for i=1:np
    succ_vals, val_datas = tester2(0,1000000,100.0, 100.0, p_grid[i])
    succ_p[i] = succ_vals
end
using Plots
succ_plot = plot(p_grid, succ_p, title= "Success rate as function of P", xlabel="Proportion saved (P)", ylabel="Prob of success")
savefig(succ_plot, "succ_plot.png")
#Idea: allow P to change over time (numerical optimization) state vars: time to retirement, current wealth