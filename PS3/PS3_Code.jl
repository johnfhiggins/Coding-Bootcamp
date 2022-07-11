cd(dirname(@__FILE__())) 
##Problem 1
using CSV
using DataFrames

lwage_pre = DataFrame(CSV.File("lwage.csv", header=0, type=Float64))
lwage_mat = Matrix(lwage_pre)
exp2 = lwage_mat[:,3].^2
lwage = hcat(lwage_mat, exp2)

using Optim, NLSolversBase, Random
using LinearAlgebra: diag
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
#function which takes initial earnings, savings, and saving rule as well as a sequence of returns and raises and outputs the amount of savings and whether the goal was satisfied
function growth_path(earn::Float64, saved::Float64, prop::Float64, return_seq::Vector{Float64}, raise_seq::Vector{Float64})
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

#function which determines whether the goal is attained in the deterministic setting. Note that in order to use the binary search function for both test functions, I have to pass the distribution functions to this one too. I don't actually use them in this function, and there is probably a more efficient way. I just didn't want to write duplicative functions
function tester_determ(sims::Int64, earn::Float64, saved::Float64, prop::Float64, ret_d::Normal{Float64}, raise_d::Uniform{Float64})
    #records the wealth level and outcome (note: because it is a deterministic process, only one simulation is needed)
    vals, successes = growth_path(earn, saved, prop, fill(1.06, 37), fill(1.03, 37))
    #convert boolean to rate 
    rate = Float64(successes)
    rate,vals
end

#parallelize this function
function tester(sims::Int64, earn::Float64, saved::Float64, prop::Float64, ret_d::ArrayLikeVariate{0}, raise_d::ArrayLikeVariate{0})
    #create empty arrays for the wealth levels and success variable
    vals = zeros(sims)
    successes = zeros(sims)
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
    rate, vals
end

function binary_searcher(test_func, target::Float64, tol::Float64, sims::Int64, earn::Float64, saved::Float64, ret_d::Normal{Float64}, raise_d::Uniform{Float64})
    p_guess = 0.5
    p_low = 0.0
    p_high = 1.0
    #continue until difference between max and guess is lower than the tolerance parameter 
    while abs(p_high - p_guess) >= tol
        #find the success rate given the current guessed p
        succ_rate, val_data = test_func(sims, earn, saved, p_guess, ret_d, raise_d)
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
    #return the guessed p and the wealth outcomes associated with it
    p_guess, val_data
end

using Random, Distributions
Random.seed!(0);

#initialize distributions
return_dist = Normal(0.06, 0.06)
raise_dist = Uniform(0.0, 0.06)

#find savings rate required to achieve the goal in a deterministic setting 
binary_searcher(tester_determ, 1.0, 0.0001, 1, 100.0, 100.0, return_dist, raise_dist)
#find savings rate required to achieve the goal when returns and raises are random
binary_searcher(tester, 0.9, 0.0001, 10000, 100.0, 100.0, return_dist, raise_dist)

#find probability of attaining goal in random setting using the answer to part a (p = 0.1062)
succ_r, val_data = tester(10000,100.0, 100.0, 0.1062, return_dist, raise_dist)
#plot histogram of wealth using optimal saving rule 
using Plots
wealth_hist = histogram(val_data, title = "Histogram of wealth using P = 0.16998")
savefig(wealth_hist, "wealthhist.png")

#find probability of attaining goal in random setting using the correct savings level - will be very close to 90% 
succ_r, val_data = tester(10000,100.0, 100.0, 0.1699829, return_dist, raise_dist)

#creates plot of success rate vs p 
p_grid = collect(0.0:0.01:1.0)
np = length(p_grid)
succ_p = zeros(np)
for i=1:np
    succ_r, val_data = tester(10000,100.0, 100.0, p_grid[i], return_dist, raise_dist)
    succ_p[i] = succ_r
end
succ_plot = plot(p_grid, succ_p, title= "Success rate as function of P", xlabel="Proportion saved (P)", ylabel="Prob of success")
savefig(succ_plot, "succ_plot.png")

#Idea: allow P to change over time (numerical optimization) state vars: time to retirement, current wealth