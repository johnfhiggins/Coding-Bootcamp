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

##
