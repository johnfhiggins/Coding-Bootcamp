##problem 1
function f(vec::Vector{Float64})
    #break vector input into x and y components
    x = vec[1]
    y = vec[2]
    #evaluate function
    val =(x^2 + y - 11)^2 + (x + y^2 - 7)^2
    val
end
#create domain grid and matrix for z values
x_grid = collect(-4.0:0.01: 4.0)
nx = length(x_grid)
z_grid = zeros(nx,nx)
#loop over x and y grids
for i=1:nx, j=1:nx
    #evaluate function at corresponding grid values and set z_grid equal to function value
    z_grid[i,j] = f(x_grid[i], x_grid[j])
end

#plot and save figure
plot_1a = Plots.surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera=(25,70))
savefig(plot_1a, "plot1a.png")

guess = [0.0, 0.0]
#create gradient function
function g(G, guess::Vector{Float64})
    #split guess vector into component parts
    x,y = guess[1], guess[2]
    G[1] = 4*x*(x^2 + y - 11) + 2(x + y^2 - 7)
    G[2] = 2*(x^2+y-11) + 4*y*(x + y^2 - 7)
    G
end
#create Hessian function
function h(H, guess::Vector{Float64})
    #split guess vector into component parts
    x,y = guess[1], guess[2]
    #define hessian matrix values
    H[1,1] = 12*x^2 + 4*y - 42
    H[1,2] = 4x + 4y
    H[2,1] = 4x + 4y
    H[1,1] = 4x + 12y^2 - 26
    H
end

using Optim
#create guesses (should have done a loop)
guess1 = [-4.0,-4.0]
guess2 = [4.0,-4.0]
guess3 = [4.0,4.0]
guess4 = [-4.0,4.0]

#find minima given initial guesses
opt1 = optimize(f, g,h, guess1)
opt2 = optimize(f, g,h, guess2)
opt3 = optimize(f, g,h, guess3)
opt4 = optimize(f, g,h, guess4)
println(opt1.minimizer)
println(opt2.minimizer)
println(opt3.minimizer)
println(opt4.minimizer)

opt1n = optimize(f, guess1)
opt2n = optimize(f, guess2)
opt3n = optimize(f, guess3)
opt4n = optimize(f, guess4)


##problem 2
#Ackley function
function Ackley(vec::Vector{Float64})
    #take vector and split into x and y components
    x,y = vec[1], vec[2]
    #evaluate function
    val = -20*exp(-0.2*sqrt(0.5*(x^2+y^2))) - exp(0.5(cos(2*pi*x) + cos(2*pi*y)))+ exp(1) + 20
    val
end

#create x_grid and 2d grid for function values
x_grid = collect(-4.0:0.01: 4.0)
nx = length(x_grid)
z_grid = zeros(nx,nx)
#loop through x and y arrays and evaluate function at each point then store in z_grid
for i=1:nx, j=1:nx
    vec = [x_grid[i], x_grid[j]]
    z_grid[i,j] = Ackley(vec)
end

#plot and save the surface and contour plots
plot_2a_surf = Plots.surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera=(25,70))
plot_2a_cont = Plots.contourf(x_grid, x_grid, z_grid, seriescolor=:viridis)

savefig(plot_2a_surf, "plot2a_surf.png")
savefig(plot_2a_cont, "plot2a_cont.png")


#initialize array of guesses to try
guesses = [[0.1, 0.1], [2,1], [2, 1.5],[0, 1], [2,2], [2.2,2.2], [3,3.2]]
#loop through guesses in array
for guess in guesses
    println(guess)
    #run both optimization algorithms
    opt_nm = optimize(Ackley, guess)
    opt_lb = optimize(Ackley, guess, LBFGS())
    #print results
    println("NM",opt_nm.minimizer, opt_nm.iterations)
    println("LBFGS",opt_lb.minimizer, opt_lb.iterations)
end

##problem 3
function Rastrigin(vec::Vector{Float64})
    #start with the constant term
    val = 10*length(vec)
    #summation term: iteratively add the x_i terms
    for x_i in vec
        val += x_i^2 - 10*cos(2*pi*x_i)
    end
    val
end

#create arrays for domain and function values
x_grid = collect(-5.12:0.01:5.12)
nx = length(x_grid)
z_grid = zeros(nx)
#loop over domain and evaluate function at each point
for i=1:nx
    z_grid[i] = Rastrigin([x_grid[i]])
end
using Plots
#plot and save figure
plot_3a = plot(x_grid, z_grid)
savefig(plot_3a, "plot3a.png")

#create array of zeros for plotting
z_grid = zeros(nx,nx)
#loop over x and y and evaluate function
for i=1:nx, j = 1:nx
    #create vector to feed into the function
    vec = [x_grid[i], x_grid[j]]
    #evaluate function
    z_grid[i,j] = Rastrigin(vec)
end

#plot and save the surface and contour plots
plot_3b_surf = Plots.surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera=(25,70))
plot_3b_cont = Plots.contourf(x_grid, x_grid, z_grid, seriescolor=:viridis)
savefig(plot_3b_surf, "plot3asurf.png")
savefig(plot_3b_cont, "plot3acont.png")

using Optim
#initialize array of guesses to try
guesses = [[0.1, 0.1], [0.5,0.5], [1,1],[1, 0], [0,1], [0,0.5], [0.5,0], [2.0, 2.0], [2.2,2.2], [3.0, 3.2]]

#loop through guesses in array
for guess in guesses
    println(guess)
    #run both optimization algorithms
    opt_nm = optimize(Rastrigin, guess)
    opt_lb = optimize(Rastrigin, guess, LBFGS())
    #print results
    println("NM",opt_nm.minimizer, opt_nm.iterations)
    println("LBFGS",opt_lb.minimizer, opt_lb.iterations, "\n")
end

##problem 4
function eval_func(f, x_arr::Vector{Float64})
    nx = length(x_arr)
    z_grid = zeros(nx)
    for i=1:nx #loop over domain
        z_grid[i] = f(x_arr[i]) #fill in function values
    end
    z_grid
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

function lin_approx(f, a::Float64, b::Float64, n::Int64, x::Float64)
    if x <= a
        val = f(a)
        return val
    elseif x >= b
        val = f(b)
        return val
    else
        #create domain and empty array for the function values
        x_grid = collect(range(a, length = n , stop= b))
        z_grid = eval_func(f,x_grid)
        x_high, i_high = next_highest(x_grid, x)
        x_low = x_grid[Int(i_high-1)]
        x_interp = ((x-x_low)/(x_high - x_low))*f(x_high) + ((x_high-x)/(x_high - x_low))*f(x_low)
        return x_interp
    end
end

f(x) = x^2

lin_approx(f, 0.0, 20.0, 10, 1.3)

fine_x = collect(0.0:0.1:20.0)
nx = length(fine_x)
fine_z = zeros(nx)
fine_z_act = zeros(nx)
for i=1:nx
    fine_z[i] = lin_approx(f, 0.0, 20.0, 10, fine_x[i])
    fine_z_act[i] = f(fine_x[i])
end
using Plots
plot(fine_x, fine_z)