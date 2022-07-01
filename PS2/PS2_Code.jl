##problem 1
function f(vec::Vector{Float64})
    x = vec[1]
    y = vec[2]
    val =(x^2 + y - 11)^2 + (x + y^2 - 7)^2
    val
end
x_grid = collect(-4.0:0.01: 4.0)
nx = length(x_grid)
z_grid = zeros(nx,nx)
for i=1:nx, j=1:nx
    z_grid[i,j] = f(x_grid[i], x_grid[j])
end

plot_1a = Plots.surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera=(25,70))
savefig(plot_1a, "plot1a.png")

guess = [0.0, 0.0]
#create gradient function
function g(G, guess::Vector{Float64})
    x,y = guess[1], guess[2]
    G[1] = 4*x*(x^2 + y - 11) + 2(x + y^2 - 7)
    G[2] = 2*(x^2+y-11) + 4*y*(x + y^2 - 7)
    G
end
#create Hessian function
function h(H, guess::Vector{Float64})
    x,y = guess[1], guess[2]
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
function Ackley(vec::Vector{Float64})
    x,y = vec[1], vec[2]
    val = -20*exp(-0.2*sqrt(0.5*(x^2+y^2))) - exp(0.5(cos(2*pi*x) + cos(2*pi*y)))+ exp(1) + 20
    val
end

x_grid = collect(-4.0:0.01: 4.0)
nx = length(x_grid)
z_grid = zeros(nx,nx)
for i=1:nx, j=1:nx
    vec = [x_grid[i], x_grid[j]]
    z_grid[i,j] = Ackley(vec)
end

plot_2a_surf = Plots.surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera=(25,70))
plot_2a_cont = Plots.contourf(x_grid, x_grid, z_grid, seriescolor=:viridis)

savefig(plot_2a_surf, "plot2a_surf.png")
savefig(plot_2a_cont, "plot2a_cont.png")

opt = optimize(Ackley, [0,0])
