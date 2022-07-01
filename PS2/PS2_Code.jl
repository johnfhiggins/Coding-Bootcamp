##problem 1
f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
x_grid = collect(-4.0:0.01: 4.0)
nx = length(x_grid)
z_grid = zeros(nx,nx)
for i=1:nx, j=1:nx
    z_grid[i,j] = f(x_grid[i], x_grid[j])
end

plot_1a = Plots.surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera=(25,70))
savefig(plot_1a, "plot1a.png")
