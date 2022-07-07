##problem 1
n = 4

function facto(n)
    y = 1
    if n >= 0 && isinteger(n)
        for i in 1:n
            y = y*i
        end
        return y
    else
        println("Invalid input! Needs to be non-negative integer")
    end
end

println(facto(3))

##problem 2
function p(coeff,x)
    y = 0
    for (i,a_i) in enumerate(coeff)
        y += a_i*x^(i-1)
    end
    return y
end

coeff = [1,2,3]
x = 2
println(p(coeff,x))

##problem 3

function sq(x)
    y = 0
    for (index, value) in enumerate(x)
        y += value^2
    end
    return y
end

function approx_n(n)
    succ = 0
    for i in 0:n
        x = rand(2,1)
        if sq(x) <=1
            succ +=1
        end
    end
    return 4*succ/n
end

n_range = [100, 1000, 10000, 100000,1000000,10000000]
for n in n_range
    println(String("$(n): $(approx_n(n))"))
end

##problem 4

n = 50
w_len = 200
x1 = randn(n)
x12 = x1.^2
x2 = randn(n)
X = hcat(x1, x12, x2, ones(n))
coeff = zeros(4,w_len)
for i in 1:w_len
    w = randn(n)
    Y = 0.1.*x1 + 0.2.*x12 + 0.5.*x2 + ones(n) + 0.1.*w
    β = inv(X'*X)*X'*Y
    pred = X*β
    for j in 1:4
        coeff[j,i] = β[j]
    end
end
using Plots
using Distributions
hista = histogram(coeff[1,:], bins=30, title="histogram of a")
histb= histogram(coeff[2,:], bins=30, title="histogram of b")
histc= histogram(coeff[3,:], bins=30, title="histogram of c")
histd = histogram(coeff[4,:], bins=30, title="histogram of d")
savefig(hista, "hista.png")
savefig(histb, "histb.png")
savefig(histc, "histc.png")
savefig(histd, "histd.png")

##problem 5
function rw_step(x, t, a)
    ε = randn()
    if t < 200
        x1 = a*x + 0.2*ε
    else
        x1 = 0
    end
    return x1
end

function first_pass(x,t,a)
    while x > 0
        t +=1
        x = rw_step(x,t,a)
    end
    return t
end

function sample_collect(a)
    hit = zeros(100)
    for i in 1:100
        x=1
        t=0
        hit[i] = first_pass(x,t,a)
    end
    return hit
end

using Statistics
a_list = [0.8, 1, 1.2]
a_mean = zeros(3)
for (index, a) in enumerate(a_list)
    a_hit = sample_collect(a)
    hist_a = histogram(a_hit, bins=30, label="histogram of T_0, a = $(a)", xlabel="T_0", ylabel = "Count")
    savefig(hist_a, "t0 hist, a = $(a).png")
    a_mean[index] = mean(a_hit)
end
mean_plot = plot(a_list, a_mean, seriestype = :scatter, title="Plot of mean of T_0 for selected alphas", xlabel="alpha", ylabel="mean T_0")
savefig(mean_plot, "mean_plot.png")

##problem 6
function f_x(x)
    return (x-1)^3
end
function f_pr(x)
    return 3*(x-1)^2
end

function f_x2(x)
    return 10 + x - x^2
end

function f_pr2(x)
    return 1-2x
end

function root_finder(f, f_prime, x_0, tol, maxiter)
    x = x_0
    t = 1
    while (f(x))/(f_prime(x)) > tol && t <= maxiter
        x = x - (f(x))/(f_prime(x))
        println(x)
        t+=1
    end
    return x
end

root_finder(f_x, f_pr, 2, 0.0001, 1000)
root_finder(f_x2, f_pr2, 8, 0.0001, 10000)


##problem 7
