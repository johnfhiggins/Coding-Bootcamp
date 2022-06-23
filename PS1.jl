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
        print("Invalid input! Needs to be non-negative integer")
    end
end

print(facto(3))

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
print(p(coeff,x))

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
    print(String("$(n): $(approx_n(n)) \n"))
end
#=


n = 100000
succ = 0
for i in 0:n
    x = rand(2,1)
    if sq(x) <= 1
        succ+=1
    end
end
pi_approx = (4*succ)/n
print(pi_approx)

#=
