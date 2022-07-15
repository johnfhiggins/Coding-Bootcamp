##problem 1

function add_coin(bal::Int64, coin_add::Array{Int64}, used::Array{Int64})
    bal -= sum(coin_add)
    push!(used, coin_add)
    bal, used
end
    
function new_coin_picker(bal::Int64, coin_list::Array{Int64}, used::Array{Int64})
    rem_coins = delete!(coin_list, used)
    for i in rem_coins
        bal
    end
end

function possible_ways(val, coin_init)
    coin_list = copy(coin_init)
    options = Int.(floor.(val ./ coin_list)) 
    count = 0
    for (i,coin) in enumerate(coin_list)
        for j in 1:options[i]
            coin_j = copy(coin_list)
            if options[i] > 0
                new = val - coin*j
                println(val, "-" ,coin, "*", j, "=",  new )
                if new == 0
                    count += 1
                else
                    deleteat!(coin_j, i)
                    count += possible_ways(new, coin_j)
                end
            end
        end
    end
    return count
end

using Combinatorics
function searcher(bal::Int64, used::Array{Int64}, coin_list::Array{Int64}, curr_iter::Int64, results)
    for vals=1:bal
        #determine which coins could possibly make change
        possible_coins = [j for j in coin_list if j <= vals] 
        candidates = powerset(possible_coins)
    end
end

        #unused_possible = delete!(possible_coins, used)
        
N=5
coin_list=[1,2,5]


function matrix_fill(N::Int64, coins::Array{Int64})
    res = zeros(N+1, length(coins)+1)
    res[1, :] .= 1
    coins = sort(coins)
    for val=1:N
        for (i, coin) in enumerate(coins)
            if coin > val
                res[val+1,i+1] = res[val+1, i]
            elseif coin == val
                res[val+1,i+1] = 1 + res[val+1, i]
            else
                #can always make change with the lower denomination coins and ignore the new coin
                res[val+1,i+1] = res[val+1,i]
                #find the highest number of the new coin that can be used
                j_range = Int(floor(val/coin))
                println(val, coin, j_range)
                for j=1:j_range
                    res[val+1, i+1] += res[val + 1  - j*coin, i]
                end
            end
        end
    end
    res
end 
