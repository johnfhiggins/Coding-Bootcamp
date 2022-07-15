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