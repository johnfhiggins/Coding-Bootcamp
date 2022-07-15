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

##problem 2
#create bellman function for problem: takes rod length N and price vector P
function rod_bellman(N::Int64, P::Vector{Int64})
    #assign empty policy and value arrays
    value_mat = zeros(N)
    pol_func = zeros(N)
    #loop over state variables n 
    for n=1:N
        #start with the candidate choice being to just sell the remaining rod and the candidate max being the value of the remaining
        cand_pol = n
        cand_max = P[n]
        #loop over all rod lengths to cut off; since n is the default, it is not included here
        for i=1:n-1
            #value of choosing length i to cut off is the price of a rod of length i plus the continuation value at n-i
            val = P[i] + value_mat[n-i]
            #if we get a value higher than the candidate max (note: it is possible that there is a tie between two different policies - this doesn't really matter though since we are interested in maximizing the value)
            if val >= cand_max
                #update the max and argmax candidates
                cand_max = val
                cand_pol = i
            end
        end
        #find the value of having a rod of length n by choosing the above candidate 
        value_mat[n] = cand_max
        pol_func[n] = cand_pol 
    end
    #return the value and policy vectors
    value_mat, pol_func
end

function rod_solver(P::Vector{Int64})
    #start with price vector corresponding to rod of length N
    N = length(P)
    #initialize empty policy sequence
    pol_seq = []
    #fill in the value function and policy vectors 
    val, pol = rod_bellman(N, P)
    #use while loop to iteratively find the policy choices required to achieve the max value
    while N >= 1
        #find the policy function for a rod of length N
        cur_pol = Int(pol[N])
        #add the current policy to the policy sequence
        append!(pol_seq, cur_pol)
        #decrease N by the amount of the current policy
        N = N - cur_pol
    end
    #return the overall value as well as the sequence required to attain it
    val[length(P)],pol_seq
end

value_n, cuts = rod_solver( [1,5,8,9,10,17,17,20])
value_n
cuts

value_n, cuts = rod_solver( [1,5,45,9,10,17,17,20])
value_n
cuts
