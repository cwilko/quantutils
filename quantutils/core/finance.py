from numpy import *

def gross_return(prices):
    N = len(prices)
    y=ones(N)
    for n in range(1,N):
        if prices[n] == prices[n-1]:
            y[n] = 1.0
        else:
            y[n] = float(prices[n]) / prices[n-1]
    return y
    
def log_return(prices):
    # gr = gross_return(prices);
    # N = length(prices);
    # y = 1:N;
    # for n=1:N
    #     if (gr(n) < 0)
    #         y(n) = 0;
    #     else
    #         y(n) = log(gr(n));
    #     end
    # end
    # y = cumsum(y);
 
    return cumsum(log(gross_return(prices)));

def net_return(prices, strict=True):
    N = len(prices)-1;        
    y=zeros(N);
    for n in range(0,N):
        if prices[n] == prices[n+1]:
            y[n] = 0.0;
        else:
            y[n] = float(prices[n+1] - prices[n]) / prices[n];
    if strict==False:
        y = insert(y,0,0)
    return y
    
def net_return2(prices):
    return gross_return(prices) - 1;    

def net_return3(prices):
    return float64(prices[1:,:] - prices[:-1,:] / prices[:-1,:])
    
def compound_return(prices):
    return cumprod(gross_return(prices))
    
def percent_return(prices):
    return (compound_return(prices) - 1) * 100;
    

