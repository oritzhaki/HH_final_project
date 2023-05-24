import numpy as np

dict= {"-20" : 0,
       "-10" : 1,
       "0" : 2,
       "10" : 3,
       "20" : 4,
       "30" : 5,
       "40" : 6,
       "50" : 7,
       "60" : 8,
       "70" : 9,
       "80" : 10,
       "90" : 11
}

def alpha_n(Vi, params):
    c1, c2, c3, c4 = params
    result_alpha = (c1 * (Vi + c2)) / (1 - np.exp(-c3 * (Vi + c4)))
    return result_alpha

def beta_n(Vi, params):
    c5, c6, c7 = params
    result_beta = c5 * (np.exp(-c6 * ( Vi + c7 )))
    return result_beta

def n_inf(alpha, beta):
    result_n_inf =  alpha / (alpha + beta)
    return result_n_inf

def tau_n(alpha, beta):
    result_tau = 1 / (alpha + beta)
    return result_tau

def n_pow_4(n, params, Vi):
    idx = dict[str(Vi)]
    c8 = params[idx]
    result_n =  n ** c8
    return result_n

def get_y_hat(params, ti, Vi):
    alpha = alpha_n(Vi, params[:4])
    beta = beta_n(Vi, params[4:7])
    n_inf_ = n_inf(alpha, beta)
    tau_n_ = tau_n(alpha, beta)
    n = n_inf_ * (1 - np.exp((-ti) / tau_n_))
    y_hat = n_pow_4(n, params[-1], Vi)
    return y_hat
