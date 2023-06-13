import numpy as np

def alpha_m(Vi, params):
    c1, c2, c3, c4 = params
    result_alpha = (c1 * (Vi + c2)) / (1 - np.exp((- (Vi + c3)) / c4))
    return result_alpha

def beta_m(Vi, params):
    c5, c6, c7 = params
    result_beta = c5 * np.exp(- (Vi + c6) / c7)
    return result_beta

def alpha_h(Vi, params):
    c9, c10, c11 = params
    result_alpha = c9 * np.exp(- (Vi + c10) / c11)
    return result_alpha

def beta_h(Vi, params):
    c12, c13 = params
    result_beta = 1 / (1 + np.exp(- (Vi + c12) / c13))
    return result_beta

def _inf(alpha, beta):
    result_m_inf = alpha / (alpha + beta)
    return result_m_inf

def tau(alpha, beta):
    result_tau = 1 / (alpha + beta)
    return result_tau

def m_pow_x(m, params):
    c8 = params
    result_m = m ** c8
    return result_m

def get_y_hat(params, ti, Vi): # todo: debug to check that correct vals go to correct funcs
    alpha_m_ = alpha_m(Vi, params[:4])
    beta_m_ = beta_m(Vi, params[4:7])
    alpha_h_ = alpha_h(Vi, params[8:11])
    beta_h_ = beta_h(Vi, params[11:])
    m_inf_ = _inf(alpha_m_, beta_m_)
    tau_m_ = tau(alpha_m_, beta_m_)
    h_inf_ = _inf(alpha_h_, beta_h_)
    tau_h_ = tau(alpha_h_, beta_h_)
    m = m_inf_ * (1 - np.exp((-ti) / tau_m_))
    h = h_inf_ * (1 - np.exp((-ti) / tau_h_))
    y_hat = m_pow_x(m, params[7]) * h
    return y_hat
