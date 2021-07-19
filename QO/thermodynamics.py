import numpy as np

def fermi_dirac(e, mu, T): 
    B = np.divide(1,T, where = T > 0) #avoids warnings about T being 0
    return np.where(T > 0, 
            1 / (1 + np.exp(B*(e-mu))),
            e < mu,
    )

def log_one_plus_exp(x): return np.logaddexp(0, x)

def GCP_contributions(e, mu, T): 
    B = np.divide(1,T, where = T > 0) #avoids warnings about T being 0
    return np.where( T > 0,
            - T * log_one_plus_exp(B*(mu-e)),
            (e - mu) * (e < mu),
    )

def GCP(e, mu, T): 
    return np.sum(GCP_contributions(e, mu, T), axis = -1) / e.size