from arch import arch_model

def fit_garch(returns, p=1, q=1, dist="normal"):
    am = arch_model(returns * 100, vol="GARCH", p=p, q=q, dist=dist, mean="Zero")
    return am.fit(disp="off")
