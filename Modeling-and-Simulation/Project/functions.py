import numpy as np
import scipy.stats as stats

from typing import Callable

# Pearson's Chi-Squared Test

def ChiSquared(freq_obs: np.ndarray, freq_exp: np.ndarray) -> float:
    '''
    Calcula el estadístico Chi-Cuadrado.
    '''

    t_obs = sum((freq_obs - freq_exp)**2 / freq_exp)
    return t_obs


def ChiSquared_LogNormal(data: np.ndarray, n_obs: int, mu_hat: float, sigma_hat: float, 
                         bin_edges: np.ndarray, Nsim: int) -> tuple[float, float]:
    '''
    Distribución Log-Normal: Calcula el estadístico Chi-Cuadrado y estima el p-valor utilizando simulaciones.
    '''

    # Calcular las frecuencias observadas
    freq_obs, _ = np.histogram(data, bins=bin_edges)

    # Calcular las frecuencias esperadas
    lognorm_hat = lambda x: stats.norm.cdf((np.log(x) - mu_hat) / sigma_hat)
    freq_exp = np.diff(lognorm_hat(bin_edges)) * n_obs

    # Calcular el estadístico Chi-Cuadrado.
    t_obs = ChiSquared(freq_obs, freq_exp)

    # Estimar el p-value
    p_value = 0
    for _ in range(Nsim):

        sample = np.random.lognormal(mu_hat, sigma_hat, n_obs)
        freq_sim, _ = np.histogram(sample, bins=bin_edges)

        log_sample = np.log(sample)
        mu_sim = np.mean(log_sample)
        sigma_sim = np.std(log_sample)

        lognorm_sim = lambda x: stats.norm.cdf((np.log(x) - mu_sim) / sigma_sim)
        freq_exp = np.diff(lognorm_sim(bin_edges)) * n_obs

        t_sim = ChiSquared(freq_sim, freq_exp)
        if t_sim >= t_obs:
            p_value = p_value + 1
        
    p_value = p_value / Nsim

    return t_obs, p_value


def ChiSquared_Gamma(data: np.ndarray, n_obs: int, alpha_hat: float, beta_hat: float, 
                     bin_edges: np.ndarray, Nsim: int) -> tuple[float, float]:
    '''
    Distribución Gamma: Calcula el estadístico Chi-Cuadrado y estima el p-valor utilizando simulaciones.
    '''

    # Calcular las frecuencias observadas
    freq_obs, _ = np.histogram(data, bins=bin_edges)

    # Calcular las frecuencias esperadas
    gamma_hat = stats.gamma(alpha_hat, scale=beta_hat)
    freq_exp = np.diff(gamma_hat.cdf(bin_edges)) * n_obs

    # Calcular el estadístico Chi-Cuadrado.
    t_obs = ChiSquared(freq_obs, freq_exp)

    # Estimar el p-value
    p_value = 0
    for _ in range(Nsim):

        sample = np.random.gamma(alpha_hat, beta_hat, n_obs)
        freq_sim, _ = np.histogram(sample, bins=bin_edges)

        mean_sim = np.mean(sample)
        variance_sim = np.var(sample)
        alpha_sim = mean_sim**2 / variance_sim
        beta_sim = variance_sim / mean_sim

        gamma_sim = stats.gamma(alpha_sim, scale=beta_sim)
        freq_exp = np.diff(gamma_sim.cdf(bin_edges)) * n_obs

        t_sim = ChiSquared(freq_sim, freq_exp)
        if t_sim >= t_obs:
            p_value = p_value + 1
        
    p_value = p_value / Nsim

    return t_obs, p_value

# Kolmogorov–Smirnov Test

def Kolmogorov_Smirnov(data: np.ndarray, F: Callable[[float], float]) -> float:
    '''
    Calcula el estadístico de Kolmogorov-Smirnov.
    '''
    
    n = len(data)
    data_ = sorted(data.copy())

    d = 0
    for j in range(n):
        d = max(d, (j + 1) / n - F(data_[j]), F(data_[j]) - j / n)

    return d


def KSTest_Uniform(data: np.ndarray, n_obs: int, F: Callable[[float], float], Nsim: int) -> tuple[float, float]:
    '''
    Calcula el estadístico Kolmogorov-Smirnov y estima el p-value utilizando simulaciones.
    '''

    # Calcular el estadístico Kolmogorov-Smirnov
    d_obs = Kolmogorov_Smirnov(data, F)

    # Distribución Uniforme (0, 1)
    G = lambda x: x if (0 < x < 1) else 0

    # Estimar p-value
    p_value = 0
    for _ in range(Nsim):

        sample = np.random.random(n_obs)

        d_sim = Kolmogorov_Smirnov(sample, G)

        if d_sim >= d_obs:
            p_value = p_value + 1

    p_value = p_value / Nsim
    
    return d_obs, p_value


def KSTest_LogNormal(data: np.ndarray, n_obs: int, mu_hat: float, sigma_hat: float, 
                     F_hat: Callable[[float], float], Nsim: int) -> tuple[float, float]:
    '''
    Distribución Log-Normal: Calcula el estadístico Kolmogorov-Smirnov y estima el p-value utilizando simulaciones.
    '''

    # Calcular el estadístico Kolmogorov-Smirnov
    d_obs = Kolmogorov_Smirnov(data, F_hat)

    # Estimar el p-value
    p_value = 0
    for _ in range(Nsim):

        sample = np.random.lognormal(mu_hat, sigma_hat, n_obs)
        log_sample = np.log(sample)

        mu_sim = np.mean(log_sample)
        sigma_sim = np.std(log_sample)

        F_sim = lambda x: stats.norm.cdf((np.log(x) - mu_sim) / sigma_sim)

        d_sim = Kolmogorov_Smirnov(sample, F_sim)
        if d_sim > d_obs:
            p_value = p_value + 1

    p_value = p_value / Nsim

    return d_obs, p_value


def KSTest_Gamma(data: np.ndarray, n_obs: int, alpha_hat: float, beta_hat: float, 
                 F_hat: Callable[[float], float], Nsim: int) -> tuple[float, float]:
    '''
    Distribución Gamma: Calcula el estadístico Kolmogorov-Smirnov y estima el p-value utilizando simulaciones.
    '''

    # Calcular el estadístico Kolmogorov-Smirnov
    d_obs = Kolmogorov_Smirnov(data, F_hat)

    # Estimar p-value
    p_value = 0
    for _ in range(Nsim):

        sample = np.random.gamma(alpha_hat, beta_hat, n_obs)

        mean_sim = np.mean(sample)
        variance_sim = np.var(sample)

        alpha_sim = mean_sim**2 / variance_sim
        beta_sim = variance_sim / mean_sim

        F_sim = lambda x: stats.gamma.cdf(x, alpha_sim, scale=beta_sim)

        d_sim = Kolmogorov_Smirnov(sample, F_sim)
        if d_sim > d_obs:
            p_value = p_value + 1

    p_value = p_value / Nsim

    return d_obs, p_value
