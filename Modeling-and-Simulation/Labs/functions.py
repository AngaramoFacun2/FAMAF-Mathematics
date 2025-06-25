import math
import random

# Random Numbers

def middle_square(seed:int) -> int:
    '''
    Generate pseudo-random numbers using the Middle Square method.
    '''

    return (seed**2 // 100) % 10000


def lcg(a:int, c:int, M:int, seed:int) -> int:
    '''
    Generate pseudo-random numbers using the Linear Congruential Generator (LCG) method.
    '''

    return (a * seed + c) % M

# Monte Carlo Method

def MonteCarlo_integration(fun, N, seed=42):
    '''
    Approximates the integral of a given function over the interval (0, 1)
    using Monte Carlo integration method.
    '''
                     
    random.seed(seed)

    S = 0
    for _ in range(N):
        U = random.random()
        S = S + fun(U)
    I = S / N

    return I
  

def MonteCarlo_double_integration(fun, N, seed=42):
    '''
    Approximates the double integral of a given function over the unit square [0, 1] x [0, 1]
    using Monte Carlo integration method.
    '''

    random.seed(seed)

    S = 0
    for _ in range(N):
        U, V = random.random(), random.random()
        S = S + fun(U, V)
    I = S / N

    return I

# Generating Discrete Random Variables

def InvTransform_Discrete(X, Pr):
    '''
    Generate a discrete random variable using the inverse transform method.
    '''

    U = random.random()

    i = 0
    F = Pr[i]

    while U >= F:
        i = i + 1
        F = F + Pr[i]
    
    return X[i]


def DiscreteUniform(n:int) -> int:
    '''
    Generate a Discrete Uniform random variable.
    '''

    U = random.random()
    X = int(n * U) + 1

    return X


def DiscreteUniform_mod(m:int, n:int) -> int:
    '''
    Generate a Discrete Uniform random variable within a specified range. 
    '''

    U = random.random()
    X = int((n - m + 1) * U) + m

    return X


def permutation(A):
    '''
    Generate a random permutation of an array.
    '''

    N = len(A)

    A_copy = A.copy()

    for i in range(N - 1, 0, -1):
        j = int((i + 1) * random.random())
        A_copy[i], A_copy[j] = A_copy[j], A_copy[i]

    return A_copy


def subset(A, r:int):
    '''
    Generate a random subset of a specified size from an array.
    '''

    N = len(A)

    A_copy = A.copy()

    for i in range(N - 1, N - 1 - r, -1):
        j = int((i + 1) * random.random())
        A_copy[i], A_copy[j] = A_copy[j], A_copy[i]

    return A_copy[N - r:]


def average(fun, N:int, Nsim:int) -> float:
    '''
    Compute the average value of a function applied to a Discrete Uniform 
    random variable over multiple simulations.
    '''
     
    SUM = 0

    for _ in range(Nsim):

        U = DiscreteUniform(N)
        SUM = SUM + fun(U)
    
    AVG = SUM / Nsim

    return AVG


def Geometric(p:float) -> int:
    '''
    Generate a Geometric random variable.
    '''

    U = random.random()
    X = int(math.log(1 - U) / math.log(1 - p)) + 1

    return X


def Bernoulli(p:float) -> int:
    '''
    Generate a Bernoulli random variable.
    '''

    U = random.random()

    if U < p:
        X = 1
    else:
        X = 0
    
    return X


def Bernoulli_mod(p:float, N:int) -> list:
    '''
    Generate a sequence of Bernoulli trials with a specified length.
    '''

    seq = [0] * N
    i = Geometric(p) - 1
    while i < N:
        seq[i] = 1
        i = i + Geometric(p)
    
    return seq


def Poisson(lambd:float) -> int:
    '''
    Generate a Poisson random variable.
    '''

    U = random.random()

    i = 0
    Pr = math.exp(-lambd)
    F = Pr

    while U >= F:

        i = i + 1
        Pr = (lambd / i) * Pr
        F = F + Pr
    
    return i


def Poisson_mod(lambd:float) -> int:
    '''
    Generate a Poisson random variable.
    '''

    Pr = math.exp(-lambd)
    F = Pr
    for i in range(1, int(lambd) + 1):
        Pr = (lambd / i) * Pr
        F = F + Pr
        
    # i = ⌊lambd⌋, Pr = P(X = ⌊lambd⌋) and F = P(X <= ⌊lambd⌋)

    U = random.random()

    if U >= F:
        while U >= F:
            i = i + 1
            Pr = (lambd / i) * Pr
            F = F + Pr
    
    else:
        F = F - Pr
        Pr = (i / lambd) * Pr
        while U < F:
            i = i - 1
            F = F - Pr
            Pr = (i / lambd) * Pr
        
    return i


def Binomial(n:int, p:float) -> int:
    '''
    Generate a Binomial random variable.
    '''

    U = random.random()

    c = p / (1 - p)
    
    Pr = (1 - p)**n
    F = Pr
    i = 0

    while U >= F:

        Pr = ((n - i) / (i + 1)) * c * Pr
        F = F + Pr
        i = i + 1        
        
    return i

# Generating Continuous Random Variables

def InvTransform_Continuous(G) -> float:
    '''
    Generate a continuous random variable using the inverse transform method.
    '''

    U = random.random()
    X = G(U)

    return X


def Exp(lambd:float) -> float:
    '''
    Generate a Exponential random variable.
    '''

    U = random.random()
    X = - math.log(1 - U) / lambd

    return X


def Poisson_with_Exp(lambd:float) -> int:

    bound = math.exp(-lambd)

    i = 0
    prod = 1 - random.random()
    while(bound <= prod):
        i = i + 1
        prod = prod * (1 - random.random())
    
    return i


def Gamma(n:int, lambd:float) -> float:
    '''
    Generate a Gamma random variable.
    '''

    PROD = 1
    for _ in range(n):
        U = random.random()
        PROD = PROD * (1 - U)
    S = - math.log(PROD) / lambd

    return S


def double_Exp(lambd:float) -> tuple[float, float]:

    t = Gamma(2, lambd)     # X + Y = t (X, Y ~ Exp(lambd))
    X = t * random.random() # X | X + Y = t ~ U(0, t)
    Y = t - X               # Y = t - X

    return X, Y


def Pareto(alpha:float) -> float:
    '''
    Generate a Pareto random variable.
    '''
    
    U = random.random()
    X = (1 - U)**(-1/alpha)

    return X


def Weibull(lambd:float, beta:float) -> float:
    '''
    Generate a Weibull random variable.
    '''
    
    U = random.random()
    X = lambd * (-math.log(1 - U))**(1/beta)

    return X


def Normal_with_Exp() -> float:
    '''
    Generate a Standard Normal random variable.
    '''

    U = random.random()
    Y = Exp(lambd=1)
    while U >= math.exp(- (Y-1)**2 / 2):
        U = random.random()
        Y = Exp(lambd=1)

    if random.random() < 0.5:
        Z = Y
    else:
        Z = -Y

    return Z 


def Normal_Polar() -> tuple[float, float]:
    '''
    Generate two independent Standard Normal random variables.
    '''
    
    r = math.sqrt(Exp(1/2))
    theta = 2 * math.pi * random.random()

    Z1 = r * math.cos(theta)
    Z2 = r * math.sin(theta)

    return Z1, Z2


def PoissonProcess(lambd, T):
    '''
    Simulate a Poisson Process.
    '''

    N = 0
    seq = []

    t = Exp(lambd)
    while t <= T:

        N = N + 1
        seq.append(t)

        t = t + Exp(lambd)
    
    return N, seq


def PoissonProcess_mod(lambd_, T, Inter, Lambd):
    '''
    Simulate a non-homogeneous Poisson Process.
    '''

    N = 0
    seq = []

    j = 0
    t = Exp(Lambd[j])

    while t <= T:

        if t <= Inter[j]:
            U = random.random()
            if U < lambd_(t) / Lambd[j]:
                N = N + 1
                seq.append(t)
            t = t + Exp(Lambd[j])
        
        else:
            t = Inter[j] + (Lambd[j] / Lambd[j + 1]) * (t - Inter[j])    
            j = j + 1

    return N, seq

# Statistical Analysis of Simulated Data

def MonteCarlo_integration_v1(fun, tol):
    '''
    Approximates the integral of a given function over the interval (0, 1)
    using Monte Carlo integration method.
    '''

    n = 1
    U = random.random()
    f = fun(U)

    mean = f
    variance = 0

    MSE = variance / n

    while (n < 100) or (math.sqrt(MSE) > tol):

        n = n + 1
        U = random.random()
        f = fun(U)
        
        prev = mean
        mean = prev + (f - prev) / n
        variance = (n - 2) / (n - 1) * variance + n * (mean - prev)**2

        MSE = variance / n

    return n, mean, variance


def MonteCarlo_integration_v2(fun, Z, L):
    '''
    Approximates the integral of a given function over the interval (0, 1)
    using Monte Carlo integration method.
    '''
    
    tol = L / (2 * Z)

    n = 1
    U = random.random()
    f = fun(U)

    mean = f
    variance = 0

    MSE = variance / n

    while (n < 100) or (math.sqrt(MSE) > tol):
        
        n = n + 1
        U = random.random()
        f = fun(U)
        
        prev = mean
        mean = prev + (f - prev) / n
        variance = (n - 2) / (n - 1) * variance + n * (mean - prev)**2 

        MSE = variance / n

    lower_limit = mean - Z * math.sqrt(variance / n)
    upper_limit = mean + Z * math.sqrt(variance / n)

    return n, mean, variance, (lower_limit, upper_limit)
