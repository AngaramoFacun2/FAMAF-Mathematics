import numpy as np
import matplotlib.pyplot as plt

# Lab1

def sol_trinffil(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b donde A es una matriz triangular inferior
    usando sustitución hacia adelante por filas.
    '''

    assert not((np.diag(A) == 0).any()), 'Error: A es una matriz singular (det(A) == 0).'

    n = A.shape[0]
    x = np.zeros(n)

    for k in range(n):
        if b[k] != 0:
            break

    for i in range(k, n):
        x[i] = (b[i] - A[i, k:i] @ x[k:i]) / A[i, i]

    return x


def sol_trinfcol(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b donde A es una matriz triangular inferior
    usando sustitución hacia adelante por columnas.
    '''

    assert not((np.diag(A) == 0).any()), 'Error: A es una matriz singular (det(A) == 0).'

    n = A.shape[0]
    x = np.zeros(n)
    y = b.copy()

    for k in range(n):
        if b[k] != 0:
            break

    for j in range(k, n):
        x[j] = y[j] / A[j, j]
        y[j+1:] = y[j+1:] - A[j+1:, j] * x[j]

    return x


def sol_trsupfil(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b donde A es una matriz triangular superior
    usando sustitución hacia atrás por filas.
    '''

    assert not((np.diag(A) == 0).any()), 'Error: A es una matriz singular (det(A) == 0).'

    n = A.shape[0]
    x = np.zeros(n)

    for k in range(n-1, -1, -1):
        if b[k] != 0:
            break

    for i in range(k, -1, -1):
        x[i] = (b[i] - A[i, i+1:k+1] @ x[i+1:k+1]) / A[i, i]

    return x


def sol_trsupcol(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b donde A es una matriz triangular superior
    usando sustitución hacia atrás por columnas.
    '''

    assert not((np.diag(A) == 0).any()), 'Error: A es una matriz singular (det(A) == 0).'

    n = A.shape[0]
    x = np.zeros(n)
    y = b.copy()

    for k in range(n-1, -1, -1):
        if b[k] != 0:
            break

    for j in range(k, -1, -1):
        x[j] = y[j] / A[j, j]
        y[:j] = y[:j] - A[:j, j] * x[j]

    return x


def nivel(c):
    '''
    Dada una lista de números positivos, la función genera una matriz aleatoria A de tamaño 2x2 
    simétrica definida positiva y gráfica las curvas de nivel de la función f(x) = x.T @ A @ x
    correspondientes a los números de la lista. 
    '''

    B = np.random.random((2, 2))
    A = B @ B.T

    f = lambda x: x.T @ A @ x

    N = 250
    M = 250

    X1 = np.linspace(-8, 8, N)
    X2 = np.linspace(-8, 8, M)

    fX = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            x = np.array([X1[j], X2[i]]).reshape(-1, 1)
            fX[i, j] = f(x)

    fig, ax = plt.subplots()
    CS = ax.contour(X1, X2, fX, c)
    ax.clabel(CS, inline=True, fontsize=10)
    plt.show()


def cholesky_outer(A: np.ndarray) -> np.ndarray:
    '''
    Descomposición de Cholesky (Versión Producto Exterior): Dada una matriz A 
    simétrica y definida positiva, se calcula G triangular superior con 
    elementos diagonales positivos tal que A = G.T @ G.
    '''

    assert np.array_equal(A, A.T), 'Error: La matriz A no es simetrica.'

    n = A.shape[0]
    A_copy = A.copy()
    G = np.zeros((n, n))
    
    for i in range(n):

        assert A_copy[i, i] > 0, 'Error: La matriz A no es definida postiva.'

        G[i, i] = np.sqrt(A_copy[i, i])
        G[i, i+1:] = A_copy[i, i+1:] / G[i, i]
        A_copy[i+1:, i+1:] = A_copy[i+1:, i+1:] - np.outer(G[i, i+1:].T, G[i, i+1:])

    return G


def cholesky_inner(A: np.ndarray) -> np.ndarray:
    '''
    Descomposición de Cholesky (Versión Producto Interior): Dada una matriz A 
    simétrica y definida positiva, se calcula G triangular superior con 
    elementos diagonales positivos tal que A = G.T @ G.
    '''

    assert np.array_equal(A, A.T), 'Error: La matriz A no es simetrica.'

    n = A.shape[0]
    G = np.zeros((n, n))

    for i in range(n):

        G[i, i:] = A[i, i:] - G[:i, i].T @ G[:i, i:]
        assert G[i, i] > 0, 'Error: La matriz A no es definida postiva.'
        G[i, i:] = G[i, i:] / np.sqrt(G[i, i])

    return G


def sol_defpos(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b donde A es una matriz simétrica y definida positiva,
    mediante una descomposición de Cholesky y la resolución de dos sistemas trigulares. 
    '''

    # Calcula el factor de Cholesky de A
    G = cholesky_inner(A)
    # Resuelve el sistema G.T @ y = b
    y = sol_trinffil(G.T, b)
    # Resuelve el sistema G @ x = y
    x = sol_trsupfil(G, y)

    return x 

# Lab2

def egauss(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calcula la eliminación Gaussiana (sin pivoteo) de un sistema Ax = b.
    '''

    n = A.shape[0]
    U = A.copy()
    y = b.copy()

    for k in range(n-1):
        
        assert U[k, k] != 0, 'Error: Existe k en {0, 1, ..., n-2} tal que det(A_k) = 0.'

        v = U[k+1:, k] / U[k, k]
        U[k+1:, k] = 0
        U[k+1:, k+1:] = U[k+1:, k+1:] - np.outer(v, U[k, k+1:])
        y[k+1:] = y[k+1:] - y[k] * v

    return U, y


def dlu(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calcula la descomposición LU (sin pivoteo) de una matriz A.
    '''

    n = A.shape[0]
    dLU = A.copy()

    for k in range(n-1):

        assert dLU[k, k] != 0, 'Error: Existe k en {0, 1, ..., n-2} tal que det(A_k) = 0.'

        dLU[k+1:, k] = dLU[k+1:, k] / dLU[k, k]
        dLU[k+1:, k+1:] = dLU[k+1:, k+1:] - np.outer(dLU[k+1:, k], dLU[k, k+1:])

    L = np.eye(n) + np.tril(dLU, -1)
    U = np.triu(dLU)

    return L, U


def egauss_mod(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calcula la eliminación Gaussiana (sin pivoteo) de un sistema Ax = b donde A es tal que 
    los únicos elementos que pueden ser no nulos son A[0, -1], A[-1, 0] y A[i, j] con |i - j| <= 1. 
    '''
    
    n = A.shape[0]
    U = A.copy()
    y = b.copy()

    for k in range(n-1):

        assert U[k, k] != 0, 'Error: Existe k en {0, 1, ..., n-2} tal que det(A_k) = 0.'
        
        v = U[[k+1, -1], k] / U[k, k]

        U[[k+1, -1], k] = 0
        U[np.ix_([k+1, -1], [k+1, -1])] = U[np.ix_([k+1, -1], [k+1, -1])] - np.outer(v , U[k, [k+1, -1]])

        y[np.ix_([k+1, -1])] = y[np.ix_([k+1, -1])] - y[k] * v

    return U, y


def egaussp(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calcula la eliminación Gaussiana (con pivoteo parcial) de un sistema Ax = b.
    '''

    n = A.shape[0]
    U = A.copy()
    y = b.copy()

    for k in range(n-1):

        l = k + np.argmax(abs(U[k:, k]))

        if U[l, k] != 0:

            U[[k, l], :] = U[[l, k], :]
            y[[k, l]] = y[[l, k]]

            v = U[k+1:, k] / U[k, k]
            U[k+1:, k] = 0
            U[k+1:, k+1:] = U[k+1:, k+1:] - np.outer(v, U[k, k+1:])
            y[k+1:] = y[k+1:] - y[k] * v

    return U, y


def dlup(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calcula la descomposición LU (con pivoteo parcial) de una matriz A.
    '''
    
    n = A.shape[0]
    dLU = A.copy()
    P = np.eye(n)

    for k in range(n-1):

        l = k + np.argmax(abs(dLU[k:, k]))

        if dLU[l, k] != 0:

            dLU[[k, l], :] = dLU[[l, k], :]
            P[[k, l], :] = P[[l, k], :]

            dLU[k+1:, k] = dLU[k+1:, k] / dLU[k, k]
            dLU[k+1:, k+1:] = dLU[k+1:, k+1:] - np.outer(dLU[k+1:, k], dLU[k, k+1:])

    L = np.eye(n) + np.tril(dLU, -1)
    U = np.triu(dLU)

    return L, U, P


def sol_egauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b con A no singular.
    '''
    
    U, y = egaussp(A, b)
    
    return sol_trsupfil(U, y)


def inv_lu(A: np.ndarray) -> np.ndarray:
    '''
    Calcula la matriz inversa de una matriz A utilizando descomposición LU con pivoteo parcial.
    '''

    n = A.shape[0]
    X = np.zeros((n, n))
    L, U, P = dlup(A)

    for j in range(n):

        y = sol_trinffil(L, P[:, j])
        x = sol_trsupfil(U, y)

        X[:, j] = x

    return X


def det_lu(A: np.ndarray) -> float:
    '''
    Calcula el determinante de una matriz A utilizando descomposición LU con pivoteo parcial.
    '''

    _, U, _ = dlup(A)
    return np.prod(np.diag(U))


def circumference(p1, p2, p3):
    '''
    Dado tres puntos no colineales en el plano (p1 = (x1, x2), p2 = (y1, y2) y p3 = (z1, z2)) la función
    realiza el gráfico de aquella circunferencia que pasa por los tres puntos dados.
    '''

    A = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1],
    ], dtype=np.float64)

    b = - np.array([sum(p1**2), sum(p2**2), sum(p3**2) ], dtype=np.float64)

    d, e, f = sol_egauss(A, b)
    r = np.sqrt((d**2 + e**2)/ 4 - f)
    X = np.linspace(0, 2 * np.pi, 100)

    x1 = r * np.cos(X) - d / 2
    x2 = r * np.sin(X)  - e / 2

    fig, ax = plt.subplots()
    plt.plot(x1, x2)
    plt.plot(p1[0], p1[1], marker ="o")
    plt.plot(p2[0], p2[1], marker ="o")
    plt.plot(p3[0], p3[1], marker ="o")
    plt.axis("equal")
    plt.show()

# Lab3

def plot_matrix(A):

    det = []
    cond = []

    for i in range(100):

        epsilon = np.exp(-i)
     
        det.append(np.linalg.det(A(epsilon)))
        cond.append(np.linalg.cond(A(epsilon), p=2))

    plt.plot(det)
    plt.plot(cond)
    plt.legend([r'$det(A(\epsilon))$', r'$\kappa(A(\epsilon))$'], loc='upper left')
    plt.grid(True)
    plt.show()


def transf(A, epsilon):

    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)

    T = A(epsilon) @ np.vstack([x, y])

    plt.plot(x, y, label='Unit Disk')
    plt.plot(T[0, :], T[1, :], label='T(unit Disk)')
    plt.axis('equal')
    plt.legend()
    plt.show()

# Lab4

def givens(x1: float, x2: float) -> tuple[float, float]:
    '''
    Calcula los parámetros de una rotación de Givens.
    '''

    c = 1
    s = 0

    ax1 = abs(x1)
    ax2 = abs(x2)

    if ax1 + ax2 != 0:

        if ax2 > ax1:
            tau = - x1 / x2
            s = - np.sign(x2) / np.sqrt(1 + tau**2)
            c = s * tau
        else:
            tau = - x2 / x1
            c = np.sign(x1) / np.sqrt(1 + tau**2)
            s = c * tau

    return c, s


def householder(x: np.ndarray) -> tuple[np.ndarray, float]:
    '''
    Calcula los parámetros de una reflexión de Householder.
    '''

    n = len(x)

    rho = 0
    u = x.copy()
    u[0] = 1

    if n == 1:
        sigma = 0
    else:
        sigma = sum(x[1:]**2)

    if (sigma != 0) or (x[0] < 0):

        mu = np.sqrt(x[0]**2 + sigma)

        if x[0] <= 0:
            gamma = x[0] - mu
        else:
            gamma = - sigma / (x[0] + mu)

        rho = 2 * gamma**2 / (gamma**2 + sigma)
        u = u/gamma
        u[0] = 1

    return u, rho


def qrgivens(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Realiza la descomposición QR de la matriz A utilizando 
    el método de rotaciones de Givens (sin pivoteo).
    '''

    m, n = A.shape

    Q = np.eye(m)
    R = A.copy()
    p = min(m-1, n)

    for j in range(p):
        for i in range(j+1, m):

            if R[i, j] != 0:

                c, s = givens(R[j, j], R[i, j])
                G = np.array([[c, -s], [s, c]])

                R[[j, i], j:] = G @ R[[j, i], j:]
                Q[:, [j, i]] = Q[:, [j, i]] @ G.T

    if (m <= n) and (R[m-1, m-1] < 0):

        R[m-1, m-1:] = - R[m-1, m-1:]
        Q[:, m-1] = - Q[:, m-1]

    return Q, R


def qrhouseholder(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Realiza la descomposición QR de la matriz A utilizando 
    el método de transformaciones de Householder (sin pivoteo).
    '''

    m, n = A.shape

    Q = np.eye(m)
    R = A.copy()
    p = min(m, n)

    for j in range(p):

        u, rho = householder(R[j:, j])
        w = rho * u
        R[j:, j:] = R[j:, j:] - np.outer(w, u.T @ R[j:, j:])
        Q[:, j:] = Q[:, j:] - Q[:, j:] @ np.outer(w, u.T)

    return Q, R


def qrgramschmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Realiza la descomposición QR de la matriz A utilizando 
    la ortonormalización de Gram-Schmidt.
    '''

    m, n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    R[0, 0] = np.linalg.norm(A[:, 0], ord=2)
    Q[:, 0] = A[:, 0] / R[0, 0]

    for j in range(1, n):

        R[:j, j] = Q[:, :j].T @ A[:, j]
        q = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(q, ord=2)
        Q[:, j] = q / R[j, j]

    return Q, R


def qrgramschmidt_mod(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Realiza la descomposición QR de la matriz A utilizando 
    la ortonormalización de Gram-Schmidt (modificado).
    '''    

    m, n = A.shape

    A = A.copy()
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n-1):

        R[j, j] = np.linalg.norm(A[:, j], ord=2)
        Q[:, j] = A[:, j] / R[j, j]
        R[j, j+1:] = Q[:, j].T @ A[:, j+1:]
        A[:, j+1:] = A[:, j+1:] - np.outer(Q[:, j], R[j, j+1:])
    
    R[-1, -1] = np.linalg.norm(A[:, -1], ord=2)
    Q[:, -1] = A[:, -1] / R[-1, -1]

    return Q, R


def qrgivensp(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Realiza la descomposición QR de la matriz A utilizando 
    el método de rotaciones de Givens con pivoteo de columnas.
    '''
    
    m, n = A.shape

    Q = np.eye(m)
    R = A.copy()
    P = np.eye(n)
    c = np.sum(R**2, axis=0)

    for j in range(min(m - 1, n)):

        l = j + np.argmax(c[j:])

        if c[l] == 0:
            break

        R[:, [j, l]] = R[:, [l, j]]
        P[:, [j, l]] = P[:, [l, j]]
        c[[j, l]] = c[[l, j]]

        for i in range(j+1, m):

            if R[i, j] != 0:

                cos, sin = givens(R[j, j], R[i, j])
                G = np.array([[cos, -sin], [sin, cos]])

                R[[j, i], j:] = G @ R[[j, i], j:]
                Q[:, [j, i]] = Q[:, [j, i]] @ G.T

        c[j:] = c[j:] - R[j, j:]**2

    if (m <= n) and (R[m-1, m-1] < 0):

        R[m-1, m-1:] = -R[m-1, m-1:]
        Q[:, m-1] = -Q[:, m-1]

    return Q, R, P


def qrhouseholderp(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Realiza la descomposición QR de la matriz A utilizando 
    el método de reflexiones de Householder con pivoteo de columnas.
    '''
    
    m, n = A.shape

    Q = np.eye(m)
    R = A.copy()
    P = np.eye(n)
    c = np.sum(R**2, axis=0)

    for j in range(min(m, n)):

        l = j + np.argmax(c[j:])

        if c[l] == 0:
            break

        R[:, [j, l]] = R[:, [l, j]]
        P[:, [j, l]] = P[:, [l, j]]
        c[[j, l]] = c[[l, j]]

        u, rho = householder(R[j:, j])
        w = rho * u

        R[j:, j:] = R[j:, j:] - np.outer(w, u.T @ R[j:, j:])
        Q[:, j:] = Q[:, j:] - np.outer(Q[:, j:] @ w, u)
        c[j:] = c[j:] - R[j, j:]**2

    return Q, R, P


def sol_cuadmin(A: np.ndarray, b: np.ndarray, qr) -> tuple[np.ndarray, float]:
    '''
    Resuelve el problema de cuadrados mínimos (minimizar ||Ax - b||_2) 
    utlizando la descomposición QR con pivoteo de columnas.
    '''

    # Calcula la descomposición QR de la matriz A.
    Q, R, P = qr(A)
    # Calcula q = Q.T @ b
    q = Q.T @ b
    # Determina el rango de R.
    p = np.sum(np.isclose(np.diag(R), 0) == False)
    # Calcula la solución de minimizar ||A @ x - b||_2 y el residuo mínimo.
    x = P @ np.concatenate((sol_trsupfil(R[:p, :p], q[:p]), np.zeros(A.shape[1]-p)))
    r2 = np.linalg.norm(q[p:])

    return x, r2

# Lab5

def cuad_min_svd(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    '''
    Resuelve el problema de cuadrados mínimos (minimizar ||Ax - b||_2) 
    utlizando la descomposición SVD.
    '''

    # Descomposición SVD de A.
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    # Rango(A) = r <= min{m, n}.
    r = sum((np.isclose(s, 0) == False))
    # Pseudoinversa de A.
    A_pseudoinverse = vh[:r, :].T @ np.diag(s[:r]**(-1)) @ u[: , :r].T
    # Solución de minimizar ||Ax - b||_2 y residuo mínimo.
    x = A_pseudoinverse @ b
    r2 = np.linalg.norm(u[:, r:].T @ b)

    return x, r2


def im_aprox_svd(A: np.ndarray, tol:int):

    # Definimos A_k.
    A_k = np.zeros_like(A)
    # Calcula la descomposición SVD de A.
    u, s, vh = np.linalg.svd(A)
    # Determina el rango de A (Rango(A) = r <= min{m, n}).
    r = sum(s > 0)

    for k in range(r):
        A_k += s[k] * np.outer(u[:, k], vh[k, :])
        if np.linalg.norm(A - A_k, ord=np.inf) < tol:
            break

    fig, ax = plt.subplots(1, 2, figsize=(12, 12))
    ax[0].imshow(A, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(A_k, cmap='gray')
    ax[1].set_title(f'Aproximada {k} valores singulares')
    plt.show()


def autjacobi2D(A: np.ndarray) -> tuple[float, float]:
    '''
    Calcula los coeficentes de la matriz de rotación para diagonalizar una matriz simétrica de tamaño 2x2.
    '''

    assert (A.shape == (2, 2)) and (np.isclose(A, A.T).all()), 'Error: A debe ser una matriz simétrica de tamaño 2x2.'

    c = 1
    s = 0

    if A[0, 1] != 0:

        tau = (A[1, 1] - A[0, 0]) / (2 * A[0, 1])

        if tau >= 0:
            t = - 1 / (tau + np.sqrt(tau**2 + 1))
        else:
            t = 1 / (- tau + np.sqrt(tau**2 + 1))

        c = 1 / np.sqrt(1 + t**2)
        s = t * c
    
    return c, s


def off(A: np.ndarray) -> float:
    '''
    Distancia en norma de Frobenius de A al conjunto de matrices diagonales. 
    '''

    assert A.shape[0] == A.shape[1], 'Error: A debe ser una matriz de tamaño nxn.'

    return np.linalg.norm(A - np.diag(np.diag(A)), ord='fro')


def autjacobi(A: np.ndarray, epsilon=1e-10, m=500) -> tuple[np.ndarray, np.ndarray]:
    '''
    Aplica el método de Jacobi para diagonalizar una matriz simétrica de tamaño nxn.
    '''

    assert np.array_equal(A, A.T), 'Error: La matriz A debe ser simétrica de tamaño nxn.'

    n = A.shape[0]

    Q = np.eye(n)
    B = A.copy()

    for k in range(m):

        if off(B) < epsilon:
            break

        i, j = np.unravel_index(np.argmax(np.abs(B - np.diag(np.diag(B)))), B.shape)

        c, s = autjacobi2D(B[np.ix_([i, j], [i, j])])
        J = np.array([[c, -s], [s, c]])

        B[[i, j], :] = J.T @ B[[i, j], :]
        B[:, [i, j]] = B[:, [i, j]] @ J
        Q[:, [i, j]] = Q[:, [i, j]] @ J

    return B, Q


def dvsingulares(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calcula la Descomposición en Valores Singulares (SVD) de una matriz A.
    '''

    # Calcula los autovalores y autovectores de C = A.T @ A.
    C = A.T @ A
    D, W = autjacobi(C)
    # Realiza la descomposición QR con permutación de columnas a la matriz A @ W
    Q, R, P = qrhouseholderp(A @ W)
    # Define U = Q, V = W @ P y S = R
    U = Q
    V = W @ P
    SIGMA = R

    return U, V, SIGMA


def autpotenciasinf(A: np.ndarray, q0: np.ndarray, epsilon=1e-10, m=500) -> tuple[np.ndarray, float]:
    '''
    Dada una matriz A de tamaño nxn con coeficiente reales calcula el autovector dominante y 
    su correspondiente autovalor utilizando el método de potencias en norma infinito.
    '''

    q_hat = A @ q0
    rho_hat = np.inf

    for k in range(m):

        j = np.argmax(np.abs(q_hat))

        q = q_hat / q_hat[j]
        q_hat = A @ q 
        rho = q_hat[j]

        if np.abs(rho - rho_hat) < epsilon:
            break

        rho_hat = rho

    print('k =', k + 1)

    return q, rho


def autpotencias2(A: np.ndarray, q0: np.ndarray, epsilon=1e-10, m=500) -> tuple[np.ndarray, float]:
    '''
    Dada una matriz A de tamaño nxn con coeficiente reales calcula el autovector dominante y 
    su correspondiente autovalor utilizando el método de potencias en norma 2.
    '''

    q_hat = A @ q0
    rho_hat = q0.T @ q_hat / np.linalg.norm(q0, ord=2)**2

    for k in range(m):

        q = q_hat / np.linalg.norm(q_hat, ord=2)
        q_hat = A @ q 
        rho = q.T @ q_hat

        if np.abs(rho - rho_hat) < epsilon:
            break

        rho_hat = rho

    print('k =', k + 1)

    return q, rho


def autrayleigh(A: np.ndarray, q0: np.ndarray, epsilon=1e-10, m=500) -> tuple[np.ndarray, float]:
    '''
    Dada una matriz A de tamaño nxn con coeficiente reales calcula el autovector dominante y 
    su correspondiente autovalor utilizando la iteración del cociente de Rayleigh.
    '''

    q = q0 / np.linalg.norm(q0, ord=2)
    q_hat = q
    rho = q.T @ A @ q 

    I = np.eye(A.shape[0])

    for k in range(m):

        z = sol_egauss(A - rho * I, q_hat)
        sigma = np.linalg.norm(z, ord=2)
        q = z /sigma
        theta = (q.T @ q_hat) / sigma

        if np.abs(theta) < epsilon:
            break

        q_hat = q
        rho = rho + theta

    print('k =', k + 1)

    return q, rho + theta


def fhess(A: np.ndarray, p=0) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calcula la forma de Hessenberg de una matriz cuadrada A 
    mediante reflexiones de Householder o rotaciones de Givens.
    '''

    n = A.shape[0]

    H = A.copy()
    Q = np.eye(n)

    # Householder
    if p == 0:

        for j in range(n-2):

            I = np.s_[j+1: n]
            J = np.s_[j: n]

            u, rho = householder(H[I, j])
            w = rho * u

            H[I, J] = H[I, J] - np.outer(w, u.T @ H[I, J])
            H[:, I] = H[:, I] - H[:, I] @ np.outer(w, u.T)
            Q[:, I] = Q[:, I] - Q[:, I] @ np.outer(w, u.T)

    # Givens
    elif p == 1:

        for j in range(n-2):
            for i in range(j+2, n):

                I = np.s_[j+1, i]
                J = np.s_[j: n]

                c, s = givens(H[j+1, j], H[i, j])
                G = np.array([[c, -s], [s, c]])

                H[I, J] = G @ H[I, J]
                H[:, I] = H[:, I] @ G.T
                Q[:, I] = Q[:, I] @ G.T
        
    else:
        print('Seleccione p = 0 ó p = 1')
        print('- Reflexiones de Householder (p = 0)')
        print('- Rotaciones de Given        (p = 1)')

    return H, Q


def autqr(A: np.ndarray, m=1000) -> tuple[np.ndarray, np.ndarray]:
    '''
    Realiza la descomposición de Schur en IR de la matriz A.
    '''

    n = A.shape[0]
    H, Q = fhess(A)
    rot = np.zeros((n-1, 2))

    for k in range(m):

        for j in range(n - 1):
            
            c, s = givens(H[j, j], H[j + 1, j])
            rot[j, :] = np.array([c, s])
            G = np.array([[c, -s], [s, c]])
            H[[j, j + 1], j:] = G @ H[[j, j + 1], j:]

        for l in range(n - 1):
            c, s = rot[l, :]
            G = np.array([[c, -s], [s, c]])
            H[:, [l, l + 1]] = H[:, [l, l + 1]] @ G.T
            Q[:, [l, l + 1]] = Q[:, [l, l + 1]] @ G.T

    return H, Q

# Lab 6 

def sol_jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray, epsilon=1e-10, m=500) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b utilizando el método de Jacobi.
    '''

    assert (np.diag(A) != 0).all(), 'Error: Existe i en {0, 1, ..., n-1} tal que A[i, i] = 0.'

    L = np.tril(A, -1)
    D = np.diag(A)
    U = np.triu(A, 1)

    J = (L + U) / D.reshape(-1, 1)
    y = b / D

    for k in range(m):

        x = y - J @ x0

        if np.linalg.norm(x - x0, ord=np.inf) < epsilon:
            break

        x0 = x

    print('k =', k+1)
    
    return x


def sol_gs(A: np.ndarray, b: np.ndarray, x0: np.ndarray, epsilon=1e-10, m=1000) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b utilizando el método de Gauss-Seidel.
    '''

    assert (np.diag(A) != 0).all(), 'Error: Existe i en {0, 1, ..., n-1} tal que A[i, i] = 0.'

    n = A.shape[0]

    L = np.tril(A, -1)
    D = np.diag(A)
    U = np.triu(A, 1)

    GS = (L + U) / D.reshape(-1, 1)
    y = b / D
    x = x0.copy()

    for k in range(m):

        for i in range(n):
            x[i] = y[i] - GS[i, :] @ x

        if np.linalg.norm(x - x0, ord=np.inf) < epsilon:
            break

        x0 = x.copy()

    print('k =', k+1)
    
    return x


def sol_cauchy(A: np.ndarray, b: np.ndarray, x: np.ndarray, epsilon=1e-10, m=1000) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b utilizando el método de Cauchy.
    '''

    r = b - A @ x
    sigma = np.linalg.norm(r, ord=2)

    for k in range(m):

        if sigma < epsilon:
            break

        v = A @ r
        t = sigma**2 / (r.T @ v)

        x = x + t * r
        r = r - t * v

        sigma = np.linalg.norm(r, ord=2)

    print('k =', k+1)

    return x


def sol_gastinel(A: np.ndarray, b: np.ndarray, x: np.ndarray, epsilon=1e-10, m=1000) -> np.ndarray:
    '''
    Resuelve el sistema lineal Ax = b utilizando el método de Gastinel.
    '''

    r = b - A @ x
    sigma = np.linalg.norm(r, ord=1)

    for k in range(m):

        if sigma < epsilon:
            break

        d = np.sign(r)
        v = A @ d
        t = (r.T @ d) / (d.T @ v)

        x = x + t * d 
        r = r - t * v

        sigma = np.linalg.norm(r , ord=1)

    print('k =', k+1)

    return x