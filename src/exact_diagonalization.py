import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import svds
from scipy.linalg import svd

def sample_J_normal(L, sigma, seed=None):
    """
    Draw couplings J_i = 1 + N(0, sigma^2), length L-1.
    J can be negative; no rejection.
    """
    rng = np.random.default_rng(seed)
    return 1.0 + sigma * rng.standard_normal(L - 1)

def sample_J_discrete(L, c, p):
    return np.random.choice(c, L, p)

def critical_h_from_J(J, tiny=1e-300):
    """
    h_c = exp( (1/(L-1)) * sum_k ln |J_k| ).
    Use absolute value to allow J_k < 0; protect J_k=0 by tiny.
    """
    absJ = np.abs(J)
    absJ = np.where(absJ == 0.0, tiny, absJ)
    return float(np.exp(np.mean(np.log(absJ))))

def smallest_two_eps(J, h="geo_mean", method="eigen_tridiag", scale_M=2.0):
    """ 
    Compute eps1, eps2 at h = h_c(J) using one of two methods: 
    
    eigen_triag: can calculate large L, but precession is bounded 
    by 1e-8 (square root of mechine precision)
    svd_dense: time complexity O(L^3). cannot calculate large L (maximumly~5000). 
    can reach mechine precision 1e-15
    svd_sparse: can caculate large L but very slow.
    can reach mechine precision 1e-15
    """
    J = np.asarray(J, dtype=np.float128)
    L = J.size + 1
    if h == "geo_mean":
        h = critical_h_from_J(J)  # uses ln|J| as you wanted

    if method.lower() == "eigen_tridiag":
        d = np.empty(L, dtype=np.float128)
        d[0] = h*h
        d[1:] = h*h + J*J
        e = h * J
        w = eigh_tridiagonal(d, e, eigvals_only=True,
                             select="i", select_range=(0, 1),
                             check_finite=False)
        eps = np.sqrt(np.maximum(w, 0.0)) * scale_M
        return float(eps[0]), float(eps[1]), h

    elif method.lower() == "svd_sparse":
        # Build sparse bidiagonal M (lower bidiagonal: diag=h, subdiag=J)
        main = np.full(L, h, dtype=np.float128)
        M = diags([J, main], offsets=[-1, 0], shape=(L, L), format="csc")

        # PROPACK backend, smallest magnitude singular values
        # Note: order not guaranteed -> sort ascending
        s = svds(M,k=2,which="SM", solver='propack', tol=1e-10, maxiter=5000, return_singular_vectors=False)
        s = np.sort(s)  # ascending
        eps = s * scale_M
        return float(eps[0]), float(eps[1]), h
    
    elif method.lower() == "svd_dense":
        M = np.zeros((L, L), dtype=np.float128)
        np.fill_diagonal(M, h)
        M[np.arange(1, L), np.arange(0, L-1)] = J
        s = svd(M, compute_uv=False)
        s = np.sort(s)  # ascending
        eps = s * scale_M
        return float(eps[0]), float(eps[1]), h
    else:
        raise ValueError("method must be 'tridiag' or 'svd'")