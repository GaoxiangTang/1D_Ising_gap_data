import numpy as np

# ---------------- upper bound utility (no exact diagonalization) ----------------
def eps2_upper_bound(J, h, num_splits="all", m_ratio=0.5, dd_ratios=(1/3, 2/3),
                     j_clip=(1e-12, 1e12), rng=None):
    """
    Upper bound for eps2 using only analytical bounds (ND on left subchain, DD on right subchain),
    combined by the two-block recipe: lambda2_ub = 2 * max(lambda1_left_ND, lambda1_right_DD),
    then eps2_ub = 2 * sqrt(lambda2_ub). No exact diagonalization is used.

    Args
    ----
    J : array-like, shape (L-1,)
        Couplings (can be any real numbers). They are clamped into [j_min, j_max] before logs.
    h : float
        Positive field.
    num_splits : "all" or int
        "all" tests all split positions m in [1..L-1]; int(k) samples k random splits.
    m_ratio : float in (0,1)
        Internal ratio for the ND bound inside the left subchain.
    dd_ratios : (float, float)
        Internal ratios (n_ratio, m_ratio) for the DD bound inside the right subchain.
    j_clip : (float, float)
        Clamp J into [j_min, j_max] to keep logs well-defined.
    rng : int or np.random.Generator or None
        Random source when num_samples is an int.

    Returns
    -------
    eps2_ub : float
        Upper bound on eps2.
    """
    # basic checks and clamping
    J = np.asarray(J, dtype=float).ravel()
    if J.ndim != 1 or J.size == 0:
        raise ValueError("J must be a 1D array with length >= 1.")
    if h <= 0:
        raise ValueError("h must be positive.")
    j_min, j_max = j_clip
    if not (j_min > 0 and j_max > j_min):
        raise ValueError("j_clip must be (positive_min, larger_positive_max).")
    J = np.clip(np.abs(J), j_min, j_max)  # force positive before logs

    L = J.size + 1
    if L < 2:
        return 0.0

    # helpers (kept private inside this utility)
    def log_potential_u(J_sub, h_sub):
        # u[0] = 0; u[i] = sum_{a=1..i} log(J_sub[a-1]/h_sub)
        Ls = J_sub.size + 1
        u = np.zeros(Ls, dtype=float)
        if Ls > 1:
            u[1:] = np.cumsum(np.log(J_sub / h_sub))
        return u

    def nd_lambda1_bound(J_sub, h_sub, m_ratio_sub):
        # ND principal eigenvalue upper bound on a subchain with Dirichlet boundary on its left end.
        Ls = J_sub.size + 1
        if Ls < 2:
            return np.inf
        m_nd = int(np.clip(int(round(m_ratio_sub * Ls)), 1, Ls - 1))
        u = log_potential_u(J_sub, h_sub)
        uL = u[-1]
        s_left  = np.exp(2.0 * (uL - u[:m_nd])).sum()      # i in [1..m_nd]
        s_right = np.exp(2.0 * (u[m_nd-1:] - uL)).sum()    # j in [m_nd..Ls]
        return (h_sub ** 2) / s_left / s_right

    def dd_lambda1_bound(J_sub, h_sub, n_ratio_sub, m_ratio_sub):
        # DD principal eigenvalue upper bound on a subchain with Dirichlet at both ends.
        Ls = J_sub.size + 1
        if Ls < 2:
            return np.inf
        n_dd = max(1, int(np.floor(n_ratio_sub * Ls)))
        m_dd = max(n_dd + 1, int(np.floor(m_ratio_sub * Ls)))
        m_dd = min(m_dd, Ls - 1)
        if not (1 <= n_dd < m_dd <= Ls - 1):
            # fallback to thirds when ratios are pathological
            n_dd, m_dd = max(1, Ls // 3), min(Ls - 1, max(Ls // 3 + 1, 2 * Ls // 3))
        u = log_potential_u(J_sub, h_sub)
        S_mid   = np.exp(-2.0 * u[n_dd - 1:m_dd]).sum()    # i in [n..m]
        S_right = np.exp( 2.0 * u[m_dd - 1:]).sum()        # j in [m..Ls]
        S_left  = np.exp( 2.0 * u[:n_dd - 1]).sum()        # j in [1..n-1]
        term_left  = np.inf if S_left  == 0.0 else 1.0 / S_left
        term_right = np.inf if S_right == 0.0 else 1.0 / S_right
        return (h_sub ** 2) / S_mid * (term_left + term_right)

    # choose split positions
    if num_splits == "all":
        m_candidates = np.arange(1, L, dtype=int)
    else:
        k = int(num_splits)
        if k <= 0:
            raise ValueError("num_splits must be 'all' or a positive integer.")
        rng = np.random.default_rng(rng)
        m_candidates = rng.integers(1, L, size=k)

    n_ratio_dd, m_ratio_dd = dd_ratios
    best_lambda2 = np.inf

    for m_split in m_candidates:
        m_split = int(m_split)
        # left subchain [1..m_split] -> bonds J[:m_split-1]
        J_left = J[:max(0, m_split - 1)]
        # right subchain [m_split+1..L] -> bonds J[m_split:]
        J_right = J[m_split:]

        lam_nd = nd_lambda1_bound(J_left, h, m_ratio) if (J_left.size + 1) >= 2 else np.inf
        lam_dd = dd_lambda1_bound(J_right, h, n_ratio_dd, m_ratio_dd) if (J_right.size + 1) >= 2 else np.inf

        lambda2_ub = 2.0 * max(lam_nd, lam_dd)
        if lambda2_ub < best_lambda2:
            best_lambda2 = lambda2_ub

    return 2.0 * np.sqrt(best_lambda2)


import numpy as np

# ---------------- upper bound utility (no exact diagonalization) ----------------
def eps2_upper_bound(J, h, num_splits="sweep", nd_ratio=0.5, dd_ratio=(1/3, 2/3),
                     j_clip=(1e-12, 1e12), rng=None):
    """
    Upper bound for eps2 using analytical bounds only (ND on left subchain, DD on right subchain),
    combined by the two-block recipe: lambda2_ub = 2 * max(lambda1_left_ND, lambda1_right_DD),
    then eps2_ub = 2 * sqrt(lambda2_ub). No exact diagonalization is used.

    New behavior:
      - num_splits: "sweep" (default) tries all split positions m in [1..L-1]; or an int -> sample k random splits.
      - nd_ratio: can be a float in (0,1) to pick one ND internal point, or "sweep" to try all ND internal points.
      - dd_ratio: can be a pair (n_ratio, m_ratio) as before, or "sweep" to try all valid DD pairs (n < m).

    Time complexity:
      - With nd_ratio="sweep", dd_ratio="sweep", num_splits="sweep": overall O(L^3) with small constants.
      - Memory: O(L).

    Args
    ----
    J : array-like, shape (L-1,)
        Couplings (any real numbers). They are clamped into [j_min, j_max] before logs (abs taken first).
    h : float
        Positive field.
    num_splits : "sweep" or int
        "sweep" tests all split positions m in [1..L-1]; int(k) samples k random splits.
    nd_ratio : float in (0,1) or "sweep"
        Internal ratio for the ND bound inside the left subchain, or "sweep" to try all.
    dd_ratio : (float, float) or "sweep"
        Internal ratios (n_ratio, m_ratio) for the DD bound inside the right subchain, or "sweep" to try all.
    j_clip : (float, float)
        Clamp |J| into [j_min, j_max] to keep logs well-defined.
    rng : int or np.random.Generator or None
        Random source when num_splits is an int.

    Returns
    -------
    eps2_ub : float
        Upper bound on eps2.
    """
    # basic checks and clamping
    J = np.asarray(J, dtype=float).ravel()
    if J.ndim != 1 or J.size == 0:
        raise ValueError("J must be a 1D array with length >= 1.")
    if h <= 0:
        raise ValueError("h must be positive.")
    j_min, j_max = j_clip
    if not (j_min > 0 and j_max > j_min):
        raise ValueError("j_clip must be (positive_min, larger_positive_max).")
    J = np.clip(np.abs(J), j_min, j_max)  # force positive before logs

    L = J.size + 1
    if L < 2:
        return 0.0

    # helpers (kept private inside this utility)
    def log_potential_u(J_sub, h_sub):
        # u[0] = 0; u[i] = sum_{a=1..i} log(J_sub[a-1]/h_sub)
        Ls = J_sub.size + 1
        u = np.zeros(Ls, dtype=float)
        if Ls > 1:
            u[1:] = np.cumsum(np.log(J_sub / h_sub))
        return u

    def nd_lambda1_bound_sweep(J_sub, h_sub, nd_ratio_opt):
        """
        ND principal eigenvalue upper bound on a subchain with Dirichlet boundary on its left end.
        If nd_ratio_opt == "sweep", returns the minimum over m_nd in [1..Ls-1].
        Else uses a single internal point based on nd_ratio_opt in (0,1).
        """
        Ls = J_sub.size + 1
        if Ls < 2:
            return np.inf

        u = log_potential_u(J_sub, h_sub)
        # Precompute arrays for O(1) queries:
        # A[i] = exp(-2*u[i]), B[i] = exp( 2*u[i])
        A = np.exp(-2.0 * u)
        B = np.exp( 2.0 * u)
        # Standard prefix with leading 0: prefX0[k] = sum_{i=0..k-1} X[i]
        prefA0 = np.concatenate(([0.0], np.cumsum(A)))
        prefB0 = np.concatenate(([0.0], np.cumsum(B)))
        # Also suffix of B for convenience (could be derived from prefB0 too)
        # suffixB[i] = sum_{j=i..Ls-1} B[j] = prefB0[Ls] - prefB0[i]
        # Formula for a given m_nd in 1..Ls-1:
        # s_left(m)  = sum_{i=0..m-1} exp(2*(uL-u[i])) = exp(2uL) * sum A[i] = exp(2uL)*prefA0[m]
        # s_right(m) = sum_{j=m-1..Ls-1} exp(2*(u[j]-uL)) = exp(-2uL) * sum B[j] = exp(-2uL)*(prefB0[Ls]-prefB0[m-1])
        # The exp(±2uL) cancel in lam_nd:
        # lam_nd(m) = h^2 / (prefA0[m] * (prefB0[Ls]-prefB0[m-1]))
        totalB = prefB0[Ls]

        def lam_for_m(m_nd):
            sL = prefA0[m_nd]
            sR = totalB - prefB0[m_nd - 1]
            if sL <= 0.0 or sR <= 0.0:
                return np.inf
            return (h_sub ** 2) / (sL * sR)

        if nd_ratio_opt == "sweep":
            # try all m_nd in 1..Ls-1
            m_vals = range(1, Ls)  # Ls-1 values
            vals = [lam_for_m(m) for m in m_vals]
            return np.min(vals) if len(vals) else np.inf
        else:
            m_nd = int(np.clip(int(round(nd_ratio_opt * Ls)), 1, Ls - 1))
            return lam_for_m(m_nd)

    def dd_lambda1_bound_sweep(J_sub, h_sub, dd_ratio_opt):
        """
        DD principal eigenvalue upper bound on a subchain with Dirichlet at both ends.
        If dd_ratio_opt == "sweep", returns the minimum over all pairs (n,m) with 1 <= n < m <= Ls-1.
        Else uses a single pair determined by dd_ratio_opt=(n_ratio,m_ratio).
        """
        Ls = J_sub.size + 1
        if Ls < 2:
            return np.inf

        u = log_potential_u(J_sub, h_sub)
        # Precompute A = exp(-2u), B = exp(2u)
        A = np.exp(-2.0 * u)
        B = np.exp( 2.0 * u)
        # Standard prefix with leading 0 for O(1) range sums:
        prefA0 = np.concatenate(([0.0], np.cumsum(A)))  # len Ls+1
        prefB0 = np.concatenate(([0.0], np.cumsum(B)))  # len Ls+1
        totalB = prefB0[Ls]

        # Given 1-based (n, m) with 1 <= n < m <= Ls-1:
        # S_mid(n,m)   = sum_{i=n-1..m-1} A[i]       = prefA0[m] - prefA0[n-1]
        # S_left(n)    = sum_{j=0..n-2} B[j]         = prefB0[n-1]
        # S_right(m)   = sum_{j=m-1..Ls-1} B[j]      = totalB - prefB0[m-1]
        def lam_for_nm(n_dd, m_dd):
            if not (1 <= n_dd < m_dd <= Ls - 1):
                return np.inf
            S_mid   = prefA0[m_dd] - prefA0[n_dd - 1]
            S_left  = prefB0[n_dd - 1]
            S_right = totalB - prefB0[m_dd - 1]
            if S_mid <= 0.0:
                return np.inf
            term_left  = np.inf if S_left  <= 0.0 else 1.0 / S_left
            term_right = np.inf if S_right <= 0.0 else 1.0 / S_right
            return (h_sub ** 2) / S_mid * (term_left + term_right)

        if dd_ratio_opt == "sweep":
            # Try all pairs (n,m): 1 <= n < m <= Ls-1
            if Ls <= 2:
                return np.inf
            best = np.inf
            for m in range(2, Ls):        # m from 2..Ls-1
                # optional micro-optimizations: none needed; O(Ls^2) here
                for n in range(1, m):     # n from 1..m-1
                    val = lam_for_nm(n, m)
                    if val < best:
                        best = val
            return best
        else:
            n_ratio_sub, m_ratio_sub = dd_ratio_opt
            n_dd = max(1, int(np.floor(n_ratio_sub * Ls)))
            m_dd = max(n_dd + 1, int(np.floor(m_ratio_sub * Ls)))
            m_dd = min(m_dd, Ls - 1)
            if not (1 <= n_dd < m_dd <= Ls - 1):
                # fallback to thirds when ratios are pathological
                n_dd, m_dd = max(1, Ls // 3), min(Ls - 1, max(Ls // 3 + 1, 2 * Ls // 3))
            return lam_for_nm(n_dd, m_dd)

    # choose split positions
    if num_splits == "sweep":
        m_candidates = np.arange(1, L, dtype=int)
    else:
        k = int(num_splits)
        if k <= 0:
            raise ValueError("num_splits must be 'sweep' or a positive integer.")
        rng = np.random.default_rng(rng)
        m_candidates = rng.integers(1, L, size=k)

    best_lambda2 = np.inf

    for m_split in m_candidates:
        m_split = int(m_split)
        # left subchain [1..m_split] -> bonds J[:m_split-1]
        J_left = J[:max(0, m_split - 1)]
        # right subchain [m_split+1..L] -> bonds J[m_split:]
        J_right = J[m_split:]

        lam_nd = nd_lambda1_bound_sweep(J_left, h, nd_ratio) if (J_left.size + 1) >= 2 else np.inf
        lam_dd = dd_lambda1_bound_sweep(J_right, h, dd_ratio) if (J_right.size + 1) >= 2 else np.inf

        lambda2_ub = 2.0 * max(lam_nd, lam_dd)
        if lambda2_ub < best_lambda2:
            best_lambda2 = lambda2_ub

    # If something degenerate happened (shouldn't), guard:
    if not np.isfinite(best_lambda2):
        return np.inf

    return 2.0 * np.sqrt(best_lambda2)
