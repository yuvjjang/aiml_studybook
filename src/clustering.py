"""
군집화 참조 구현 (NumPy).

Part 2.9 가 쓰는 k-means, GMM(EM), DBSCAN, 계층적 군집화와 평가 지표.
모두 반복 과정을 스냅샷으로 남길 수 있어 애니메이션에 쓸 수 있다.
"""
import numpy as np


# ── k-means ───────────────────────────────────────────────────────────

def kmeans_pp_init(X, k, rng):
    """k-means++ 초기화: 기존 중심에서 멀수록 뽑힐 확률이 높다."""
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(len(X), 1 / len(X))
        centers.append(X[rng.choice(len(X), p=probs)])
    return np.array(centers)


def kmeans(X, k, n_iter=100, init="k-means++", seed=0, tol=1e-9):
    """Lloyd 알고리즘. 할당 ↔ 중심 갱신을 번갈아 반복.

    Returns
    -------
    labels, centers, history : history 는 (centers, labels, inertia) 스냅샷 목록
    """
    rng = np.random.default_rng(seed)
    if init == "k-means++":
        C = kmeans_pp_init(X, k, rng)
    else:                                        # 무작위 데이터점
        C = X[rng.choice(len(X), k, replace=False)].copy()

    history = []
    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        labels = d2.argmin(1)
        inertia = float(d2[np.arange(len(X)), labels].sum())
        history.append((C.copy(), labels.copy(), inertia))

        C_new = np.array([X[labels == j].mean(0) if (labels == j).any() else C[j]
                          for j in range(k)])
        if np.abs(C_new - C).max() < tol:
            C = C_new
            break
        C = C_new

    d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
    labels = d2.argmin(1)
    return labels, C, history


def inertia(X, labels, centers):
    return float(sum(((X[labels == j] - centers[j]) ** 2).sum()
                     for j in range(len(centers))))


# ── 가우시안 혼합 모형 (EM) ───────────────────────────────────────────

def gmm_em(X, k, n_iter=100, seed=0, reg=1e-6):
    """가우시안 혼합모형을 EM 으로 적합.

    E단계: 각 점의 성분 소속 확률(책임도) 계산
    M단계: 책임도 가중으로 파라미터 갱신
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    C = kmeans_pp_init(X, k, rng)
    pi = np.full(k, 1.0 / k)
    Sigma = np.array([np.cov(X.T) + reg * np.eye(d) for _ in range(k)])

    history, ll_hist = [], []
    R = np.full((n, k), 1.0 / k)
    for _ in range(n_iter):
        # E 단계 — 로그 공간에서 (0.5 수치 컴퓨팅)
        logp = np.zeros((n, k))
        for j in range(k):
            L = np.linalg.cholesky(Sigma[j])
            diff = np.linalg.solve(L, (X - C[j]).T)
            logdet = 2 * np.sum(np.log(np.diag(L)))
            logp[:, j] = (-0.5 * (diff ** 2).sum(0) - 0.5 * logdet
                          - 0.5 * d * np.log(2 * np.pi) + np.log(pi[j] + 1e-300))
        m = logp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logp - m).sum(1))
        R = np.exp(logp - lse[:, None])
        ll_hist.append(float(lse.sum()))
        history.append((C.copy(), Sigma.copy(), pi.copy(), R.copy()))

        # M 단계
        Nk = R.sum(0) + 1e-12
        pi = Nk / n
        C = (R.T @ X) / Nk[:, None]
        for j in range(k):
            diff = X - C[j]
            Sigma[j] = (R[:, j][:, None] * diff).T @ diff / Nk[j] + reg * np.eye(d)

    return R.argmax(1), C, Sigma, pi, R, history, ll_hist


def ellipse_points(mu, Sigma, n_std=2.0, n=100):
    """공분산 행렬의 등고선 타원 좌표 (고유분해)."""
    vals, vecs = np.linalg.eigh(Sigma)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.c_[np.cos(t), np.sin(t)]
    pts = circle * (n_std * np.sqrt(np.maximum(vals, 1e-12)))
    return pts @ vecs.T + mu


# ── DBSCAN ────────────────────────────────────────────────────────────

def dbscan(X, eps=0.3, min_pts=5):
    """밀도 기반 군집화.

    Returns
    -------
    labels : -1 은 잡음
    core   : 핵심점 여부 bool 배열
    """
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    neighbors = [np.where(D[i] <= eps)[0] for i in range(n)]
    core = np.array([len(nb) >= min_pts for nb in neighbors])

    labels = np.full(n, -1)
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            p = stack.pop()
            for q in neighbors[p]:
                if labels[q] == -1:
                    labels[q] = cid
                    if core[q]:
                        stack.append(q)
        cid += 1
    return labels, core


# ── 계층적 군집화 ─────────────────────────────────────────────────────

def hierarchical(X, linkage="ward"):
    """응집형 계층 군집화. 병합 이력을 반환한다.

    Returns
    -------
    merges : [(a, b, distance, size)] — scipy 의 linkage 행렬과 유사
    """
    n = len(X)
    clusters = {i: [i] for i in range(n)}
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)

    active = list(range(n))
    cur = {i: D.copy() for i in [0]}      # 거리 행렬 하나만 유지
    Dc = D.copy()
    merges = []
    next_id = n
    sizes = {i: 1 for i in range(n)}
    centroids = {i: X[i].copy() for i in range(n)}

    while len(active) > 1:
        best, bi, bj = np.inf, None, None
        for ai in range(len(active)):
            for aj in range(ai + 1, len(active)):
                i, j = active[ai], active[aj]
                d = _linkage_dist(clusters[i], clusters[j], D, linkage,
                                  sizes, centroids, i, j)
                if d < best:
                    best, bi, bj = d, i, j
        merges.append((bi, bj, float(best), sizes[bi] + sizes[bj]))
        clusters[next_id] = clusters[bi] + clusters[bj]
        sizes[next_id] = sizes[bi] + sizes[bj]
        centroids[next_id] = X[clusters[next_id]].mean(0)
        active.remove(bi); active.remove(bj); active.append(next_id)
        next_id += 1
    return merges, clusters


def _linkage_dist(a, b, D, linkage, sizes, centroids, i, j):
    sub = D[np.ix_(a, b)]
    if linkage == "single":
        return sub.min()
    if linkage == "complete":
        return sub.max()
    if linkage == "average":
        return sub.mean()
    if linkage == "ward":
        na, nb = len(a), len(b)
        d = np.linalg.norm(centroids[i] - centroids[j])
        return np.sqrt(2 * na * nb / (na + nb)) * d
    raise ValueError(linkage)


def cut_tree(merges, n, n_clusters):
    """병합 이력에서 k개 군집이 되도록 자른다."""
    parent = {}
    members = {i: [i] for i in range(n)}
    nid = n
    for a, b, _, _ in merges[: n - n_clusters]:
        members[nid] = members[a] + members[b]
        parent[a] = nid
        parent[b] = nid
        nid += 1
    roots = [c for c in members if c not in parent]
    labels = np.full(n, -1)
    for k, r in enumerate(sorted(roots)):
        labels[members[r]] = k
    return labels


# ── 평가 지표 ─────────────────────────────────────────────────────────

def silhouette(X, labels):
    """실루엣 계수 (전체 평균). 잡음(-1)은 제외."""
    mask = labels >= 0
    Xs, ls = X[mask], labels[mask]
    uniq = np.unique(ls)
    if len(uniq) < 2:
        return 0.0
    D = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))
    s = np.zeros(len(Xs))
    for i in range(len(Xs)):
        same = ls == ls[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, ls == c].mean() for c in uniq if c != ls[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def adjusted_rand_index(true, pred):
    """ARI — 우연 일치를 보정한 외부 평가 지표. 무작위면 0, 완벽하면 1."""
    from math import comb
    t, p = np.asarray(true), np.asarray(pred)
    tu, pu = np.unique(t), np.unique(p)
    C = np.array([[np.sum((t == a) & (p == b)) for b in pu] for a in tu])
    sum_c = sum(comb(v, 2) for v in C.ravel() if v > 1)
    sum_a = sum(comb(v, 2) for v in C.sum(1) if v > 1)
    sum_b = sum(comb(v, 2) for v in C.sum(0) if v > 1)
    n_pairs = comb(len(t), 2)
    exp = sum_a * sum_b / n_pairs
    mx = (sum_a + sum_b) / 2
    return float((sum_c - exp) / (mx - exp)) if mx != exp else 0.0
