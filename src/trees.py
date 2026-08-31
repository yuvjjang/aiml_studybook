"""
결정트리 참조 구현 (NumPy).

Part 2 의 트리(2.6)와 앙상블(2.7)이 공유한다.
scikit-learn 을 쓰지 않는 이유는 이 책의 렌더 타임 의존성 제한 때문이고,
동시에 분할 기준·가지치기가 실제로 어떻게 계산되는지 코드로 보여주기 위해서다.
"""
import numpy as np


# ── 불순도 ────────────────────────────────────────────────────────────

def gini(y, n_classes):
    """지니 불순도 1 − Σp². 최대 = 1 − 1/K (균등)."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=n_classes) / len(y)
    return float(1.0 - np.sum(p ** 2))


def entropy(y, n_classes):
    """섀넌 엔트로피 −Σp log₂p. 최대 = log₂K."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=n_classes) / len(y)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def mse_impurity(y, n_classes=None):
    """회귀용: 분산. 분할로 줄이려는 대상."""
    return float(np.var(y)) if len(y) else 0.0


IMPURITY = {"gini": gini, "entropy": entropy, "mse": mse_impurity}


# ── 트리 ──────────────────────────────────────────────────────────────

class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n", "impurity")

    def __init__(self, value, n, impurity):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value          # 잎의 예측 (분류: 확률 벡터, 회귀: 평균)
        self.n = n
        self.impurity = impurity

    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTree:
    """CART 방식 결정트리. 축 정렬 이진 분할, 탐욕적 학습.

    task : "classify" 또는 "regress"
    criterion : classify 면 "gini"/"entropy", regress 면 "mse"
    """

    def __init__(self, task="classify", criterion=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features=None,
                 random_state=None):
        self.task = task
        self.criterion = criterion or ("gini" if task == "classify" else "mse")
        self.max_depth = max_depth if max_depth is not None else 2**31
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features          # 랜덤 포레스트용
        self.rng = np.random.default_rng(random_state)
        self.root = None
        self.n_classes = None

    # ── 학습 ──
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classify":
            y = y.astype(int)
            self.n_classes = int(y.max()) + 1
        self.root = self._build(X, y, depth=0)
        return self

    def _leaf_value(self, y):
        if self.task == "classify":
            return np.bincount(y, minlength=self.n_classes) / len(y)
        return float(np.mean(y))

    def _imp(self, y):
        return IMPURITY[self.criterion](y, self.n_classes)

    def _best_split(self, X, y):
        """모든 (특성, 임계값) 후보 중 가중 불순도가 최소인 것."""
        n, p = X.shape
        parent = self._imp(y)
        best = (None, None, 0.0)              # (feature, threshold, 이득)

        feats = np.arange(p)
        if self.max_features is not None and self.max_features < p:
            feats = self.rng.choice(p, self.max_features, replace=False)

        for f in feats:
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            # 값이 바뀌는 지점만 후보로 (중복 임계값 계산 방지)
            change = np.where(np.diff(xs) > 1e-12)[0]
            for i in change:
                nl = i + 1
                if nl < self.min_samples_leaf or n - nl < self.min_samples_leaf:
                    continue
                child = (nl * self._imp(ys[:nl]) + (n - nl) * self._imp(ys[nl:])) / n
                gain = parent - child
                if gain > best[2] + 1e-12:
                    best = (int(f), (xs[i] + xs[i + 1]) / 2, float(gain))
        return best

    def _build(self, X, y, depth):
        node = Node(self._leaf_value(y), len(y), self._imp(y))
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or node.impurity <= 1e-12):
            return node

        f, thr, gain = self._best_split(X, y)
        if f is None or gain <= 0:
            return node

        mask = X[:, f] <= thr
        node.feature, node.threshold = f, thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    # ── 예측 ──
    def _walk(self, x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._walk(x, self.root) for x in X])

    def predict(self, X):
        out = self.predict_proba(X)
        return out.argmax(1) if self.task == "classify" else out

    # ── 구조 조회 ──
    def n_leaves(self):
        def count(nd):
            return 1 if nd.is_leaf else count(nd.left) + count(nd.right)
        return count(self.root)

    def depth(self):
        def d(nd):
            return 0 if nd.is_leaf else 1 + max(d(nd.left), d(nd.right))
        return d(self.root)

    def splits(self):
        """(feature, threshold, depth) 목록 — 결정경계 그리기용."""
        out = []

        def rec(nd, depth):
            if nd.is_leaf:
                return
            out.append((nd.feature, nd.threshold, depth))
            rec(nd.left, depth + 1)
            rec(nd.right, depth + 1)

        rec(self.root, 0)
        return out

    def feature_importance(self, n_features):
        """불순도 감소 기반 중요도 (정규화됨)."""
        imp = np.zeros(n_features)

        def rec(nd):
            if nd.is_leaf:
                return
            drop = (nd.impurity
                    - (nd.left.n * nd.left.impurity
                       + nd.right.n * nd.right.impurity) / nd.n)
            imp[nd.feature] += nd.n * drop
            rec(nd.left)
            rec(nd.right)

        rec(self.root)
        total = imp.sum()
        return imp / total if total > 0 else imp


def decision_surface(model, gx, gy, proba=True):
    """2D 격자에서 예측을 계산해 (len(gy), len(gx)) 배열로."""
    GX, GY = np.meshgrid(gx, gy)
    G = np.c_[GX.ravel(), GY.ravel()]
    if proba and getattr(model, "task", "classify") == "classify":
        z = model.predict_proba(G)[:, 1]
    else:
        z = np.asarray(model.predict(G), dtype=float)
    return z.reshape(GX.shape)
