"""
앙상블 참조 구현 (NumPy).

Part 2.7 이 쓰는 배깅 / 랜덤 포레스트 / AdaBoost / 그래디언트 부스팅.
`src/trees.py` 의 DecisionTree 를 기본 학습기로 재사용한다.
"""
import numpy as np

from .trees import DecisionTree


# ── 배깅 계열 ─────────────────────────────────────────────────────────

class BaggingTrees:
    """부트스트랩 표본으로 트리를 여러 개 학습해 평균/투표.

    max_features 를 주면 랜덤 포레스트가 된다 (분할마다 특성 부분집합).
    """

    def __init__(self, n_estimators=50, task="classify", max_depth=None,
                 max_features=None, min_samples_leaf=1, random_state=0):
        self.n_estimators = n_estimators
        self.task = task
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees_ = []
        self.oob_index_ = []          # 각 트리가 보지 못한 샘플

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.trees_, self.oob_index_ = [], []
        for b in range(self.n_estimators):
            idx = rng.integers(0, n, n)                 # 복원추출
            oob = np.setdiff1d(np.arange(n), np.unique(idx))
            t = DecisionTree(task=self.task, max_depth=self.max_depth,
                             max_features=self.max_features,
                             min_samples_leaf=self.min_samples_leaf,
                             random_state=int(rng.integers(0, 2**31)))
            t.fit(X[idx], y[idx])
            self.trees_.append(t)
            self.oob_index_.append(oob)
        return self

    def predict_proba(self, X, n_use=None):
        ts = self.trees_[:n_use] if n_use else self.trees_
        return np.mean([t.predict_proba(X) for t in ts], axis=0)

    def predict(self, X, n_use=None):
        p = self.predict_proba(X, n_use)
        return p.argmax(1) if self.task == "classify" else p

    def oob_score(self, X, y):
        """OOB 예측으로 추정한 일반화 성능 (별도 검증셋 없이)."""
        n = len(X)
        acc = np.zeros((n, self.trees_[0].n_classes)) if self.task == "classify" \
            else np.zeros(n)
        cnt = np.zeros(n)
        for t, oob in zip(self.trees_, self.oob_index_):
            if len(oob) == 0:
                continue
            acc[oob] += t.predict_proba(X[oob])
            cnt[oob] += 1
        ok = cnt > 0
        pred = acc[ok] / cnt[ok][:, None] if self.task == "classify" \
            else acc[ok] / cnt[ok]
        if self.task == "classify":
            return float((pred.argmax(1) == y[ok]).mean())
        return float(np.mean((pred - y[ok]) ** 2))


# ── 부스팅 계열 ───────────────────────────────────────────────────────

class AdaBoostStumps:
    """AdaBoost.M1 (이진, ±1 레이블). 학습기는 깊이 1 결정 그루터기."""

    def __init__(self, n_estimators=50, random_state=0):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.stumps_, self.alphas_, self.weights_hist_ = [], [], []

    def fit(self, X, y):
        """y 는 {-1, +1}."""
        n = len(X)
        w = np.ones(n) / n
        self.stumps_, self.alphas_, self.weights_hist_ = [], [], []

        for m in range(self.n_estimators):
            # 가중치를 표본 수로 반영해 그루터기 학습 (재표집 방식)
            rng = np.random.default_rng(self.random_state + m)
            idx = rng.choice(n, n, replace=True, p=w / w.sum())
            st = DecisionTree(max_depth=1, random_state=self.random_state + m)
            st.fit(X[idx], (y[idx] > 0).astype(int))

            pred = np.where(st.predict(X) == 1, 1.0, -1.0)
            err = float(np.sum(w * (pred != y)) / np.sum(w))
            err = min(max(err, 1e-10), 1 - 1e-10)
            alpha = 0.5 * np.log((1 - err) / err)

            self.weights_hist_.append(w.copy())
            self.stumps_.append(st)
            self.alphas_.append(alpha)

            w = w * np.exp(-alpha * y * pred)      # 틀린 것의 가중치 증가
            w /= w.sum()
        return self

    def decision_function(self, X, n_use=None):
        k = n_use or len(self.stumps_)
        f = np.zeros(len(X))
        for st, a in zip(self.stumps_[:k], self.alphas_[:k]):
            f += a * np.where(st.predict(X) == 1, 1.0, -1.0)
        return f

    def predict(self, X, n_use=None):
        return np.where(self.decision_function(X, n_use) >= 0, 1, -1)


class GradientBoosting:
    """그래디언트 부스팅 (회귀, 제곱손실). 함수공간의 경사하강.

    제곱손실에서 음의 그래디언트가 정확히 잔차라서,
    각 라운드는 "지금까지의 잔차를 맞추는 트리"를 학습한다.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.init_ = 0.0
        self.trees_ = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.init_ = float(np.mean(y))
        F = np.full(n, self.init_)
        self.trees_ = []
        for m in range(self.n_estimators):
            resid = y - F                              # 음의 그래디언트
            if self.subsample < 1.0:
                sel = rng.choice(n, max(2, int(self.subsample * n)), replace=False)
            else:
                sel = np.arange(n)
            t = DecisionTree(task="regress", max_depth=self.max_depth)
            t.fit(X[sel], resid[sel])
            F = F + self.learning_rate * t.predict(X)
            self.trees_.append(t)
        return self

    def staged_predict(self, X):
        """라운드별 누적 예측을 순서대로 산출 (학습 곡선용)."""
        F = np.full(len(X), self.init_)
        for t in self.trees_:
            F = F + self.learning_rate * t.predict(X)
            yield F.copy()

    def predict(self, X, n_use=None):
        k = n_use if n_use is not None else len(self.trees_)
        F = np.full(len(X), self.init_)
        for t in self.trees_[:k]:
            F = F + self.learning_rate * t.predict(X)
        return F
