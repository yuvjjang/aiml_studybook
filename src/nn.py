"""
신경망 참조 구현 (NumPy).

Part 3(딥러닝 기초) 전체가 공유한다. 순전파·역전파를 손으로 유도해 두었으므로
프레임워크가 감추는 계산을 그대로 볼 수 있다.

설계 원칙
- 모든 역전파는 수치 미분으로 검증 가능하다 (`grad_check`)
- 옵티마이저는 상태를 딕셔너리로 들고 다녀 비교 실험이 쉽다
- 활성화·초기화·정규화를 문자열로 바꿔 끼울 수 있다
"""
import numpy as np


# ══════════════════════════════════════════════════════════════════
# 활성화 함수와 도함수
# ══════════════════════════════════════════════════════════════════

_C_GELU = np.sqrt(2 / np.pi)


def sigmoid(z):
    """수치적으로 안전한 시그모이드 (양/음수 분기)."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


ACTIVATIONS = {
    "identity":  (lambda z: z,
                  lambda z, a: np.ones_like(z)),
    "sigmoid":   (sigmoid,
                  lambda z, a: a * (1 - a)),
    "tanh":      (np.tanh,
                  lambda z, a: 1 - a ** 2),
    "relu":      (lambda z: np.maximum(z, 0.0),
                  lambda z, a: (z > 0).astype(float)),
    "leaky_relu": (lambda z: np.where(z > 0, z, 0.01 * z),
                   lambda z, a: np.where(z > 0, 1.0, 0.01)),
    "elu":       (lambda z: np.where(z > 0, z, np.expm1(np.clip(z, -60, 0))),
                  lambda z, a: np.where(z > 0, 1.0, a + 1.0)),
    "gelu":      (lambda z: 0.5 * z * (1 + np.tanh(_C_GELU * (z + 0.044715 * z**3))),
                  lambda z, a: (lambda t: 0.5 * (1 + t) + 0.5 * z * (1 - t**2)
                                * _C_GELU * (1 + 3 * 0.044715 * z**2))(
                      np.tanh(_C_GELU * (z + 0.044715 * z**3)))),
    "swish":     (lambda z: z * sigmoid(z),
                  lambda z, a: (lambda s: s + z * s * (1 - s))(sigmoid(z))),
}


def softmax(z, axis=-1):
    """수치 안정 softmax (최댓값 차감)."""
    z = np.asarray(z, dtype=float)
    m = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ══════════════════════════════════════════════════════════════════
# 초기화
# ══════════════════════════════════════════════════════════════════

def init_weights(fan_in, fan_out, scheme="he", rng=None, gain=1.0):
    """가중치 초기화. scheme: zeros / small / xavier / he / orthogonal"""
    rng = rng or np.random.default_rng(0)
    if scheme == "zeros":
        return np.zeros((fan_in, fan_out))
    if scheme == "small":
        return rng.normal(0, 0.01, (fan_in, fan_out))
    if scheme == "xavier":
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    elif scheme == "he":
        std = gain * np.sqrt(2.0 / fan_in)
    elif scheme == "lecun":
        std = gain * np.sqrt(1.0 / fan_in)
    elif scheme == "orthogonal":
        A = rng.normal(size=(fan_in, fan_out))
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        Q = U if U.shape == (fan_in, fan_out) else Vt
        return gain * Q
    else:
        raise ValueError(scheme)
    return rng.normal(0, std, (fan_in, fan_out))


# ══════════════════════════════════════════════════════════════════
# 손실
# ══════════════════════════════════════════════════════════════════

def mse_loss(pred, y):
    """평균제곱오차. 반환: (손실, d손실/d예측)"""
    diff = pred - y
    return float(np.mean(diff ** 2)), 2.0 * diff / y.size


def softmax_ce_loss(logits, y_idx):
    """softmax + 교차엔트로피. 그래디언트가 (p − onehot)/n 으로 단순해진다."""
    p = softmax(logits)
    n = len(y_idx)
    loss = float(-np.mean(np.log(p[np.arange(n), y_idx] + 1e-12)))
    g = p.copy()
    g[np.arange(n), y_idx] -= 1.0
    return loss, g / n


def bce_loss(prob, y):
    """이진 교차엔트로피 (확률 입력)."""
    p = np.clip(prob, 1e-12, 1 - 1e-12)
    loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return loss, (p - y) / (p * (1 - p) * y.size)


# ══════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════

class MLP:
    """다층 퍼셉트론. 순전파·역전파를 직접 구현한다.

    sizes      : [입력, 은닉1, ..., 출력]
    activation : 은닉층 활성화 이름
    out_activation : "identity"(회귀) 또는 "softmax"(분류)
    norm       : None / "batch" / "layer"
    """

    def __init__(self, sizes, activation="relu", out_activation="identity",
                 init="he", norm=None, seed=0, dropout=0.0):
        self.sizes = list(sizes)
        self.act_name = activation
        self.out_activation = out_activation
        self.norm = norm
        self.dropout = dropout
        rng = np.random.default_rng(seed)
        self.rng = rng

        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            self.W.append(init_weights(sizes[i], sizes[i+1], init, rng))
            self.b.append(np.zeros(sizes[i+1]))
        # 정규화 층의 스케일·시프트 (마지막 층 제외)
        self.gamma = [np.ones(sizes[i+1]) for i in range(len(sizes) - 2)]
        self.beta = [np.zeros(sizes[i+1]) for i in range(len(sizes) - 2)]
        self.running = [dict(mean=np.zeros(sizes[i+1]), var=np.ones(sizes[i+1]))
                        for i in range(len(sizes) - 2)]

    # ── 파라미터 접근 (옵티마이저·그래디언트 검사용) ──
    def params(self):
        p = {}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            p[f"W{i}"] = w
            p[f"b{i}"] = b
        if self.norm:
            for i, (g, bt) in enumerate(zip(self.gamma, self.beta)):
                p[f"gamma{i}"] = g
                p[f"beta{i}"] = bt
        return p

    def n_params(self):
        return sum(v.size for v in self.params().values())

    # ── 순전파 ──
    def forward(self, X, training=True):
        act, dact = ACTIVATIONS[self.act_name]
        cache = {"X": X, "z": [], "a": [], "norm": [], "mask": []}
        h = X
        n_layers = len(self.W)

        for i in range(n_layers):
            z = h @ self.W[i] + self.b[i]

            if i < n_layers - 1:
                if self.norm:
                    z, nc = self._normalize(z, i, training)
                    cache["norm"].append(nc)
                cache["z"].append(z)
                a = act(z)
                if self.dropout > 0 and training:
                    mask = (self.rng.random(a.shape) > self.dropout) / (1 - self.dropout)
                    a = a * mask
                    cache["mask"].append(mask)
                cache["a"].append(a)
                h = a
            else:
                cache["z"].append(z)
                h = z                                   # 출력층은 선형 (손실에서 처리)
        cache["out"] = h
        return h, cache

    def _normalize(self, z, i, training):
        eps = 1e-5
        if self.norm == "batch":
            if training:
                mu, var = z.mean(0), z.var(0)
                self.running[i]["mean"] = 0.9*self.running[i]["mean"] + 0.1*mu
                self.running[i]["var"] = 0.9*self.running[i]["var"] + 0.1*var
            else:
                mu, var = self.running[i]["mean"], self.running[i]["var"]
            axis = 0
        else:                                            # layer norm
            mu = z.mean(1, keepdims=True)
            var = z.var(1, keepdims=True)
            axis = 1
        std = np.sqrt(var + eps)
        zhat = (z - mu) / std
        out = self.gamma[i] * zhat + self.beta[i]
        return out, dict(zhat=zhat, std=std, axis=axis, i=i,
                         batch_stats=(self.norm == "batch" and training))

    # ── 역전파 ──
    def backward(self, cache, dout):
        act, dact = ACTIVATIONS[self.act_name]
        grads = {}
        n_layers = len(self.W)
        d = dout

        for i in reversed(range(n_layers)):
            h_in = cache["a"][i-1] if i > 0 else cache["X"]
            grads[f"W{i}"] = h_in.T @ d
            grads[f"b{i}"] = d.sum(0)
            if i == 0:
                break
            dh = d @ self.W[i].T

            if self.dropout > 0 and len(cache["mask"]) > i-1:
                dh = dh * cache["mask"][i-1]
            d = dh * dact(cache["z"][i-1], cache["a"][i-1])

            if self.norm:
                d, dg, db = self._normalize_backward(
                    d, cache["norm"][i-1], self.gamma[i-1])
                grads[f"gamma{i-1}"] = dg
                grads[f"beta{i-1}"] = db
        return grads

    @staticmethod
    def _normalize_backward(dy, nc, gamma):
        """정규화 층의 역전파.

        y = gamma * zhat + beta,  zhat = (z - mu) / std

        학습 시에는 mu, std 가 z 의 함수이므로 두 개의 보정항이 추가된다.
        평가 시(BatchNorm)에는 running 통계를 쓰므로 상수 취급이라 단순해진다.
        """
        zhat, std, axis = nc["zhat"], nc["std"], nc["axis"]
        dgamma = (dy * zhat).sum(0)
        dbeta = dy.sum(0)

        dzhat = dy * gamma                              # gamma 를 통과
        if not nc["batch_stats"] and nc["axis"] == 0:
            # BatchNorm 평가 모드: mu, std 가 입력과 무관한 상수
            return dzhat / std, dgamma, dbeta

        mean_d = dzhat.mean(axis=axis, keepdims=True)
        mean_dz = (dzhat * zhat).mean(axis=axis, keepdims=True)
        return (dzhat - mean_d - zhat * mean_dz) / std, dgamma, dbeta

    # ── 손실 + 그래디언트 ──
    def loss_and_grads(self, X, y, loss="mse", training=True):
        out, cache = self.forward(X, training)
        if loss == "mse":
            L, dout = mse_loss(out, y)
        elif loss == "softmax_ce":
            L, dout = softmax_ce_loss(out, y)
        else:
            raise ValueError(loss)
        return L, self.backward(cache, dout)

    def predict(self, X):
        out, _ = self.forward(X, training=False)
        if self.out_activation == "softmax":
            return softmax(out)
        return out


# ══════════════════════════════════════════════════════════════════
# 옵티마이저
# ══════════════════════════════════════════════════════════════════

class Optimizer:
    """SGD / momentum / nesterov / adagrad / rmsprop / adam / adamw"""

    def __init__(self, kind="sgd", lr=0.01, momentum=0.9, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.kind = kind
        self.lr = lr
        self.momentum = momentum
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay
        self.state = {}
        self.t = 0

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for k, p in params.items():
            g = grads.get(k)
            if g is None:
                continue
            g = np.asarray(g, dtype=float)

            if self.weight_decay and self.kind != "adamw":
                g = g + self.weight_decay * p          # L2 를 그래디언트에 섞음

            st = self.state.setdefault(k, {})
            if self.kind == "sgd":
                upd = lr * g
            elif self.kind in ("momentum", "nesterov"):
                v = st.get("v", np.zeros_like(p))
                v = self.momentum * v + g
                st["v"] = v
                upd = lr * (g + self.momentum * v) if self.kind == "nesterov" else lr * v
            elif self.kind == "adagrad":
                s = st.get("s", np.zeros_like(p)) + g**2
                st["s"] = s
                upd = lr * g / (np.sqrt(s) + self.eps)
            elif self.kind == "rmsprop":
                s = self.beta2 * st.get("s", np.zeros_like(p)) + (1-self.beta2) * g**2
                st["s"] = s
                upd = lr * g / (np.sqrt(s) + self.eps)
            elif self.kind in ("adam", "adamw"):
                m = self.beta1 * st.get("m", np.zeros_like(p)) + (1-self.beta1) * g
                v = self.beta2 * st.get("v", np.zeros_like(p)) + (1-self.beta2) * g**2
                st["m"], st["v"] = m, v
                mh = m / (1 - self.beta1 ** self.t)
                vh = v / (1 - self.beta2 ** self.t)
                upd = lr * mh / (np.sqrt(vh) + self.eps)
                if self.kind == "adamw" and self.weight_decay:
                    upd = upd + lr * self.weight_decay * p   # 갱신에 직접 적용
            else:
                raise ValueError(self.kind)
            p -= upd


def clip_grads(grads, max_norm):
    """전역 노름 클리핑. 반환: (클리핑 전 노름, 적용 배율)"""
    total = np.sqrt(sum(float((g**2).sum()) for g in grads.values()))
    scale = min(1.0, max_norm / (total + 1e-9))
    if scale < 1.0:
        for k in grads:
            grads[k] = grads[k] * scale
    return total, scale


def lr_schedule(step, total, base_lr, warmup=0, kind="cosine"):
    """warmup + cosine / step / constant"""
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(total - warmup, 1)
    if kind == "cosine":
        return base_lr * 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))
    if kind == "step":
        return base_lr * (0.1 ** int(prog * 3))
    return base_lr


# ══════════════════════════════════════════════════════════════════
# 검증 도구
# ══════════════════════════════════════════════════════════════════

def grad_check(model, X, y, loss="mse", keys=None, n_sample=4, eps=1e-5,
               seed=0, training=False, atol=1e-8):
    """해석적 그래디언트를 수치 미분과 비교. 최대 상대오차를 반환.

    `atol` — 해석적·수치적 그래디언트가 **둘 다** 이 값보다 작으면 오차 0으로 본다.

    참 그래디언트가 정확히 0인 파라미터가 있다 (예: BatchNorm 바로 앞의 bias —
    BN 이 배치 평균을 빼므로 그 bias 는 출력에 아무 영향이 없다).
    그때 상대오차는 0/0 이 되고, 유한차분 잡음(~1e-11)이 분자에 남아
    의미 없는 큰 값이 나온다. 그래서 절대 허용오차로 걸러낸다.
    """
    rng = np.random.default_rng(seed)
    _, grads = model.loss_and_grads(X, y, loss=loss, training=training)
    params = model.params()
    keys = keys or list(params)
    worst = 0.0
    detail = {}
    for k in keys:
        flat = params[k].ravel()
        gf = grads[k].ravel()
        errs = []
        for idx in rng.choice(flat.size, min(n_sample, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + eps
            lp, _ = model.loss_and_grads(X, y, loss=loss, training=training)
            flat[idx] = old - eps
            lm, _ = model.loss_and_grads(X, y, loss=loss, training=training)
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            den = abs(num) + abs(gf[idx])
            errs.append(0.0 if den < atol else abs(num - gf[idx]) / den)
        detail[k] = max(errs)
        worst = max(worst, detail[k])
    return worst, detail


def train(model, X, y, loss="mse", opt=None, steps=500, batch_size=None,
          lr=None, warmup=0, schedule="constant", clip=None, seed=0,
          eval_every=1, X_val=None, y_val=None):
    """공통 학습 루프. 히스토리를 dict 로 반환한다."""
    opt = opt or Optimizer("sgd", lr=lr or 0.05)
    base_lr = lr if lr is not None else opt.lr
    rng = np.random.default_rng(seed)
    n = len(X)
    hist = {"loss": [], "lr": [], "gnorm": [], "val": []}
    params = model.params()

    for s in range(steps):
        if batch_size and batch_size < n:
            idx = rng.choice(n, batch_size, replace=False)
            xb, yb = X[idx], y[idx]
        else:
            xb, yb = X, y
        L, g = model.loss_and_grads(xb, yb, loss=loss)
        gn = np.sqrt(sum(float((v**2).sum()) for v in g.values()))
        if clip:
            gn, _ = clip_grads(g, clip)
        cur_lr = lr_schedule(s, steps, base_lr, warmup, schedule)
        opt.step(params, g, lr=cur_lr)

        hist["loss"].append(L)
        hist["lr"].append(cur_lr)
        hist["gnorm"].append(gn)
        if X_val is not None and s % eval_every == 0:
            Lv, _ = model.loss_and_grads(X_val, y_val, loss=loss, training=False)
            hist["val"].append((s, Lv))
    return hist
