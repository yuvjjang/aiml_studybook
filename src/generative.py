"""생성 모델용 공통 부품 — NumPy 로 직접 구현한 VAE 계열.

Part 6 (생성 모델) 장들이 공유한다. 렌더 타임 의존성은 numpy 뿐이다.
"""

import numpy as np

__all__ = ["VAE", "vae_grad_check", "blobs64", "sharpness",
           "Net", "gan_step", "sigmoid", "ring_modes", "mode_stats"]


def _init(rng, fan_in, fan_out, gain=1.0):
    return rng.normal(0, gain*np.sqrt(1.0/fan_in), (fan_in, fan_out))


def blobs64(n=1500, seed=0):
    """8×8 이미지 데이터. 실제 자유도는 5개 — 중심·폭·밝기.

    반환: (X (n,64), Z_true (n,5))
    """
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:8, 0:8] / 7.0
    cx = r.uniform(.25, .75, n); cy = r.uniform(.25, .75, n)
    sx = r.uniform(.10, .30, n); sy = r.uniform(.10, .30, n)
    amp = r.uniform(.5, 1., n)
    X = np.empty((n, 64))
    for i in range(n):
        X[i] = (amp[i]*np.exp(-((xx-cx[i])**2/(2*sx[i]**2)
                                + (yy-cy[i])**2/(2*sy[i]**2)))).ravel()
    return X, np.stack([cx, cy, sx, sy, amp], 1)


def sharpness(X, size=8):
    """이미지 선명도 — 인접 픽셀 차이의 RMS. 흐릿할수록 작다."""
    I = X.reshape(-1, size, size)
    dx = np.diff(I, axis=2)
    dy = np.diff(I, axis=1)
    return float(np.sqrt((dx**2).mean() + (dy**2).mean()))


class VAE:
    """가우스 인코더·디코더를 가진 VAE. 순전파·역전파를 직접 구현한다.

    구조: d ─tanh→ h ─→ (μ, log σ²) ─재파라미터화→ z ─tanh→ h ─→ x̂
    손실: (1/n)[ Σ‖x−x̂‖²/(2σ_x²) + β·Σ KL(q(z|x) ‖ N(0,I)) ]
    """

    def __init__(self, d, h, k, seed=0, sigma_x=0.1):
        r = np.random.default_rng(seed)
        self.d, self.h, self.k, self.sigma_x = d, h, k, sigma_x
        self.p = {
            "W1": _init(r, d, h), "b1": np.zeros(h),
            "W2": _init(r, h, 2*k, gain=0.1), "b2": np.zeros(2*k),
            "W3": _init(r, k, h), "b3": np.zeros(h),
            "W4": _init(r, h, d), "b4": np.zeros(d),
        }

    # ── 인코더 / 디코더 ──
    def encode(self, X):
        he = np.tanh(X @ self.p["W1"] + self.p["b1"])
        st = he @ self.p["W2"] + self.p["b2"]
        return he, st[:, :self.k], np.clip(st[:, self.k:], -8, 8)

    def decode(self, Z):
        hd = np.tanh(Z @ self.p["W3"] + self.p["b3"])
        return hd, hd @ self.p["W4"] + self.p["b4"]

    def forward(self, X, eps=None, sample=True):
        he, mu, lv = self.encode(X)
        if eps is None:
            eps = np.random.default_rng(0).normal(size=mu.shape)
        z = mu + np.exp(0.5*lv)*eps if sample else mu
        hd, xh = self.decode(z)
        return dict(he=he, mu=mu, lv=lv, eps=eps, z=z, hd=hd, xh=xh)

    # ── 손실 ──
    def loss_terms(self, X, c):
        n = len(X)
        rec = float(((X - c["xh"])**2).sum()/(2*self.sigma_x**2)/n)
        kl = float(0.5*(c["mu"]**2 + np.exp(c["lv"]) - 1 - c["lv"]).sum()/n)
        return rec, kl

    def loss_and_grads(self, X, beta=1.0, eps=None):
        n = len(X)
        c = self.forward(X, eps=eps)
        rec, kl = self.loss_terms(X, c)
        P = self.p

        dxh = (c["xh"] - X)/(self.sigma_x**2)/n
        g = {"W4": c["hd"].T @ dxh, "b4": dxh.sum(0)}
        dhd = dxh @ P["W4"].T
        dz3 = dhd*(1 - c["hd"]**2)
        g["W3"] = c["z"].T @ dz3; g["b3"] = dz3.sum(0)
        dz = dz3 @ P["W3"].T

        dmu = dz + beta*c["mu"]/n
        dlv = dz*c["eps"]*np.exp(0.5*c["lv"])*0.5 \
            + beta*0.5*(np.exp(c["lv"]) - 1)/n
        dst = np.concatenate([dmu, dlv], 1)

        g["W2"] = c["he"].T @ dst; g["b2"] = dst.sum(0)
        dhe = dst @ P["W2"].T
        dz1 = dhe*(1 - c["he"]**2)
        g["W1"] = X.T @ dz1; g["b1"] = dz1.sum(0)
        return rec + beta*kl, g, (rec, kl)

    def fit(self, X, beta=1.0, steps=3000, lr=3e-3, seed=0, opt=None,
            batch=None, track_every=0):
        from src.nn import Optimizer
        opt = opt or Optimizer("adam", lr=lr)
        r = np.random.default_rng(seed)
        hist = []
        for s in range(steps):
            xb = X if not batch else X[r.choice(len(X), batch, replace=False)]
            eps = r.normal(size=(len(xb), self.k))
            L, g, (rec, kl) = self.loss_and_grads(xb, beta=beta, eps=eps)
            opt.step(self.p, g)
            if track_every and s % track_every == 0:
                hist.append((s, L, rec, kl))
        self.hist = hist
        return self

    def sample(self, n, seed=0):
        z = np.random.default_rng(seed).normal(size=(n, self.k))
        return self.decode(z)[1]

    def active_units(self, X, thresh=1e-2):
        """활성 잠재 단위 수 — Var_x[μ(x)] 가 임계값을 넘는 차원 (Burda 2016)."""
        _, mu, _ = self.encode(X)
        return int((mu.var(0) > thresh).sum()), mu.var(0)


def vae_grad_check(model, X, beta=1.0, eps_fd=1e-5, n_sample=4, seed=0):
    """유한차분 그래디언트 검사. eps 를 고정해 확률성을 제거한다."""
    r = np.random.default_rng(seed)
    eps = r.normal(size=(len(X), model.k))
    _, g, _ = model.loss_and_grads(X, beta=beta, eps=eps)
    worst = 0.0
    for key, P in model.p.items():
        flat = P.ravel()
        idx = r.choice(flat.size, min(n_sample, flat.size), replace=False)
        for i in idx:
            old = flat[i]
            flat[i] = old + eps_fd
            Lp = model.loss_and_grads(X, beta=beta, eps=eps)[0]
            flat[i] = old - eps_fd
            Lm = model.loss_and_grads(X, beta=beta, eps=eps)[0]
            flat[i] = old
            num = (Lp - Lm)/(2*eps_fd)
            ana = g[key].ravel()[i]
            worst = max(worst, abs(num - ana)/(abs(num) + abs(ana) + 1e-12))
    return worst


# ══════════════════════════════════════════════════════════════════
# GAN — 생성자·판별자와 미니맥스 학습 (6.3)
# ══════════════════════════════════════════════════════════════════

def _lrelu(z, a=0.2):
    return np.where(z > 0, z, a*z)


def _dlrelu(z, a=0.2):
    return np.where(z > 0, 1.0, a)


class Net:
    """입력 그래디언트까지 돌려주는 MLP. GAN 양쪽이 공유한다.

    생성자 역전파는 판별자를 통과해야 하므로 dX 가 반드시 필요하다.
    """

    def __init__(self, sizes, seed=0, gain=1.0):
        r = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.p = {}
        for i in range(len(sizes)-1):
            self.p[f"W{i}"] = r.normal(0, gain*np.sqrt(2.0/sizes[i]),
                                       (sizes[i], sizes[i+1]))
            self.p[f"b{i}"] = np.zeros(sizes[i+1])
        self.n = len(sizes)-1

    def forward(self, X, sn=False):
        """sn=True 면 스펙트럼 정규화 (W / σ_max(W)) 를 적용한다."""
        h, cache = X, {"X": X, "z": [], "a": [], "W": []}
        for i in range(self.n):
            W = self.p[f"W{i}"]
            if sn:
                W = W/np.linalg.norm(W, 2)
            cache["W"].append(W)
            z = h @ W + self.p[f"b{i}"]
            cache["z"].append(z)
            h = _lrelu(z) if i < self.n-1 else z
            cache["a"].append(h)
        return h, cache

    def backward(self, cache, dout):
        g, d = {}, dout
        for i in range(self.n-1, -1, -1):
            if i < self.n-1:
                d = d*_dlrelu(cache["z"][i])
            h_in = cache["X"] if i == 0 else cache["a"][i-1]
            g[f"W{i}"] = h_in.T @ d
            g[f"b{i}"] = d.sum(0)
            d = d @ cache["W"][i].T
        return g, d

    def clip(self, c):
        for k in self.p:
            np.clip(self.p[k], -c, c, out=self.p[k])

    def lipschitz(self, sn=False):
        """층별 스펙트럼 노름의 곱 — 립시츠 상수의 상한.

        sn=True 면 순전파와 같은 정규화된 가중치로 계산한다 (항상 1 이하).
        """
        L = 1.0
        for i in range(self.n):
            W = self.p[f"W{i}"]
            sig = np.linalg.norm(W, 2)
            L *= 1.0 if sn else sig
        return float(L)


def gan_step(G, D, optG, optD, real, z_dim, rng, mode="ns",
             clip=None, sn=False, d_steps=1):
    """GAN 한 스텝. mode: 'ns'(비포화) / 'mm'(포화) / 'wgan'.

    반환: (d_loss, g_loss)
    """
    n = len(real)
    for _ in range(d_steps):
        z = rng.normal(size=(n, z_dim))
        fake = G.forward(z)[0]
        d_r, c_r = D.forward(real, sn=sn)
        d_f, c_f = D.forward(fake, sn=sn)
        if mode == "wgan":
            gr, _ = D.backward(c_r, -np.ones_like(d_r)/n)
            gf, _ = D.backward(c_f, np.ones_like(d_f)/n)
            d_loss = float(d_f.mean() - d_r.mean())
        else:
            pr, pf = sigmoid(d_r), sigmoid(d_f)
            gr, _ = D.backward(c_r, (pr - 1)/n)
            gf, _ = D.backward(c_f, pf/n)
            d_loss = float(-(np.log(pr+1e-9).mean() + np.log(1-pf+1e-9).mean()))
        optD.step(D.p, {k: gr[k] + gf[k] for k in gr})
        if clip:
            D.clip(clip)

    z = rng.normal(size=(n, z_dim))
    fake, c_g = G.forward(z)
    d_f, c_f = D.forward(fake, sn=sn)
    if mode == "wgan":
        g_loss = float(-d_f.mean())
        _, dfake = D.backward(c_f, -np.ones_like(d_f)/n)
    elif mode == "ns":                       # −log D(G(z))
        pf = sigmoid(d_f)
        g_loss = float(-np.log(pf+1e-9).mean())
        _, dfake = D.backward(c_f, (pf - 1)/n)
    else:                                    # log(1 − D(G(z)))  포화 손실
        pf = sigmoid(d_f)
        g_loss = float(np.log(1-pf+1e-9).mean())
        _, dfake = D.backward(c_f, pf/n)
    gG, _ = G.backward(c_g, dfake)
    optG.step(G.p, gG)
    return d_loss, g_loss


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0/(1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e/(1.0 + e)
    return out


def ring_modes(k=8, radius=3.0):
    a = np.linspace(0, 2*np.pi, k, endpoint=False)
    return np.stack([radius*np.cos(a), radius*np.sin(a)], 1)


def mode_stats(S, centers, thresh=0.8):
    """커버한 모드 수와 모드별 샘플 비율의 엔트로피(균형도)."""
    d = ((S[:, None, :] - centers[None])**2).sum(-1)
    near = d.min(1) < thresh**2
    a = d.argmin(1)[near]
    cnt = np.bincount(a, minlength=len(centers)).astype(float)
    p = cnt/max(cnt.sum(), 1)
    nz = p[p > 0]
    ent = float(np.exp(-(nz*np.log(nz)).sum())) if len(nz) else 0.0
    return int((cnt > 0).sum()), ent, float(near.mean())
