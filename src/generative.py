"""생성 모델용 공통 부품 — NumPy 로 직접 구현한 VAE 계열.

Part 6 (생성 모델) 장들이 공유한다. 렌더 타임 의존성은 numpy 뿐이다.
"""

import numpy as np

__all__ = ["VAE", "vae_grad_check", "blobs64", "sharpness"]


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
