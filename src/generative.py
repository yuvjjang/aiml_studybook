"""생성 모델용 공통 부품 — NumPy 로 직접 구현한 VAE 계열.

Part 6 (생성 모델) 장들이 공유한다. 렌더 타임 의존성은 numpy 뿐이다.
"""

import numpy as np

__all__ = ["VAE", "vae_grad_check", "blobs64", "sharpness",
           "Net", "gan_step", "sigmoid", "ring_modes", "mode_stats",
           "Coupling", "RealNVP", "Diffusion", "make_schedule",
           "FlowMatching", "straightness",
           "time_embedding"]


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


# ══════════════════════════════════════════════════════════════════
# 정규화 흐름 — RealNVP 결합 층 (6.4)
# ══════════════════════════════════════════════════════════════════

class Coupling:
    """RealNVP 아핀 결합 층.

    마스크 m 이 1 인 좌표는 그대로 통과하고, 0 인 좌표만 변환한다.

        y_a = x_a                                   (m = 1)
        y_b = x_b · exp(s(x_a)) + t(x_a)            (m = 0)
        log|det J| = Σ_b s(x_a)

    야코비안이 삼각행렬이라 행렬식이 대각원소의 곱 — **O(d) 로 계산된다.**
    역변환도 닫힌 형태다: x_b = (y_b − t(x_a))·exp(−s(x_a)).
    """

    def __init__(self, d, mask, hidden=32, seed=0, s_scale=2.0):
        r = np.random.default_rng(seed)
        self.m = mask.astype(float)
        self.d, self.s_scale = d, s_scale
        self.p = {
            "W0": r.normal(0, np.sqrt(2.0/d), (d, hidden)),
            "b0": np.zeros(hidden),
            "W1": r.normal(0, np.sqrt(2.0/hidden), (hidden, hidden)),
            "b1": np.zeros(hidden),
            "W2": np.zeros((hidden, 2*d)),          # s, t 를 0 으로 시작 → 항등변환
            "b2": np.zeros(2*d),
        }

    def _st(self, xa):
        h1 = np.tanh(xa @ self.p["W0"] + self.p["b0"])
        h2 = np.tanh(h1 @ self.p["W1"] + self.p["b1"])
        out = h2 @ self.p["W2"] + self.p["b2"]
        s_raw, t = out[:, :self.d], out[:, self.d:]
        s = self.s_scale*np.tanh(s_raw)             # 스케일을 묶어 수치 안정
        return s, t, (h1, h2, s_raw)

    def forward(self, x):
        xa = x*self.m
        s, t, c = self._st(xa)
        u = 1 - self.m
        y = xa + u*(x*np.exp(s) + t)
        logdet = (u*s).sum(1)
        return y, logdet, dict(x=x, xa=xa, s=s, t=t, c=c, u=u)

    def inverse(self, y):
        ya = y*self.m
        s, t, _ = self._st(ya)
        u = 1 - self.m
        return ya + u*((y - t)*np.exp(-s))

    def backward(self, cache, dy, dlogdet):
        """dy: ∂L/∂y, dlogdet: ∂L/∂logdet (샘플별 스칼라). 반환 (grads, dx)."""
        x, xa, s, t, (h1, h2, s_raw) = (cache["x"], cache["xa"], cache["s"],
                                        cache["t"], cache["c"])
        u, m = cache["u"], self.m
        ds = dy*u*x*np.exp(s) + dlogdet[:, None]*u
        dt = dy*u
        dx = dy*u*np.exp(s)                          # x_b 경로
        # s = s_scale·tanh(s_raw)
        ds_raw = ds*self.s_scale*(1 - (s/self.s_scale)**2)   # tanh 미분
        dout = np.concatenate([ds_raw, dt], 1)
        g = {"W2": h2.T @ dout, "b2": dout.sum(0)}
        dh2 = (dout @ self.p["W2"].T)*(1 - h2**2)
        g["W1"] = h1.T @ dh2; g["b1"] = dh2.sum(0)
        dh1 = (dh2 @ self.p["W1"].T)*(1 - h1**2)
        g["W0"] = xa.T @ dh1; g["b0"] = dh1.sum(0)
        dxa = dh1 @ self.p["W0"].T + dy*m            # 항등 경로 + 조건 경로
        return g, dx*u + dxa*m


class RealNVP:
    """결합 층을 번갈아 쌓은 흐름. 최대우도로 학습한다."""

    def __init__(self, d, n_layers=6, hidden=32, seed=0):
        self.d, self.n = d, n_layers
        self.layers = []
        for i in range(n_layers):
            mask = np.array([(j + i) % 2 for j in range(d)])
            self.layers.append(Coupling(d, mask, hidden, seed=seed*100+i))

    def params(self):
        out = {}
        for i, L in enumerate(self.layers):
            for k, v in L.p.items():
                out[f"L{i}.{k}"] = v
        return out

    def forward(self, x):
        ld = np.zeros(len(x))
        caches = []
        for L in self.layers:
            x, d_, c = L.forward(x)
            ld = ld + d_
            caches.append(c)
        return x, ld, caches

    def inverse(self, z):
        for L in reversed(self.layers):
            z = L.inverse(z)
        return z

    def log_prob(self, x):
        z, ld, _ = self.forward(x)
        lp_z = -0.5*(z**2).sum(1) - 0.5*self.d*np.log(2*np.pi)
        return lp_z + ld

    def nll_and_grads(self, x):
        n = len(x)
        z, ld, caches = self.forward(x)
        lp_z = -0.5*(z**2).sum(1) - 0.5*self.d*np.log(2*np.pi)
        nll = float(-(lp_z + ld).mean())
        dz = z/n                                     # ∂(−mean lp_z)/∂z
        dld = -np.ones(n)/n
        g = {}
        for i in range(self.n-1, -1, -1):
            gi, dz = self.layers[i].backward(caches[i], dz, dld)
            for k, v in gi.items():
                g[f"L{i}.{k}"] = v
        return nll, g

    def fit(self, X, steps=3000, lr=3e-3, seed=0, batch=None, track=0):
        from src.nn import Optimizer
        opt = Optimizer("adam", lr=lr)
        P = self.params()
        r = np.random.default_rng(seed)
        hist = []
        for s in range(steps):
            xb = X if not batch else X[r.choice(len(X), batch, replace=False)]
            nll, g = self.nll_and_grads(xb)
            opt.step(P, g)
            if track and s % track == 0:
                hist.append((s, nll))
        self.hist = hist
        return self

    def sample(self, n, seed=0):
        z = np.random.default_rng(seed).normal(size=(n, self.d))
        return self.inverse(z)


# ══════════════════════════════════════════════════════════════════
# 확산 모델 — DDPM / DDIM (6.5, 6.6)
# ══════════════════════════════════════════════════════════════════

def time_embedding(t, dim=16, T=1000):
    """사인·코사인 시간 임베딩. t 는 (n,) 정수 배열."""
    half = dim//2
    freqs = np.exp(-np.log(10000.0)*np.arange(half)/half)
    ang = (t[:, None]/T)*1000.0*freqs[None]
    return np.concatenate([np.sin(ang), np.cos(ang)], 1)


def make_schedule(T=200, kind="linear", s=0.008):
    """베타 스케줄 → (betas, alphas, alpha_bars)."""
    if kind == "linear":
        betas = np.linspace(1e-4, 0.02, T)*(1000.0/T)
        betas = np.clip(betas, 1e-8, 0.999)
    elif kind == "cosine":                       # Nichol & Dhariwal 2021
        ts = np.linspace(0, 1, T+1)
        f = np.cos((ts + s)/(1 + s)*np.pi/2)**2
        ab = f/f[0]
        betas = np.clip(1 - ab[1:]/ab[:-1], 1e-8, 0.999)
    elif kind == "quadratic":
        betas = np.linspace(1e-4**0.5, 0.02**0.5, T)**2*(1000.0/T)
        betas = np.clip(betas, 1e-8, 0.999)
    else:
        raise ValueError(kind)
    alphas = 1 - betas
    return betas, alphas, np.cumprod(alphas)


class Diffusion:
    """2D 데이터용 DDPM. ε-예측 신경망을 직접 학습한다.

    입력: [x (d), 시간 임베딩 (t_dim), 조건 원-핫 (n_class+1)]
    출력: ε̂ (d)
    """

    def __init__(self, d=2, T=200, schedule="cosine", hidden=128,
                 t_dim=16, n_class=0, seed=0):
        self.d, self.T, self.t_dim, self.n_class = d, T, t_dim, n_class
        self.betas, self.alphas, self.abar = make_schedule(T, schedule)
        in_dim = d + t_dim + (n_class + 1 if n_class else 0)
        self.net = Net([in_dim, hidden, hidden, hidden, d], seed=seed)

    def _inp(self, x, t, c=None):
        parts = [x, time_embedding(t, self.t_dim, self.T)]
        if self.n_class:
            oh = np.zeros((len(x), self.n_class + 1))
            if c is None:                        # 무조건부 = 마지막 슬롯
                oh[:, -1] = 1.0
            else:
                oh[np.arange(len(x)), c] = 1.0
            parts.append(oh)
        return np.concatenate(parts, 1)

    def eps(self, x, t, c=None):
        return self.net.forward(self._inp(x, t, c))[0]

    def q_sample(self, x0, t, noise):
        a = self.abar[t][:, None]
        return np.sqrt(a)*x0 + np.sqrt(1 - a)*noise

    def fit(self, X, y=None, steps=6000, lr=2e-3, batch=256, seed=0,
            p_uncond=0.1, track=0):
        from src.nn import Optimizer
        opt = Optimizer("adam", lr=lr)
        r = np.random.default_rng(seed)
        hist = []
        for s in range(steps):
            idx = r.choice(len(X), batch, replace=False)
            x0 = X[idx]
            t = r.integers(0, self.T, batch)
            noise = r.normal(size=x0.shape)
            xt = self.q_sample(x0, t, noise)
            c = None
            if self.n_class and y is not None:
                c = y[idx].copy()
                drop = r.random(batch) < p_uncond
                inp = self._inp(xt, t, c)
                oh = inp[:, -(self.n_class+1):]
                oh[drop] = 0.0
                oh[drop, -1] = 1.0
            else:
                inp = self._inp(xt, t, c)
            out, cache = self.net.forward(inp)
            L = float(((out - noise)**2).mean())
            g, _ = self.net.backward(cache, 2*(out - noise)/(batch*self.d))
            opt.step(self.net.p, g)
            if track and s % track == 0:
                hist.append((s, L))
        self.hist = hist
        return self

    def _guided_eps(self, x, t, c, w):
        if c is None or w == 1.0:
            return self.eps(x, t, c)
        e_c = self.eps(x, t, c)
        e_u = self.eps(x, t, None)
        return e_u + w*(e_c - e_u)

    def sample(self, n, seed=0, c=None, w=1.0, method="ddpm", n_steps=None,
               return_path=False, clip=4.0):
        """clip: x̂₀ 예측을 [−clip, clip] 로 자른다 (None 이면 자르지 않음).

        결정론적 샘플러(DDIM)에서는 이 클리핑이 사실상 필수다 —
        x̂₀ = (x_t − √(1−ᾱ)ε̂)/√ᾱ 의 오차가 1/√ᾱ 배로 증폭되기 때문이다.
        """
        r = np.random.default_rng(seed)
        x = r.normal(size=(n, self.d))
        ts = (np.arange(self.T)[::-1] if n_steps is None else
              np.linspace(self.T-1, 0, n_steps).round().astype(int))
        path = [x.copy()]
        for i, t in enumerate(ts):
            tt = np.full(n, t)
            e = self._guided_eps(x, tt, c, w)
            ab = self.abar[t]
            x0h = (x - np.sqrt(1-ab)*e)/np.sqrt(ab)
            if clip is not None:
                x0h = np.clip(x0h, -clip, clip)
                e = (x - np.sqrt(ab)*x0h)/np.sqrt(1-ab)   # ε̂ 도 일관되게 갱신
            t_prev = ts[i+1] if i+1 < len(ts) else -1
            ab_prev = self.abar[t_prev] if t_prev >= 0 else 1.0
            if method == "ddim":
                x = np.sqrt(ab_prev)*x0h + np.sqrt(1-ab_prev)*e
            else:
                beta = 1 - ab/ab_prev if t_prev >= 0 else 1 - ab
                mean = (np.sqrt(ab_prev)*(1-ab/ab_prev)*x0h
                        + np.sqrt(ab/ab_prev)*(1-ab_prev)*x)/(1-ab)
                if t_prev >= 0:
                    sig = np.sqrt(beta*(1-ab_prev)/(1-ab))
                    x = mean + sig*r.normal(size=x.shape)
                else:
                    x = x0h
            if return_path:
                path.append(x.copy())
        return (x, np.array(path)) if return_path else x


# ══════════════════════════════════════════════════════════════════
# 흐름 매칭 / Rectified Flow (6.8)
# ══════════════════════════════════════════════════════════════════

class FlowMatching:
    """조건부 흐름 매칭 — 선형 보간 경로의 속도장을 학습한다.

        x_t = (1−t)·x₀ + t·x₁,   x₀ ~ 데이터,  x₁ ~ N(0, I)
        목표 속도 v = x₁ − x₀     (경로를 t 로 미분한 값, 상수다)
        손실 = E‖v_θ(x_t, t) − (x₁ − x₀)‖²

    t=0 이 데이터, t=1 이 잡음이다. 샘플링은 t=1 → 0 으로 적분한다.
    """

    def __init__(self, d=2, hidden=128, t_dim=16, seed=0):
        self.d, self.t_dim = d, t_dim
        self.net = Net([d + t_dim, hidden, hidden, hidden, d], seed=seed)

    def _inp(self, x, t):
        return np.concatenate([x, time_embedding(t*1000, self.t_dim, 1000)], 1)

    def v(self, x, t):
        return self.net.forward(self._inp(x, t))[0]

    def fit(self, X, steps=6000, lr=2e-3, batch=256, seed=0, pairs=None):
        """pairs=(x1, x0) 를 주면 그 짝을 그대로 쓴다 (reflow)."""
        from src.nn import Optimizer
        opt = Optimizer("adam", lr=lr)
        r = np.random.default_rng(seed)
        for _ in range(steps):
            if pairs is None:
                x0 = X[r.choice(len(X), batch, replace=False)]
                x1 = r.normal(size=x0.shape)
            else:
                idx = r.choice(len(pairs[0]), batch, replace=False)
                x1, x0 = pairs[0][idx], pairs[1][idx]
            t = r.random(batch)
            xt = (1 - t[:, None])*x0 + t[:, None]*x1
            target = x1 - x0
            out, cache = self.net.forward(self._inp(xt, t))
            g, _ = self.net.backward(cache, 2*(out - target)/(batch*self.d))
            opt.step(self.net.p, g)
        return self

    def sample(self, n, n_steps=50, seed=0, x1=None, return_path=False):
        r = np.random.default_rng(seed)
        x = r.normal(size=(n, self.d)) if x1 is None else x1.copy()
        ts = np.linspace(1.0, 0.0, n_steps + 1)
        path = [x.copy()]
        for i in range(n_steps):
            t_cur, t_next = ts[i], ts[i+1]
            x = x + (t_next - t_cur)*self.v(x, np.full(len(x), t_cur))
            if return_path:
                path.append(x.copy())
        return (x, np.array(path)) if return_path else x

    def make_pairs(self, n, n_steps=100, seed=0):
        """reflow 용 짝 (x₁, x₀) 생성 — 같은 잡음에서 출발한 결과를 짝지운다."""
        r = np.random.default_rng(seed)
        x1 = r.normal(size=(n, self.d))
        return x1, self.sample(n, n_steps=n_steps, x1=x1)


def straightness(path):
    """경로의 굽은 정도 — 경로 길이 / 직선 거리. 1.0 이면 완전한 직선."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=-1).sum(0)
    direct = np.linalg.norm(path[-1] - path[0], axis=-1)
    return float((seg/np.maximum(direct, 1e-9)).mean())
