"""
순환 신경망 계열의 NumPy 구현 (순전파 + 역전파).

Part 4 의 4.2~4.4 챕터가 쓴다. 바닐라 RNN / LSTM / GRU 를 같은 인터페이스로
제공해서, 동일 과제에서 직접 비교할 수 있게 하는 것이 목적이다.

모든 셀은 (손실, 그래디언트 dict) 를 반환하며 `grad_check` 로 검증되어 있다.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _init(rng, n_in, n_out, scale=None):
    scale = scale if scale is not None else 1.0 / np.sqrt(n_in)
    return rng.normal(0, scale, (n_in, n_out))


# ══════════════════════════════════════════════════════════════════
# 바닐라 RNN
# ══════════════════════════════════════════════════════════════════

class VanillaRNN:
    """h_t = tanh(W_x x_t + W_h h_{t-1} + b)"""

    n_gates = 1

    def __init__(self, d_in, d_hid, seed=0, spectral=None, in_scale=0.5):
        rng = np.random.default_rng(seed)
        self.d_in, self.d_hid = d_in, d_hid
        Wh = _init(rng, d_hid, d_hid)
        if spectral is not None:                       # 스펙트럼 반지름 고정
            Wh *= spectral / np.max(np.abs(np.linalg.eigvals(Wh)))
        self.p = {
            "Wx": _init(rng, d_in, d_hid, in_scale),
            "Wh": Wh,
            "b": np.zeros(d_hid),
        }

    def forward(self, x):
        """x: (B, T, d_in) → (h_T, cache). cache["outs"] 에 매 시점 h_t 가 쌓인다."""
        B, T, _ = x.shape
        h = np.zeros((B, self.d_hid))
        hs, pres = [], []
        for t in range(T):
            pre = x[:, t] @ self.p["Wx"] + h @ self.p["Wh"] + self.p["b"]
            h = np.tanh(pre)
            hs.append(h); pres.append(pre)
        return h, dict(x=x, hs=hs, outs=hs, pres=pres, T=T, B=B)

    def backward(self, cache, dh=None, dh_seq=None):
        """dh: 마지막 시점 그래디언트. dh_seq: (B, T, H) 매 시점 그래디언트.

        어텐션처럼 모든 시점의 은닉 상태를 쓰는 구조는 dh_seq 가 필요하다.
        """
        x, hs, T, B = cache["x"], cache["hs"], cache["T"], cache["B"]
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        acc = np.zeros((B, self.d_hid)) if dh is None else dh
        for t in range(T - 1, -1, -1):
            if dh_seq is not None:
                acc = acc + dh_seq[:, t]
            dpre = acc * (1 - hs[t] ** 2)
            hprev = hs[t - 1] if t > 0 else np.zeros((B, self.d_hid))
            g["Wx"] += x[:, t].T @ dpre
            g["Wh"] += hprev.T @ dpre
            g["b"] += dpre.sum(0)
            acc = dpre @ self.p["Wh"].T
        return g


# ══════════════════════════════════════════════════════════════════
# LSTM
# ══════════════════════════════════════════════════════════════════

class LSTM:
    """게이트 4개를 하나의 행렬로 묶어 계산한다 (순서: i, f, o, g).

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t      ← 셀 상태: 곱이 아니라 **덧셈** 갱신
    h_t = o_t ⊙ tanh(c_t)

    forget_bias : 망각 게이트 bias 초기값. 1.0 이면 초기에 f≈0.73 이라
                  셀 상태가 잘 보존된다 (Jozefowicz et al. 2015 의 권장).
    """

    n_gates = 4

    def __init__(self, d_in, d_hid, seed=0, forget_bias=1.0, in_scale=0.5):
        rng = np.random.default_rng(seed)
        self.d_in, self.d_hid = d_in, d_hid
        H = d_hid
        b = np.zeros(4 * H)
        b[H:2 * H] = forget_bias                        # f 게이트만 편향
        self.p = {
            "Wx": _init(rng, d_in, 4 * H, in_scale),
            "Wh": _init(rng, H, 4 * H),
            "b": b,
        }

    def forward(self, x):
        B, T, _ = x.shape
        H = self.d_hid
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = dict(x=x, T=T, B=B, hs=[], cs=[], gates=[], tc=[], outs=[])
        for t in range(T):
            z = x[:, t] @ self.p["Wx"] + h @ self.p["Wh"] + self.p["b"]
            i = sigmoid(z[:, 0:H])
            f = sigmoid(z[:, H:2 * H])
            o = sigmoid(z[:, 2 * H:3 * H])
            gg = np.tanh(z[:, 3 * H:4 * H])
            cache["hs"].append(h)                       # h_{t-1}
            cache["cs"].append(c)                       # c_{t-1}
            c = f * c + i * gg                          # ← 덧셈 갱신
            tc = np.tanh(c)
            h = o * tc
            cache["gates"].append((i, f, o, gg))
            cache["tc"].append(tc)
            cache["outs"].append(h)                     # h_t
        cache["c_final"], cache["h_final"] = c, h
        return h, cache

    def backward(self, cache, dh=None, dh_seq=None):
        x, T, B, H = cache["x"], cache["T"], cache["B"], self.d_hid
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        dc = np.zeros((B, H))
        dh = np.zeros((B, H)) if dh is None else dh
        for t in range(T - 1, -1, -1):
            if dh_seq is not None:
                dh = dh + dh_seq[:, t]
            i, f, o, gg = cache["gates"][t]
            tc = cache["tc"][t]
            c_prev = cache["cs"][t]
            h_prev = cache["hs"][t]

            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)            # 출력 경로 + 다음 스텝에서 온 것
            di = dc * gg
            dg = dc * i
            df = dc * c_prev
            dc = dc * f                                 # ← 셀 상태의 항등 경로

            dz = np.concatenate([di * i * (1 - i),
                                 df * f * (1 - f),
                                 do * o * (1 - o),
                                 dg * (1 - gg ** 2)], axis=1)
            g["Wx"] += x[:, t].T @ dz
            g["Wh"] += h_prev.T @ dz
            g["b"] += dz.sum(0)
            dh = dz @ self.p["Wh"].T
        return g


# ══════════════════════════════════════════════════════════════════
# GRU
# ══════════════════════════════════════════════════════════════════

class GRU:
    """게이트 2개 (r, z) + 후보 상태. LSTM 보다 파라미터가 3/4.

    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ n_t
    """

    n_gates = 3

    def __init__(self, d_in, d_hid, seed=0, in_scale=0.5):
        rng = np.random.default_rng(seed)
        self.d_in, self.d_hid = d_in, d_hid
        H = d_hid
        self.p = {
            "Wxrz": _init(rng, d_in, 2 * H, in_scale),
            "Whrz": _init(rng, H, 2 * H),
            "brz": np.zeros(2 * H),
            "Wxn": _init(rng, d_in, H, in_scale),
            "Whn": _init(rng, H, H),
            "bn": np.zeros(H),
        }

    def forward(self, x):
        B, T, _ = x.shape
        H = self.d_hid
        h = np.zeros((B, H))
        cache = dict(x=x, T=T, B=B, hs=[], parts=[], outs=[])
        for t in range(T):
            zr = sigmoid(x[:, t] @ self.p["Wxrz"] + h @ self.p["Whrz"] + self.p["brz"])
            r, z = zr[:, :H], zr[:, H:]
            hr = r * h
            n = np.tanh(x[:, t] @ self.p["Wxn"] + hr @ self.p["Whn"] + self.p["bn"])
            cache["hs"].append(h)
            cache["parts"].append((r, z, n, hr))
            h = (1 - z) * h + z * n
            cache["outs"].append(h)
        return h, cache

    def backward(self, cache, dh=None, dh_seq=None):
        x, T, B, H = cache["x"], cache["T"], cache["B"], self.d_hid
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        dh = np.zeros((B, H)) if dh is None else dh
        for t in range(T - 1, -1, -1):
            if dh_seq is not None:
                dh = dh + dh_seq[:, t]
            r, z, n, hr = cache["parts"][t]
            h_prev = cache["hs"][t]

            dz_gate = dh * (n - h_prev)
            dn = dh * z
            dh_direct = dh * (1 - z)                    # ← 항등 경로

            dn_pre = dn * (1 - n ** 2)
            g["Wxn"] += x[:, t].T @ dn_pre
            g["Whn"] += hr.T @ dn_pre
            g["bn"] += dn_pre.sum(0)
            dhr = dn_pre @ self.p["Whn"].T
            dr_gate = dhr * h_prev
            dh_from_r = dhr * r

            drz = np.concatenate([dr_gate * r * (1 - r),
                                  dz_gate * z * (1 - z)], axis=1)
            g["Wxrz"] += x[:, t].T @ drz
            g["Whrz"] += h_prev.T @ drz
            g["brz"] += drz.sum(0)

            dh = dh_direct + dh_from_r + drz @ self.p["Whrz"].T
        return g


CELLS = {"rnn": VanillaRNN, "lstm": LSTM, "gru": GRU}


def n_params(cell):
    return sum(v.size for v in cell.p.values())


# ══════════════════════════════════════════════════════════════════
# 검증
# ══════════════════════════════════════════════════════════════════

def grad_check(cell, x, seed=0, n_sample=6, eps=1e-6):
    """무작위 시드 그래디언트로 해석적/수치적 미분을 비교. 최대 상대오차 반환.

    손실은 L = sum(h_T * s) 로 잡는다 (s 는 고정 난수) — 셀 자체만 검증한다.
    """
    rng = np.random.default_rng(seed)
    h, cache = cell.forward(x)
    s = rng.normal(size=h.shape)

    def loss_of():
        hh, _ = cell.forward(x)
        return float((hh * s).sum())

    ga = cell.backward(cache, s.copy())
    worst, detail = 0.0, {}
    for k, P in cell.p.items():
        flat = P.ravel()
        gf = ga[k].ravel()
        errs = []
        for idx in rng.choice(flat.size, min(n_sample, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + eps; lp = loss_of()
            flat[idx] = old - eps; lm = loss_of()
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            den = abs(num) + abs(gf[idx])
            errs.append(0.0 if den < 1e-9 else abs(num - gf[idx]) / den)
        detail[k] = max(errs)
        worst = max(worst, detail[k])
    return worst, detail


def grad_check_seq(cell, x, seed=0, n_sample=6, eps=1e-6):
    """모든 시점의 출력을 쓰는 손실로 검증한다 (어텐션이 쓰는 경로).

    L = sum_t (h_t * s_t) — dh_seq 경로가 맞는지 확인한다.
    """
    rng = np.random.default_rng(seed)
    _, cache = cell.forward(x)
    T = cache["T"]
    s = rng.normal(size=(x.shape[0], T, cell.d_hid))

    def loss_of():
        _, c = cell.forward(x)
        return float(sum((c["outs"][t] * s[:, t]).sum() for t in range(T)))

    ga = cell.backward(cache, dh_seq=s)
    worst = 0.0
    for k, P in cell.p.items():
        flat = P.ravel()
        gf = ga[k].ravel()
        for idx in rng.choice(flat.size, min(n_sample, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + eps; lp = loss_of()
            flat[idx] = old - eps; lm = loss_of()
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            den = abs(num) + abs(gf[idx])
            if den >= 1e-9:
                worst = max(worst, abs(num - gf[idx]) / den)
    return worst


def cell_grad_norms(cell, x, seed=0):
    """시점을 거슬러 올라가며 dh 의 노름을 기록한다 (인덱스 0 = 가장 먼 과거)."""
    rng = np.random.default_rng(seed)
    h, cache = cell.forward(x)
    B, H = h.shape
    dh = rng.normal(0, 1, (B, H)) / np.sqrt(B * H)
    norms = [float(np.linalg.norm(dh))]

    T = cache["T"]
    if isinstance(cell, LSTM):
        dc = np.zeros((B, H))
        for t in range(T - 1, 0, -1):
            i, f, o, gg = cache["gates"][t]
            tc = cache["tc"][t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di, dg, df = dc * gg, dc * i, dc * cache["cs"][t]
            dc = dc * f
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 do * o * (1 - o), dg * (1 - gg ** 2)], axis=1)
            dh = dz @ cell.p["Wh"].T
            norms.append(float(np.linalg.norm(dh) + np.linalg.norm(dc)))
    elif isinstance(cell, GRU):
        for t in range(T - 1, 0, -1):
            r, z, n, hr = cache["parts"][t]
            h_prev = cache["hs"][t]
            dz_gate = dh * (n - h_prev)
            dn_pre = (dh * z) * (1 - n ** 2)
            dhr = dn_pre @ cell.p["Whn"].T
            drz = np.concatenate([(dhr * h_prev) * r * (1 - r),
                                  dz_gate * z * (1 - z)], axis=1)
            dh = dh * (1 - z) + dhr * r + drz @ cell.p["Whrz"].T
            norms.append(float(np.linalg.norm(dh)))
    else:
        for t in range(T - 1, 0, -1):
            dpre = dh * (1 - cache["hs"][t] ** 2)
            dh = dpre @ cell.p["Wh"].T
            norms.append(float(np.linalg.norm(dh)))
    return np.array(norms[::-1])
