"""강화학습 공통 도구 — 그리드월드 MDP, DP 해법, 밴딧.

Part 10 의 모든 장이 이 모듈을 재사용한다. 외부 의존성은 numpy 뿐이다.
"""

import numpy as np

# 행동: 0=위, 1=오른쪽, 2=아래, 3=왼쪽
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ARROWS = ["↑", "→", "↓", "←"]


class GridWorld:
    """격자 MDP. 전이확률 P (S,A,S') 와 보상 R (S,A) 를 명시적으로 만든다.

    Parameters
    ----------
    rows, cols : int
    walls      : 벽 좌표 [(r, c), ...] — 들어갈 수 없다
    terminals  : {(r, c): 보상} — 도착하면 에피소드 종료
    step_cost  : 매 스텝 보상 (보통 음수)
    slip       : 의도한 방향 대신 직각 방향으로 갈 확률 (좌우로 절반씩)
    """

    def __init__(self, rows=4, cols=4, walls=(), terminals=None,
                 step_cost=-0.04, slip=0.0):
        self.rows, self.cols = rows, cols
        self.walls = set(map(tuple, walls))
        self.terminals = dict(terminals or {})
        self.step_cost = step_cost
        self.slip = slip
        self.nS = rows*cols
        self.nA = len(ACTIONS)
        self.P, self.R = self._build()

    # ── 좌표 ↔ 상태 인덱스 ──────────────────────────────
    def idx(self, rc):
        return rc[0]*self.cols + rc[1]

    def rc(self, s):
        r, c = divmod(int(s), self.cols)
        return (r, c)

    def is_wall(self, rc):
        return tuple(rc) in self.walls

    def is_terminal(self, s):
        return self.rc(s) in self.terminals

    def _move(self, rc, a):
        dr, dc = ACTIONS[a]
        nr, nc = rc[0] + dr, rc[1] + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return rc                       # 벽 밖 — 제자리
        if (nr, nc) in self.walls:
            return rc
        return (nr, nc)

    def _build(self):
        P = np.zeros((self.nS, self.nA, self.nS))
        R = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            rc = self.rc(s)
            if self.is_wall(rc):
                P[s, :, s] = 1.0
                continue
            if self.is_terminal(s):
                P[s, :, s] = 1.0            # 흡수 상태
                continue
            for a in range(self.nA):
                outcomes = [(a, 1.0 - self.slip)]
                if self.slip > 0:
                    outcomes += [((a - 1) % 4, self.slip/2),
                                 ((a + 1) % 4, self.slip/2)]
                for a2, p in outcomes:
                    nrc = self._move(rc, a2)
                    ns = self.idx(nrc)
                    P[s, a, ns] += p
                    R[s, a] += p*(self.terminals.get(nrc, 0.0) + self.step_cost)
        return P, R

    # ── 표시용 ────────────────────────────────────────
    def grid(self, v, fill=np.nan):
        """상태 벡터를 (rows, cols) 격자로. 벽은 fill."""
        g = np.asarray(v, float).reshape(self.rows, self.cols).copy()
        for r, c in self.walls:
            g[r, c] = fill
        return g

    def arrows(self, pi):
        """결정적 정책을 화살표 문자열 격자로."""
        out = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.walls:
                    row.append("█")
                elif (r, c) in self.terminals:
                    row.append("+" if self.terminals[(r, c)] > 0 else "−")
                else:
                    row.append(ARROWS[int(pi[self.idx((r, c))])])
            out.append("".join(row))
        return out


# ── 동적계획법 ─────────────────────────────────────────
def policy_evaluation(P, R, pi, gamma=0.95, tol=1e-10, max_iter=10000):
    """결정적 또는 확률적 정책의 V^pi. pi: (S,) 정수 또는 (S,A) 확률."""
    nS, nA = R.shape
    Pi = np.eye(nA)[np.asarray(pi, int)] if np.ndim(pi) == 1 else np.asarray(pi)
    Ppi = np.einsum('sa,sat->st', Pi, P)
    Rpi = (Pi*R).sum(1)
    V = np.zeros(nS)
    for i in range(max_iter):
        Vn = Rpi + gamma*(Ppi @ V)
        if np.max(np.abs(Vn - V)) < tol:
            return Vn, i + 1
        V = Vn
    return V, max_iter


def policy_value(P, R, pi, gamma=0.95):
    """V^pi 를 선형 방정식으로 정확히 푼다: (I - γP_pi) V = R_pi."""
    nS, nA = R.shape
    Pi = np.eye(nA)[np.asarray(pi, int)] if np.ndim(pi) == 1 else np.asarray(pi)
    Ppi = np.einsum('sa,sat->st', Pi, P)
    Rpi = (Pi*R).sum(1)
    return np.linalg.solve(np.eye(nS) - gamma*Ppi, Rpi)


def q_from_v(P, R, V, gamma=0.95):
    return R + gamma*np.einsum('sat,t->sa', P, V)


def greedy(Q):
    return Q.argmax(1)


def value_iteration(P, R, gamma=0.95, tol=1e-10, max_iter=10000,
                    track=False):
    """벨만 최적 연산자 반복. track=True 면 매 반복의 V 를 함께 돌려준다."""
    nS = R.shape[0]
    V = np.zeros(nS)
    hist = [V.copy()]
    for i in range(max_iter):
        Q = q_from_v(P, R, V, gamma)
        Vn = Q.max(1)
        if track:
            hist.append(Vn.copy())
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    Q = q_from_v(P, R, V, gamma)
    return (V, greedy(Q), i + 1, hist) if track else (V, greedy(Q), i + 1)


def policy_iteration(P, R, gamma=0.95, max_iter=1000):
    """평가와 개선을 번갈아. (V, pi, 반복 수, 평가에 쓴 총 스윕 수)"""
    nS, nA = R.shape
    pi = np.zeros(nS, int)
    sweeps = 0
    for i in range(max_iter):
        V, k = policy_evaluation(P, R, pi, gamma)
        sweeps += k
        pi_new = greedy(q_from_v(P, R, V, gamma))
        if np.array_equal(pi_new, pi):
            return V, pi, i + 1, sweeps
        pi = pi_new
    return V, pi, max_iter, sweeps


def state_visitation(P, pi, gamma=0.95, mu0=None, tol=1e-12):
    """할인 상태 방문 분포 d^pi (합이 1이 되도록 정규화)."""
    nS = P.shape[0]
    nA = P.shape[1]
    Pi = np.eye(nA)[np.asarray(pi, int)] if np.ndim(pi) == 1 else np.asarray(pi)
    Ppi = np.einsum('sa,sat->st', Pi, P)
    mu0 = np.ones(nS)/nS if mu0 is None else np.asarray(mu0, float)
    d = mu0.copy()
    acc = np.zeros(nS)
    w = 1.0
    for _ in range(20000):
        acc += w*d
        d = d @ Ppi
        w *= gamma
        if w < tol:
            break
    return acc/acc.sum()


# ── 샘플링 (10.2 이후) ──────────────────────────────────
def rollout(P, R, pi, s0, rng, max_len=200):
    """정책을 따라 한 에피소드. [(s, a, r, s'), ...] 를 돌려준다."""
    nA = P.shape[1]
    traj = []
    s = int(s0)
    for _ in range(max_len):
        if np.ndim(pi) == 1:
            a = int(pi[s])
        else:
            a = int(rng.choice(nA, p=pi[s]))
        ns = int(rng.choice(P.shape[2], p=P[s, a]))
        traj.append((s, a, float(R[s, a]), ns))
        if ns == s and P[s, a, s] == 1.0:
            break
        s = ns
    return traj


def eps_greedy_policy(Q, eps):
    """Q 로부터 ε-탐욕 확률 정책 (S,A)."""
    nS, nA = Q.shape
    pi = np.full((nS, nA), eps/nA)
    pi[np.arange(nS), Q.argmax(1)] += 1 - eps
    return pi


# ── 다중 슬롯머신 ───────────────────────────────────────
def bandit_run(means, algo, n_steps=1000, rng=None, eps=0.1, c=2.0,
               sigma=1.0):
    """알고리즘별 누적 후회. algo: 'eps' | 'ucb' | 'thompson' | 'greedy'"""
    rng = rng or np.random.default_rng(0)
    k = len(means)
    best = np.max(means)
    n = np.zeros(k)
    s = np.zeros(k)
    regret = np.zeros(n_steps)
    tot = 0.0
    for t in range(n_steps):
        if algo == "eps":
            a = int(rng.integers(k)) if rng.random() < eps else int(
                np.argmax(np.where(n > 0, s/np.maximum(n, 1), np.inf)))
        elif algo == "greedy":
            a = int(np.argmax(np.where(n > 0, s/np.maximum(n, 1), np.inf)))
        elif algo == "ucb":
            if (n == 0).any():
                a = int(np.argmin(n))
            else:
                a = int(np.argmax(s/n + c*np.sqrt(np.log(t + 1)/n)))
        else:                                    # thompson (가우시안)
            mu = np.where(n > 0, s/np.maximum(n, 1), 0.0)
            sd = sigma/np.sqrt(np.maximum(n, 1e-9))
            sd = np.where(n > 0, sd, 1e3)
            a = int(np.argmax(rng.normal(mu, sd)))
        r = rng.normal(means[a], sigma)
        n[a] += 1
        s[a] += r
        tot += best - means[a]
        regret[t] = tot
    return regret, n
