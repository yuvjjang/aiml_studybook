"""
재현 가능한 합성 데이터 생성기.

이 책의 그래픽은 외부 데이터셋을 내려받지 않는다. 모든 예제는 여기서
시드를 고정해 만들어지므로, 어느 PC에서 빌드해도 같은 그림이 나온다.
"""
import numpy as np


def _rng(seed):
    return np.random.default_rng(seed)


def polynomial(n=30, noise=0.25, x_range=(0.0, 1.0), seed=0):
    """1D 회귀용. 참 함수는 sin(2πx) — 다항 차수 실험에 쓴다.

    Returns
    -------
    x, y, f_true : 관측 x, 잡음 섞인 y, 잡음 없는 참값
    """
    rng = _rng(seed)
    x = np.sort(rng.uniform(*x_range, n))
    f_true = np.sin(2 * np.pi * x)
    y = f_true + rng.normal(0, noise, n)
    return x, y, f_true


def linear(n=60, slope=2.0, intercept=1.0, noise=1.0, x_range=(-3.0, 3.0), seed=0):
    """단순 선형 회귀용 (기울기·절편이 알려진 데이터)."""
    rng = _rng(seed)
    x = rng.uniform(*x_range, n)
    y = slope * x + intercept + rng.normal(0, noise, n)
    return x, y


def blobs(n=300, centers=((-2, -2), (2, 2), (-2, 2)), spread=0.8, seed=0):
    """등방 가우스 군집. 군집화·분류 예제용.

    Returns
    -------
    X : (n, 2), labels : (n,)
    """
    rng = _rng(seed)
    centers = np.asarray(centers, dtype=float)
    k = len(centers)
    labels = rng.integers(0, k, n)
    X = centers[labels] + rng.normal(0, spread, (n, 2))
    return X, labels


def two_moons(n=300, noise=0.15, seed=0):
    """초승달 두 개 — 선형 분리가 안 되는 고전 예제."""
    rng = _rng(seed)
    n_out = n // 2
    n_in = n - n_out

    t_out = np.pi * rng.uniform(0, 1, n_out)
    t_in = np.pi * rng.uniform(0, 1, n_in)

    outer = np.c_[np.cos(t_out), np.sin(t_out)]
    inner = np.c_[1 - np.cos(t_in), 0.5 - np.sin(t_in)]

    X = np.vstack([outer, inner]) + rng.normal(0, noise, (n, 2))
    labels = np.hstack([np.zeros(n_out, int), np.ones(n_in, int)])
    return X, labels


def correlated_gaussian(n=400, rho=0.8, seed=0):
    """지정한 상관계수를 갖는 2D 정규 표본. 공분산·PCA 예제용."""
    rng = _rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    return rng.normal(size=(n, 2)) @ L.T


def image_grid(size=64):
    """외부 파일 없이 만드는 테스트 이미지.

    원·사각형·대각 줄무늬·부드러운 배경 그래디언트를 합쳐, 에지·주파수·
    필터 실험에 필요한 성분을 모두 담는다. 값 범위는 [0, 1].
    """
    y, x = np.mgrid[0:size, 0:size] / (size - 1)

    img = 0.25 + 0.25 * x                                   # 배경 그래디언트
    img += 0.35 * (((x - 0.30) ** 2 + (y - 0.32) ** 2) < 0.028)   # 원
    img -= 0.30 * ((x > 0.60) & (x < 0.88) & (y > 0.58) & (y < 0.86))  # 사각형
    img += 0.12 * np.sin(2 * np.pi * 12 * (x + y))          # 대각 고주파 줄무늬
    return np.clip(img, 0.0, 1.0)
