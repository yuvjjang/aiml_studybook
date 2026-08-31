"""
영상처리 기본 연산 (NumPy 구현).

Part 9(비전) 챕터가 쓰는 최소 도구. 합성곱 커널이 실제로 무엇을 하는지,
그리고 그것이 CNN 의 첫 층과 어떻게 대응하는지 보여주기 위한 것이다.
"""
import numpy as np

# 고전 커널 — CNN 이 학습으로 얻는 것과 같은 종류의 연산이다.
KERNELS = {
    "항등": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "박스 블러": np.ones((3, 3)) / 9.0,
    "샤픈": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "소벨 x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "소벨 y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "라플라시안": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    "엠보스": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
}


def gaussian_kernel(size=5, sigma=1.0):
    """정규화된 2D 가우시안 커널."""
    ax = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def convolve2d(img, kernel, pad="edge"):
    """2D 상호상관(cross-correlation) — 딥러닝에서 관례적으로 합성곱이라 부르는 연산.

    수학적 합성곱은 커널을 180도 뒤집지만, 커널을 학습하는 신경망에서는
    그 차이가 흡수되므로 프레임워크들도 이 형태를 쓴다.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode=pad)

    # 슬라이딩 윈도우 뷰로 벡터화 (루프 없이 한 번에)
    view = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", view, kernel)


def gradient_magnitude(img):
    """소벨 그래디언트 크기 — 에지 강도."""
    gx = convolve2d(img, KERNELS["소벨 x"])
    gy = convolve2d(img, KERNELS["소벨 y"])
    return np.sqrt(gx ** 2 + gy ** 2), gx, gy


def fft2_shifted(img):
    """중심을 원점으로 옮긴 2D 스펙트럼 (저주파가 가운데)."""
    return np.fft.fftshift(np.fft.fft2(img))


def spectrum_db(F, floor_db=-60.0):
    """2D 스펙트럼 -> dB 크기 (표시용)."""
    mag = np.abs(F)
    db = 20 * np.log10(np.maximum(mag, 1e-12) / max(mag.max(), 1e-12))
    return np.maximum(db, floor_db)


def frequency_filter(img, cutoff, mode="low"):
    """주파수 영역에서 원형 마스크로 저역/고역 통과 필터링.

    cutoff : 0~1, 이미지 절반 크기 기준 반지름 비율
    mode   : "low" (저역 통과, 흐려짐) 또는 "high" (고역 통과, 윤곽만)
    """
    F = fft2_shifted(img)
    h, w = img.shape
    cy, cx = h / 2, w / 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    radius = cutoff * min(cy, cx)

    mask = (r <= radius) if mode == "low" else (r > radius)
    filtered = np.fft.ifft2(np.fft.ifftshift(F * mask))
    return np.real(filtered)


def downsample(img, factor=2):
    """평균 풀링으로 해상도를 낮춘다 (가우시안 피라미드의 한 단계)."""
    h, w = img.shape
    h2, w2 = h // factor * factor, w // factor * factor
    trimmed = img[:h2, :w2]
    return trimmed.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def gaussian_pyramid(img, levels=4, sigma=1.0):
    """블러 후 다운샘플을 반복해 피라미드를 만든다."""
    out = [img]
    cur = img
    for _ in range(levels - 1):
        cur = downsample(convolve2d(cur, gaussian_kernel(5, sigma)))
        out.append(cur)
    return out


def receptive_field(n_layers, kernel=3, stride=1):
    """층을 쌓을 때 한 뉴런이 보는 입력 영역의 크기.

    r_{l} = r_{l-1} + (k - 1) * prod(stride_{1..l-1})
    """
    r, jump = 1, 1
    sizes = []
    for _ in range(n_layers):
        r = r + (kernel - 1) * jump
        jump = jump * stride
        sizes.append(r)
    return sizes
