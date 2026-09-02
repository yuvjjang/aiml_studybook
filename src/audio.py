"""
오디오 신호 생성과 시간-주파수 변환.

Part 8(오디오) 챕터들이 쓰는 최소 도구. 외부 음원 파일 없이 신호를 합성해
샘플링·STFT·스펙트로그램 실험을 재현 가능하게 만든다.
"""
import numpy as np


def time_axis(duration, sr):
    """길이 duration 초, 샘플레이트 sr 의 시간 축."""
    return np.arange(int(duration * sr)) / sr


def tone(freq, duration=1.0, sr=16000, amp=1.0, phase=0.0):
    """단일 정현파."""
    t = time_axis(duration, sr)
    return amp * np.sin(2 * np.pi * freq * t + phase)


def harmonic_tone(f0, n_harmonics=6, duration=1.0, sr=16000, decay=1.0):
    """배음 구조를 갖는 음. decay 가 클수록 고음이 빨리 줄어 부드러운 음색."""
    t = time_axis(duration, sr)
    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        y += (1.0 / k ** decay) * np.sin(2 * np.pi * f0 * k * t)
    return y / np.max(np.abs(y))


def chirp(f0=200.0, f1=4000.0, duration=1.0, sr=16000):
    """선형 주파수 스윕. 스펙트로그램에서 대각선으로 나타난다."""
    t = time_axis(duration, sr)
    k = (f1 - f0) / duration
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))


def am_tone(carrier=800.0, mod=5.0, duration=1.0, sr=16000):
    """진폭 변조음 — 시간축 엔벨로프 실험용."""
    t = time_axis(duration, sr)
    return (1 + 0.7 * np.sin(2 * np.pi * mod * t)) * np.sin(2 * np.pi * carrier * t)


def window(name, N):
    """분석 윈도우. name: rect / hann / hamming / blackman"""
    n = np.arange(N)
    if name == "rect":
        return np.ones(N)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / N)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / N)
    if name == "blackman":
        return (0.42 - 0.5 * np.cos(2 * np.pi * n / N)
                + 0.08 * np.cos(4 * np.pi * n / N))
    raise ValueError(f"알 수 없는 윈도우: {name}")


def dft_magnitude(x, sr):
    """실수 신호의 단측 크기 스펙트럼.

    Returns
    -------
    freqs : (N/2+1,) Hz,  mag : 같은 길이 크기
    """
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return freqs, np.abs(X) / len(x) * 2


def stft(x, n_fft=512, hop=128, win="hann"):
    """단시간 푸리에 변환.

    Parameters
    ----------
    n_fft : 프레임 길이(샘플). 클수록 주파수 해상도↑ 시간 해상도↓
    hop   : 홉 길이. 작을수록 프레임이 촘촘히 겹친다.

    Returns
    -------
    S : (n_freq, n_frames) 복소 스펙트럼
    """
    w = window(win, n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames < 1:
        raise ValueError("신호가 n_fft 보다 짧다")
    frames = np.stack([x[i * hop: i * hop + n_fft] * w for i in range(n_frames)])
    return np.fft.rfft(frames, axis=1).T


def stft_axes(x, sr, n_fft=512, hop=128):
    """stft() 결과에 대응하는 (시간축 초, 주파수축 Hz)."""
    n_frames = 1 + (len(x) - n_fft) // hop
    times = (np.arange(n_frames) * hop + n_fft / 2) / sr
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return times, freqs


def spectrogram_db(S, floor_db=-80.0, ref=None):
    """복소 스펙트럼 -> dB 스케일 크기 스펙트로그램 (하한 클리핑)."""
    mag = np.abs(S)
    ref = ref if ref is not None else max(mag.max(), 1e-12)
    db = 20 * np.log10(np.maximum(mag, 1e-12) / ref)
    return np.maximum(db, floor_db)


def resample_naive(x, sr_from, sr_to):
    """안티에일리어싱 없이 단순 재추출.

    **일부러 필터를 생략한 구현이다** — 에일리어싱을 눈으로 보여주기 위한 것이므로
    실제 리샘플링에는 쓰지 말 것 (scipy.signal.resample_poly 등을 사용).
    """
    step = sr_from / sr_to
    idx = (np.arange(int(len(x) / step)) * step).astype(int)
    return x[idx]


def quantize(x, bits):
    """균등 양자화. 비트 깊이에 따른 계단 오차를 만든다."""
    levels = 2 ** bits
    x_norm = np.clip(x, -1.0, 1.0)
    q = np.round((x_norm + 1) / 2 * (levels - 1))
    return q / (levels - 1) * 2 - 1


# ══════════════════════════════════════════════════════════════════
# 멜 스케일과 켑스트럼 특성 (8.2 이후)
# ══════════════════════════════════════════════════════════════════

def hz_to_mel(f):
    """Hz -> mel (O'Shaughnessy 근사)."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def mel_to_hz(m):
    """mel -> Hz."""
    return 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)


def mel_filterbank(n_filters=40, n_fft=512, sr=16000, fmin=0.0, fmax=None):
    """삼각 멜 필터뱅크.

    Returns
    -------
    fb      : (n_filters, n_fft//2+1) 필터 행렬
    centers : (n_filters,) 각 필터의 중심 주파수 Hz
    freqs   : (n_fft//2+1,) FFT 빈의 주파수 Hz
    """
    fmax = fmax if fmax else sr / 2
    mpts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz = mel_to_hz(mpts)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        left = (freqs - lo) / max(ce - lo, 1e-9)
        right = (hi - freqs) / max(hi - ce, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
    return fb, hz[1:-1], freqs


def log_mel(x, sr=16000, n_fft=512, hop=160, n_mels=40, fb=None):
    """로그 멜 스펙트로그램 (n_mels, n_frames)."""
    if fb is None:
        fb, _, _ = mel_filterbank(n_mels, n_fft, sr)
    S = np.abs(stft(x, n_fft, hop)) ** 2
    return np.log(fb @ S + 1e-10)


def mfcc(x, n_mfcc=13, sr=16000, n_fft=512, hop=160, n_mels=40, fb=None):
    """로그 멜에 DCT-II 를 걸어 앞 n_mfcc 개를 취한다."""
    from scipy.fft import dct
    lm = log_mel(x, sr, n_fft, hop, n_mels, fb)
    return dct(lm, type=2, axis=0, norm="ortho")[:n_mfcc]


def cepstrum(x, window_name="hann"):
    """실수 켑스트럼 — 로그 크기 스펙트럼의 역변환."""
    seg = x * window(window_name, len(x))
    return np.fft.irfft(np.log(np.abs(np.fft.rfft(seg)) + 1e-12))


def cepstral_envelope(x, lifter=30, window_name="hann"):
    """낮은 케프렌시만 남겨 얻은 로그 스펙트럼 포락.

    lifter 는 **피치 주기(샘플)보다 작아야** 한다 — 넘으면 하모닉 구조가
    포락에 섞여 들어온다 (8.2에서 측정).
    """
    c = cepstrum(x, window_name)
    c2 = c.copy()
    c2[lifter:-lifter] = 0
    return np.fft.rfft(c2).real


def cmvn(F, var_norm=False):
    """켑스트럼 평균(·분산) 정규화. F: (n_coef, n_frames)"""
    out = F - F.mean(axis=1, keepdims=True)
    if var_norm:
        out = out / (F.std(axis=1, keepdims=True) + 1e-9)
    return out


def deltas(F, width=2):
    """델타 특성 — 회귀식 기반 시간 미분. F: (n_coef, n_frames)"""
    d = np.zeros_like(F)
    denom = 2 * sum(t * t for t in range(1, width + 1))
    P = np.pad(F, ((0, 0), (width, width)), mode="edge")
    for t in range(1, width + 1):
        d += t * (P[:, width + t: P.shape[1] - width + t]
                  - P[:, width - t: P.shape[1] - width - t])
    return d / denom
