"""
어텐션 메커니즘의 NumPy 참조 구현.

Part 5(트랜스포머) 챕터들이 이 모듈을 호출해 실제 숫자를 계산하고 그린다.
프레임워크 없이 순수 NumPy 로만 작성되어 있으므로, 각 줄이 수식의 어느
항에 대응하는지 그대로 따라갈 수 있다.

표기
----
  n : 시퀀스 길이,  d : 모델 차원,  h : 헤드 수,  d_k = d / h
"""
import numpy as np


def softmax(z, axis=-1):
    """수치적으로 안정한 softmax (최댓값을 빼고 지수화).

    naive 구현 exp(z)/sum(exp(z)) 은 z 가 크면 오버플로한다.
    Part 0.5 의 log-sum-exp 트릭과 같은 이유.
    """
    z = np.asarray(z, dtype=float)
    z_max = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True):
    """Attention(Q,K,V) = softmax(QKᵀ / √d_k) V

    Parameters
    ----------
    Q : (n_q, d_k)   질의
    K : (n_k, d_k)   키
    V : (n_k, d_v)   값
    mask : (n_q, n_k) bool 또는 None
        True 인 위치를 **가린다**(점수를 −inf 로). 인과 마스크는
        `causal_mask(n)` 로 만든다.
    scale : bool
        False 면 √d_k 로 나누지 않는다 (스케일링 효과 비교 실험용).

    Returns
    -------
    out : (n_q, d_v)   가중합 결과
    attn : (n_q, n_k)  어텐션 가중치 (행 합 = 1)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    attn = softmax(scores, axis=-1)
    return attn @ V, attn


def causal_mask(n):
    """(n, n) 인과 마스크. True = 가림 (미래 토큰)."""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def padding_mask(lengths, n):
    """가변 길이 배치용 패딩 마스크. True = 가림.

    lengths : 각 시퀀스의 실제 길이 목록
    """
    idx = np.arange(n)
    return np.array([idx >= L for L in lengths])


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    """멀티헤드 셀프 어텐션 한 층.

    Parameters
    ----------
    X  : (n, d)      입력 시퀀스
    Wq, Wk, Wv : (d, d)   질의·키·값 사영 (헤드 전체를 한 번에)
    Wo : (d, d)      출력 사영
    n_heads : int    헤드 수 h. d 를 나눠떨어뜨려야 한다.

    Returns
    -------
    out : (n, d)
    attn : (h, n, n)  헤드별 어텐션 가중치
    """
    n, d = X.shape
    assert d % n_heads == 0, "d_model 은 헤드 수로 나눠떨어져야 한다"
    d_k = d // n_heads

    # 한 번에 사영한 뒤 헤드로 쪼갠다 — 파라미터 수는 단일 헤드와 동일하다.
    Q = (X @ Wq).reshape(n, n_heads, d_k).transpose(1, 0, 2)   # (h, n, d_k)
    K = (X @ Wk).reshape(n, n_heads, d_k).transpose(1, 0, 2)
    V = (X @ Wv).reshape(n, n_heads, d_k).transpose(1, 0, 2)

    heads, attns = [], []
    for i in range(n_heads):
        o, a = scaled_dot_product_attention(Q[i], K[i], V[i], mask=mask)
        heads.append(o)
        attns.append(a)

    concat = np.concatenate(heads, axis=-1)        # (n, d)
    return concat @ Wo, np.stack(attns)


def sinusoidal_encoding(n, d, base=10000.0):
    """원 논문의 사인파 절대 위치 인코딩. (n, d)"""
    pos = np.arange(n)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / np.power(base, i / d)

    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, : d // 2])
    return pe


def rope_angles(n, d_k, base=10000.0):
    """RoPE 회전각 θ_{pos,i}. (n, d_k/2)"""
    pos = np.arange(n)[:, None]
    i = np.arange(d_k // 2)[None, :]
    return pos / np.power(base, 2 * i / d_k)


def apply_rope(X, base=10000.0):
    """RoPE 적용: 인접 차원 쌍을 위치에 따라 복소평면에서 회전시킨다.

    X : (n, d_k), d_k 는 짝수
    회전 후 두 벡터의 내적은 절대 위치가 아니라 **상대 위치**에만 의존한다.
    """
    n, d_k = X.shape
    assert d_k % 2 == 0, "RoPE 는 짝수 차원을 요구한다"
    theta = rope_angles(n, d_k, base)              # (n, d_k/2)
    cos, sin = np.cos(theta), np.sin(theta)

    x_even, x_odd = X[:, 0::2], X[:, 1::2]
    out = np.empty_like(X)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out


def alibi_bias(n, slope):
    """ALiBi 선형 거리 바이어스. (n, n)

    점수에 더해지며, 멀리 있는 키일수록 큰 음수 → 감쇠 효과.
    """
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return -slope * np.abs(i - j)


def attention_flops(n, d, n_heads=1):
    """어텐션 한 층의 대략적 연산량과 어텐션 행렬 메모리.

    Returns
    -------
    dict : qkv_proj / scores / weighted_sum / total FLOPs, attn_bytes(float32)
    """
    qkv = 3 * n * d * d * 2          # Q,K,V 사영
    scores = n * n * d * 2           # QKᵀ
    wsum = n * n * d * 2             # attn @ V
    out = n * d * d * 2              # W_O
    return dict(
        qkv_proj=qkv,
        scores=scores,
        weighted_sum=wsum,
        out_proj=out,
        total=qkv + scores + wsum + out,
        attn_bytes=n * n * n_heads * 4,
    )
