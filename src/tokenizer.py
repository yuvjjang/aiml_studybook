"""토크나이저 — BPE / WordPiece / byte-level 을 직접 구현한다.

Part 7 (언어) 장들이 공유한다. 렌더 타임 의존성은 표준 라이브러리뿐이다.
"""

import collections

__all__ = ["KO_CORPUS", "EN_CORPUS", "word_freqs", "train_bpe", "apply_bpe",
           "train_wordpiece", "bpe_vocab", "encode_corpus",
           "byte_word_freqs", "BYTE_ALPHABET"]


KO_CORPUS = """
인공지능 모델은 텍스트를 토큰으로 나누어 처리한다.
토크나이저는 모델이 보는 최소 단위를 정하는 부품이다.
학습이 끝난 뒤에는 토크나이저를 바꿀 수 없다.
어휘 크기를 키우면 시퀀스가 짧아지지만 임베딩 행렬이 커진다.
한국어는 교착어라서 어절 단위로 나누면 어휘가 폭발한다.
서브워드 방식은 자주 나오는 조각을 하나의 토큰으로 묶는다.
자주 나오는 단어는 통째로 남고 드문 단어는 조각으로 쪼개진다.
모델은 토큰 단위로 확률을 계산하고 토큰 단위로 생성한다.
토큰이 많아지면 계산 비용과 문맥 창 사용량이 함께 늘어난다.
바이트 단위로 내려가면 어떤 문자열도 표현할 수 있다.
학습 데이터에 없는 단어도 조각으로 나누어 처리한다.
토크나이저의 품질은 모델의 성능에 직접 영향을 준다.
"""

EN_CORPUS = """
Language models process text by splitting it into tokens.
The tokenizer decides the smallest unit the model can see.
Once training is finished the tokenizer cannot be changed.
A larger vocabulary shortens sequences but grows the embedding matrix.
Subword methods merge frequent character pairs into single tokens.
Frequent words stay whole while rare words are split into pieces.
The model computes probabilities over tokens and generates tokens.
More tokens mean more computation and more context window usage.
Working at the byte level lets any string be represented.
Words unseen during training are handled as smaller pieces.
The quality of the tokenizer directly affects model performance.
"""

BYTE_ALPHABET = [bytes([i]) for i in range(256)]


def word_freqs(text):
    """공백으로 나눈 단어(어절)의 빈도. 사전 토큰화 단계에 해당한다."""
    c = collections.Counter()
    for line in text.strip().splitlines():
        for w in line.split():
            c[w] += 1
    return c


def byte_word_freqs(text):
    """단어를 UTF-8 바이트 튜플로 표현한 빈도 — byte-level BPE 용."""
    c = collections.Counter()
    for w, f in word_freqs(text).items():
        c[tuple(bytes([b]) for b in w.encode("utf-8"))] += f
    return c


def _apply_merge(splits, pair):
    a, b = pair
    new = {}
    for w, s in splits.items():
        out, i = [], 0
        while i < len(s):
            if i < len(s)-1 and s[i] == a and s[i+1] == b:
                out.append(a + b); i += 2
            else:
                out.append(s[i]); i += 1
        new[w] = tuple(out)
    return new


def train_bpe(freqs, n_merges, pre_split=True):
    """BPE — 가장 **빈번한** 인접 쌍을 반복 병합한다.

    freqs      : {단어: 빈도}. 단어가 튜플이면 이미 쪼개진 것으로 본다.
    반환        : ([((a, b), 빈도), ...], {단어: 토큰 튜플})
    """
    splits = {w: (tuple(w) if pre_split and isinstance(w, str) else tuple(w))
              for w in freqs}
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, f in freqs.items():
            s = splits[w]
            for i in range(len(s)-1):
                pairs[(s[i], s[i+1])] += f
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append((best, pairs[best]))
        splits = _apply_merge(splits, best)
    return merges, splits


def train_wordpiece(freqs, n_merges, min_freq=1):
    """WordPiece — 빈도가 아니라 **우도 증가량**으로 병합을 고른다.

        score(a, b) = freq(ab) / (freq(a) · freq(b))

    이미 흔한 조각끼리의 결합은 점수가 낮아진다.
    min_freq 로 바닥을 두지 않으면 희귀 쌍이 과도하게 선택된다.
    """
    splits = {w: tuple(w) for w in freqs}
    merges = []
    for _ in range(n_merges):
        pair_f, tok_f = collections.Counter(), collections.Counter()
        for w, f in freqs.items():
            s = splits[w]
            for i, t in enumerate(s):
                tok_f[t] += f
                if i < len(s)-1:
                    pair_f[(s[i], s[i+1])] += f
        cand = {p: c for p, c in pair_f.items() if c >= min_freq}
        if not cand:
            break
        best = max(cand, key=lambda p: (cand[p]/(tok_f[p[0]]*tok_f[p[1]]), p))
        merges.append((best, cand[best],
                       cand[best]/(tok_f[best[0]]*tok_f[best[1]])))
        splits = _apply_merge(splits, best)
    return merges, splits


def apply_bpe(word, merges):
    """학습된 병합 규칙을 **순서대로** 새 단어에 적용한다."""
    s = tuple(word)
    rank = {m[0]: i for i, m in enumerate(merges)}
    while len(s) > 1:
        cand = [(rank[(s[i], s[i+1])], i) for i in range(len(s)-1)
                if (s[i], s[i+1]) in rank]
        if not cand:
            break
        _, i = min(cand)
        s = s[:i] + (s[i] + s[i+1],) + s[i+2:]
    return s


def bpe_vocab(freqs, merges):
    """기본 문자 + 병합으로 생긴 토큰 전체."""
    base = {c for w in freqs for c in w}
    return base | {a + b for (a, b), *_ in merges}


def encode_corpus(text, merges):
    """텍스트 전체를 토큰 리스트로."""
    out = []
    for line in text.strip().splitlines():
        for w in line.split():
            out.extend(apply_bpe(w, merges))
    return out
