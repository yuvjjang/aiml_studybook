"""
공통 Plotly 시각화 헬퍼.

모든 챕터의 그래픽은 여기의 팔레트·레이아웃 관례를 따른다.
테마는 plotly_dark, 폰트는 Pretendard, 강조색은 ACCENT.

주의 — 색상 변환:
  hex 색에 알파를 붙일 때 문자열 연결(`col + '1a'`)이나 replace 를 쓰면
  Plotly 가 ValueError 를 낸다. 반드시 `hex_rgba()` 로 rgba(...) 문자열을 만들 것.
"""
import numpy as np
import plotly.graph_objects as go

TEMPLATE = "plotly_dark"
FONT = "Pretendard, Noto Sans KR, sans-serif"

ACCENT = "#7ea8f7"   # 파랑 — 주 대상
WARM = "#f7a07e"     # 주황 — 대비 대상
GREEN = "#7ef7a0"    # 초록 — 결과/정답
YELLOW = "#f7e07e"   # 노랑 — 보조 강조
PINK = "#f77eb8"     # 분홍 — 네 번째 계열

# 계열형 데이터 기본 순서
PALETTE = [ACCENT, WARM, GREEN, YELLOW, PINK, "#b87ef7", "#7ef7e0", "#f7f07e"]

# 연속형(순차) 스케일 — 히트맵 기본값
SEQUENTIAL = "Blues"
# 발산형 — 부호가 의미를 갖는 값(그래디언트, 상관, 어텐션 차이 등)
DIVERGING = "RdBu"


def hex_rgba(hex_color, alpha=0.15):
    """'#rrggbb' + alpha -> 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def color(i):
    """계열 인덱스 -> 팔레트 색 (순환)."""
    return PALETTE[i % len(PALETTE)]


def fig_defaults(fig, title="", height=450, **layout):
    """모든 그래프에 공통 레이아웃을 적용하고 fig 를 돌려준다."""
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, x=0.5, font=dict(size=15)),
        height=height,
        margin=dict(l=55, r=30, t=60, b=55),
        font=dict(family=FONT, size=13),
        legend=dict(
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
        ),
        **layout,
    )
    return fig


def annotate_box(fig, text, x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                 color_=None):
    """그래프 모서리에 수치 요약 상자를 붙인다 (paper 좌표)."""
    fig.add_annotation(
        x=x, y=y, xref="paper", yref="paper",
        xanchor=xanchor, yanchor=yanchor,
        text=text, showarrow=False,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor=color_ or "rgba(255,255,255,0.3)",
        borderwidth=1,
    )
    return fig


def slider(steps, prefix="", suffix="", active=0, pad_t=50):
    """update 방식 슬라이더 하나를 만드는 축약 헬퍼.

    steps : [(label, {trace 속성 dict}), ...]
    """
    return dict(
        active=active,
        currentvalue=dict(prefix=prefix, suffix=suffix, font=dict(size=13)),
        pad=dict(t=pad_t),
        steps=[
            dict(method="update", label=label, args=[args])
            for label, args in steps
        ],
    )


def play_buttons(duration=60, y=1.12, x=0.5):
    """프레임 애니메이션용 재생/정지 버튼."""
    return dict(
        type="buttons", showactive=False, y=y, x=x, xanchor="center",
        buttons=[
            dict(label="▶ 재생", method="animate",
                 args=[None, dict(frame=dict(duration=duration, redraw=True),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸ 정지", method="animate",
                 args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
        ],
    )


def heatmap(matrix, x_title="", y_title="", title="", colorscale=None,
            zmid=None, height=460, text=None, hover=None):
    """행렬 하나를 표준 스타일 히트맵으로."""
    fig = go.Figure(go.Heatmap(
        z=np.asarray(matrix),
        colorscale=colorscale or SEQUENTIAL,
        zmid=zmid,
        text=text,
        texttemplate="%{text}" if text is not None else None,
        hovertemplate=hover or "행 %{y}, 열 %{x}<br>값 %{z:.3f}<extra></extra>",
        colorbar=dict(thickness=12, outlinewidth=0),
    ))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title,
                      yaxis=dict(autorange="reversed"))
    return fig_defaults(fig, title=title, height=height)


def line(x, ys, labels=None, x_title="", y_title="", title="", height=420,
         fill=False, dashes=None):
    """여러 곡선을 한 축에 그리는 표준 선 그래프."""
    fig = go.Figure()
    ys = list(ys)
    for i, y in enumerate(ys):
        c = color(i)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=labels[i] if labels else f"계열 {i + 1}",
            line=dict(color=c, width=2,
                      dash=(dashes[i] if dashes else None)),
            fill="tozeroy" if fill else None,
            fillcolor=hex_rgba(c, 0.10) if fill else None,
        ))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title,
                      showlegend=labels is not None)
    return fig_defaults(fig, title=title, height=height)


def vector_field(X, Y, U, V, scale=0.15, color_=None, name="", step=1):
    """2D 벡터장을 짧은 선분 목록으로 반환 (Scatter 하나로 그리기 위함).

    Returns
    -------
    go.Scatter — None 으로 끊어 이은 선분 집합
    """
    xs, ys = [], []
    for i in range(0, X.shape[0], step):
        for j in range(0, X.shape[1], step):
            x0, y0 = X[i, j], Y[i, j]
            xs += [x0, x0 + scale * U[i, j], None]
            ys += [y0, y0 + scale * V[i, j], None]
    return go.Scatter(x=xs, y=ys, mode="lines", name=name,
                      line=dict(color=color_ or hex_rgba(ACCENT, 0.55), width=1),
                      hoverinfo="skip", showlegend=bool(name))
