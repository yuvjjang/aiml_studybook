"""ai-ml-study 재사용 계산·시각화 모듈."""

# ── Plotly 가 그림마다 끼워 넣는 MathJax 를 끈다 ────────────────────
#
# plotly.io._base_renderers.HtmlRenderer.to_mimebundle() 은
# include_mathjax="cdn" 을 **하드코딩**해서, 그림 하나마다
#   <script src=".../mathjax/3.2.2/es5/tex-svg.min.js"></script>
# 를 본문에 **동기 스크립트**로 넣는다.
#
# Quarto 는 <head> 에서 MathJax v4 를 defer 로 넣는다. defer 는 파싱이
# 끝난 뒤 실행되므로, 본문의 v3 가 **먼저** 실행되어 window.MathJax 를
# 선점하고 v4 는 "이미 로드됨" 으로 중단된다. 그 결과 본문 수식이
# LaTeX 원문 그대로 남는다.
#
# to_mimebundle() 이 호출 시점에 plotly.io.to_html 을 다시 조회하므로,
# 여기서 감싸 두면 이후 모든 그림에 적용된다.
# (이 프로젝트의 plotly 라벨에는 LaTeX 를 쓰지 않으므로 잃는 것이 없다.)
try:
    import plotly.io as _pio

    if not getattr(_pio, "_ai_ml_study_no_mathjax", False):
        _orig_to_html = _pio.to_html

        def _to_html_without_mathjax(*args, **kwargs):
            kwargs["include_mathjax"] = False
            return _orig_to_html(*args, **kwargs)

        _pio.to_html = _to_html_without_mathjax
        _pio._ai_ml_study_no_mathjax = True
except ImportError:          # plotly 없이 src 를 쓰는 경우
    pass
