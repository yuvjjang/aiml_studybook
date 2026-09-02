"""이미 만들어진 _freeze / _site 에서 Plotly 가 주입한 MathJax 를 제거한다.

왜 필요한가
-----------
Plotly 는 그림마다 MathJax v3 (tex-svg) 를 **동기 스크립트**로 본문에 넣는다.
Quarto 가 <head> 에 defer 로 넣는 MathJax v4 보다 먼저 실행되어 v4 를
중단시키고, 그 결과 본문 수식이 LaTeX 원문 그대로 남는다.

앞으로 실행되는 셀은 src/__init__.py 의 패치로 이 스크립트를 넣지 않는다.
다만 **이미 캐시된 실행 결과**(_freeze)와 렌더된 페이지(_site)에는 남아 있어서,
다시 실행하지 않고 지우려면 이 스크립트가 필요하다.

사용:  python scripts/strip_plotly_mathjax.py
"""
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAT = re.compile(
    r'<script src="https://cdnjs\.cloudflare\.com/ajax/libs/mathjax/'
    r'[^"]*"></script>'
    r'(<script>if \(window\.MathJax[^<]*</script>)?'
)


def scrub(s):
    """문자열에서 주입된 스크립트를 지우고 (새 문자열, 제거 수) 를 돌려준다."""
    if not isinstance(s, str) or "cdnjs.cloudflare.com/ajax/libs/mathjax" not in s:
        return s, 0
    new, n = PAT.subn("", s)
    return new, n


def walk(obj):
    """중첩 구조 안의 모든 문자열을 훑는다."""
    total = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            nv, n = walk(v)
            obj[k] = nv
            total += n
        return obj, total
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            nv, n = walk(v)
            obj[i] = nv
            total += n
        return obj, total
    if isinstance(obj, str):
        return scrub(obj)
    return obj, 0


def main():
    n_files = n_hits = 0

    freeze = os.path.join(ROOT, "_freeze")
    for base, _, files in os.walk(freeze):
        for f in files:
            if not f.endswith(".json"):
                continue
            p = os.path.join(base, f)
            raw = io.open(p, encoding="utf-8").read()
            if "cdnjs.cloudflare.com/ajax/libs/mathjax" not in raw:
                continue
            data = json.loads(raw)
            data, n = walk(data)
            if n:
                io.open(p, "w", encoding="utf-8").write(
                    json.dumps(data, ensure_ascii=False))
                n_files += 1
                n_hits += n

    site = os.path.join(ROOT, "_site")
    for base, _, files in os.walk(site):
        for f in files:
            if not f.endswith(".html"):
                continue
            p = os.path.join(base, f)
            s = io.open(p, encoding="utf-8").read()
            new, n = scrub(s)
            if n:
                io.open(p, "w", encoding="utf-8").write(new)
                n_files += 1
                n_hits += n

    print(f"파일 {n_files}개에서 주입된 MathJax 스크립트 {n_hits}개 제거")
    return 0


if __name__ == "__main__":
    sys.exit(main())
