"""번역투 문장 탐지기 — chapters/**/*.qmd 의 본문에서 다듬을 후보를 찾는다.

코드 청크·디스플레이 수식·표·YAML 은 검사 대상이 아니다.
인라인 수식은 `@` 한 글자로 치환해 문장 구조만 보이게 한다.

사용법:
    python scripts/prose_audit.py              # 파트별 요약
    python scripts/prose_audit.py --chapter    # 챕터별 표
    python scripts/prose_audit.py <경로>       # 그 챕터의 실제 문장 출력
"""
import io
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"

# (이름, 정규식, 설명) — 실제로 고칠 값이 있는 것만 넣는다.
PATTERNS = [
    ("lies-in",
     re.compile(r"이유[는가][^.\n]{0,40}(?:에 있다|에 있고|에 있으며)"),
     "영어 'lies in' 직역 — '이유는 ~에 있다'"),
    ("것이-이다",
     re.compile(r"(?:것|점|일)이\s[^.\n]{0,30}(?:이다|다)\."),
     "'~하는 것이 ~이다' 정의문"),
    ("것은-것이다",
     re.compile(r"것은[^.\n]{0,40}것이다"),
     "'~하는 것은 ~하는 것이다'"),
    ("사실이-만든다",
     re.compile(r"(?:사실|점|것)이\s[^.\n]{0,40}(?:만든다|만들었다|가능하게 한다)"),
     "영어 'makes X Y' 직역"),
    ("에-대한-답",
     re.compile(r"에 대한 (?:답|해답|대답|해결책)"),
     "'~에 대한 답' 번역투"),
    ("지시어",
     re.compile(r"(?:^|[\s(“\"])(?:이것|그것)(?:이|은|을|의|에)"),
     "지시어를 영어 it/this 처럼 사용"),
    ("문두-접속",
     re.compile(r"(?m)^(?:그런데|그래서|그리고|그러므로|하지만|따라서)\s"),
     "문단 첫머리 접속부사"),
    ("일-일",
     re.compile(r"일은[^.\n]{0,40}일이다"),
     "같은 명사 반복 ('~하는 일은 ~하는 일이다')"),
    ("긴-관형절",
     re.compile(r"[가-힣\w)][은는이가]\s\*\*[^*\n]{25,}\*\*를\s(?:재는|뜻하는|의미하는|나타내는)"),
     "관형절 과적재 — 결론이 뒤로 밀림"),
]


def prose_of(path):
    """qmd 에서 검사 대상 본문만 남긴다."""
    s = io.open(path, encoding="utf-8").read()
    s = re.sub(r"^---\n.*?\n---\n", "", s, flags=re.S)        # YAML 머리말

    # 코드 청크는 줄 단위로 걷어낸다. 본문에 인라인으로 ```python 을 언급하는
    # 문장이 있어서, 정규식으로 ```...``` 를 짝지으면 펜스가 어긋난다.
    out, in_fence = [], False
    for line in s.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    s = "\n".join(out)

    s = re.sub(r"\$\$.*?\$\$", "", s, flags=re.S)             # 디스플레이 수식
    s = re.sub(r"(?m)^\s*\|.*$", "", s)                       # 표
    s = re.sub(r"\$[^$\n]*\$", "@", s)                        # 인라인 수식
    return s


def scan(path):
    s = prose_of(path)
    return {name: len(rx.findall(s)) for name, rx, _ in PATTERNS}, len(s)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and "--dump" in sys.argv:          # 본문만 덤프 (사람이 통독할 용도)
        for a in args:
            p = Path(a)
            if not p.is_absolute():
                p = ROOT / p
            for f in ([p] if p.suffix == ".qmd" else sorted(p.rglob("*.qmd"))):
                s = prose_of(f)
                s = re.sub(r"\n{3,}", "\n\n", s).strip()
                print(f"\n{'=' * 70}\n### {f.relative_to(ROOT).as_posix()}\n{'=' * 70}")
                print(s)
        return

    if args:                                   # 단일 챕터의 실제 문장 보기
        p = Path(args[0])
        if not p.is_absolute():
            p = ROOT / p
        s = prose_of(p)
        # 문장 단위로 보여준다. 한 문장이 여러 패턴에 걸려도 한 번만 출력한다.
        seen = {}
        for name, rx, _ in PATTERNS:
            for m in rx.finditer(s):
                a = s.rfind("\n\n", 0, m.start()) + 2
                b = s.find("\n\n", m.end())
                if b == -1:
                    b = len(s)
                block = s[a:b].strip()
                seen.setdefault(block, set()).add(name)
        for block, names in seen.items():
            if block.startswith(("#", ":::", "|", "-", "*", ">")) and "\n" not in block:
                continue
            print(f"\n[{','.join(sorted(names))}]")
            for line in block.split("\n"):
                print("   " + line)
        print(f"\n총 {len(seen)}개 문단")
        return

    per_chapter = []
    for f in sorted(CHAPTERS.rglob("*.qmd")):
        counts, size = scan(f)
        per_chapter.append((f.relative_to(ROOT).as_posix(), counts, size))

    names = [n for n, _, _ in PATTERNS]
    by_chapter = "--chapter" in sys.argv

    if by_chapter:
        rows = sorted(per_chapter, key=lambda r: -sum(r[1].values()))
        print(f"{'챕터':<46}{'합계':>5}  " + "".join(f"{n[:9]:>11}" for n in names))
        for path, c, _ in rows[:40]:
            print(f"{path[9:]:<46}{sum(c.values()):>5}  "
                  + "".join(f"{c[n]:>11}" for n in names))
        return

    parts = {}
    for path, c, size in per_chapter:
        part = path.split("/")[1]
        d = parts.setdefault(part, [Counter(), 0, 0])
        d[0].update(c)
        d[1] += size
        d[2] += 1

    print(f"{'파트':<22}{'챕터':>4}{'본문(자)':>10}{'후보':>7}   패턴별")
    total = Counter()
    for part in sorted(parts):
        c, size, n = parts[part]
        total.update(c)
        top = ", ".join(f"{k} {v}" for k, v in c.most_common(3) if v)
        print(f"{part:<22}{n:>4}{size:>10,}{sum(c.values()):>7}   {top}")
    print(f"\n{'합계':<22}{sum(p[2] for p in parts.values()):>4}"
          f"{sum(p[1] for p in parts.values()):>10,}{sum(total.values()):>7}")
    print("\n패턴별 합계")
    for name, _, desc in PATTERNS:
        print(f"   {total[name]:>5}  {name:<14} {desc}")


if __name__ == "__main__":
    main()
