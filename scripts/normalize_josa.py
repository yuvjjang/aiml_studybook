"""로마자 뒤 조사의 띄어쓰기를 '붙임'으로 통일한다.

건드리지 않는 것
  - 코드 청크(``` 펜스 안), 인라인 코드(`...`), 링크 URL, YAML 머리말
  - 수식 뒤(`$x$ 를`) — 수식은 띄어 쓰는 것이 이 책의 관례다
  - 조사처럼 보이지만 낱말인 경우(`Python 도구`) — 조사 뒤가 한글이면 건너뛴다
"""
import io
import re
import sys
import glob

# 긴 것부터: 짧은 조사가 긴 조사의 앞부분을 먼저 먹지 않게
JOSA = ("으로써|으로서|으로|이라는|이라고|이라|이고|이다|입니다|에서는|에서도|에서|"
        "에게서|에게|부터|까지|처럼|보다|마다|조차|밖에|만큼|과의|와의|의|은|는|"
        "이|가|을|를|과|와|로|도|만|에")

# 로마자(또는 로마자+숫자) 뒤 공백 하나 + 조사, 그리고 그 조사 뒤가 한글이 아닐 것
PAT = re.compile(rf"(?<=[A-Za-z0-9])[ ]({JOSA})(?![가-힣])")

INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"\]\([^)\s]*\)")
MATH = re.compile(r"\$\$?[^$\n]*\$\$?")   # $x$ 와 한 줄짜리 $$...$$


def convert(text):
    """마스킹으로 보호한 뒤 치환하고 되돌린다."""
    holes = []

    def stash(m):
        holes.append(m.group(0))
        return f"\x00{len(holes)-1}\x00"

    t = INLINE_CODE.sub(stash, text)
    t = MATH.sub(stash, t)
    t = LINK.sub(stash, t)
    t, n = PAT.subn(r"\1", t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: holes[int(m.group(1))], t)
    return t, n


def process(path, dry):
    s = io.open(path, encoding="utf-8").read()
    lines = s.split("\n")

    # YAML 머리말 범위
    end_yaml = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_yaml = i
                break

    out, in_fence, total = [], False, 0
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(ln)
            continue
        if in_fence or i <= end_yaml:
            out.append(ln)
            continue
        new, n = convert(ln)
        total += n
        out.append(new)

    if total and not dry:
        io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    return total


def main():
    dry = "--dry" in sys.argv
    files = sorted(glob.glob("chapters/**/*.qmd", recursive=True))
    grand = 0
    for f in files:
        n = process(f, dry)
        if n:
            grand += n
            if dry:
                print(f"{n:>4}  {f[9:]}")
    print(f"\n{'[dry-run] ' if dry else ''}총 {grand}곳")


if __name__ == "__main__":
    main()
