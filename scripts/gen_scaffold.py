"""
스캐폴드 생성기 — `scripts/curriculum.py` 의 PARTS 로부터 세 산출물을 만든다.

  1. `_quarto.yml`            : 책 설정 + 전체 목차
  2. `chapters/**/*.qmd`      : 챕터 스텁 (아직 본문이 없는 것만)
  3. `PLAN.md`                : 챕터별 이론·그래픽 상세 플래닝 문서

사용법:
    python scripts/gen_scaffold.py            # 생성/갱신
    python scripts/gen_scaffold.py --dry-run  # 무엇이 바뀌는지만 출력

**본문을 채운 .qmd 는 덮어쓰지 않는다.** 생성된 스텁에는 아래 STUB_MARK 주석이
들어 있고, 생성기는 그 주석이 있는 파일만 갱신한다. 본문 작업을 시작할 때
그 주석 줄을 지우면 이후 재생성에서 보호된다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from curriculum import PARTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "ai-ml-study"
REPO_URL = "https://github.com/yuvjjang/aiml_studybook"
STUB_MARK = "<!-- STUB: scripts/gen_scaffold.py 가 생성한 뼈대입니다. 본문을 채우면 이 줄을 지우세요. -->"


# ──────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ──────────────────────────────────────────────────────────────────────

def flat_chapters():
    """(part, chapter, 전역 순번) 튜플을 순서대로 산출."""
    i = 0
    for part in PARTS:
        for ch in part["chapters"]:
            yield part, ch, i
            i += 1


ALL = list(flat_chapters())


def qmd_path(part, ch):
    return ROOT / "chapters" / part["dir"] / f"{ch['file']}.qmd"


def rel_link(src_part, dst_part, dst_ch):
    """src 챕터에서 dst 챕터로 가는 상대 경로."""
    if src_part["dir"] == dst_part["dir"]:
        return f"{dst_ch['file']}.qmd"
    return f"../{dst_part['dir']}/{dst_ch['file']}.qmd"


def graphic_id(ch, idx):
    return f"G{ch['no']}.{idx + 1}"


# ──────────────────────────────────────────────────────────────────────
# 1. _quarto.yml
# ──────────────────────────────────────────────────────────────────────

def build_quarto_yml():
    lines = [
        "project:",
        "  type: book",
        "  output-dir: _site",
        "",
        "book:",
        '  title: "AI · 머신러닝: 기초부터 최신까지"',
        '  subtitle: "데이터 사이언스 · 머신러닝 · 트랜스포머 · 언어 · 오디오 · 비디오"',
        '  author: "AI/ML Study Project"',
        '  date: "2026"',
        "  language: ko",
        f"  repo-url: {REPO_URL}",
        "  repo-actions: [edit, issue]",
        "  search: true",
        "  sidebar:",
        "    style: docked",
        "    background: dark",
        "  chapters:",
        "    - index.qmd",
    ]
    for part in PARTS:
        lines.append(f'    - part: "{part["title"]}"')
        lines.append("      chapters:")
        for ch in part["chapters"]:
            lines.append(f'        - chapters/{part["dir"]}/{ch["file"]}.qmd')

    lines += [
        "",
        "format:",
        "  html:",
        "    theme:",
        "      dark: [slate, custom.scss]",
        "      light: [flatly, custom.scss]",
        "    css: styles.css",
        "    toc: true",
        "    toc-depth: 3",
        '    toc-title: "이 페이지"',
        "    number-sections: true",
        "    code-fold: true",
        '    code-summary: "코드 보기"',
        "    code-tools: true",
        "    highlight-style: dracula",
        "    fig-align: center",
        "    html-math-method: mathjax",
        "    smooth-scroll: true",
        "    anchor-sections: true",
        "",
        f"jupyter: {KERNEL}",
        "",
        "execute:",
        "  echo: true",
        "  warning: false",
        "  freeze: auto",
        "",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# 2. 챕터 스텁
# ──────────────────────────────────────────────────────────────────────

def build_stub(part, ch, idx):
    prev_ = ALL[idx - 1] if idx > 0 else None
    next_ = ALL[idx + 1] if idx + 1 < len(ALL) else None

    out = [
        "---",
        f'title: "{ch["no"]} {ch["title"]}"',
        f'subtitle: "{ch["sub"]}"',
        f"jupyter: {KERNEL}",
        "---",
        "",
        STUB_MARK,
        "",
        '::: {.callout-important appearance="simple"}',
        "## 아직 집필 전인 장입니다",
        "",
        "이 페이지는 **계획만 있는 상태**입니다. 아래는 다룰 이론 항목과 준비 중인 그래픽 목록이며,",
        "본문과 실행 가능한 시각화는 아직 없습니다.",
        "",
        "다른 장에서 이 페이지로 연결된 링크를 따라오셨다면, 그 개념은 여기서 다룰 예정이라는 뜻입니다.",
        "전체 집필 현황은 [PLAN.md](https://github.com/yuvjjang/aiml_studybook/blob/main/PLAN.md) 를 참고하세요.",
        ":::",
        "",
        ch["why"],
        "",
        "---",
        "",
        "## 이 장에서 다루는 것",
        "",
    ]
    for t in ch["theory"]:
        out.append(f"- {t}")

    out += ["", "---", ""]

    for i, (gtitle, interaction, desc) in enumerate(ch["graphics"]):
        gid = graphic_id(ch, i)
        out += [
            f"## {gid} — {gtitle}",
            "",
            desc,
            "",
            "::: {.callout-note appearance=\"minimal\"}",
            f"**계획된 그래픽** · 인터랙션: {interaction}",
            ":::",
            "",
            "---",
            "",
        ]

    out += ["## 핵심 용어", "", "| 용어 | 비고 |", "|------|------|"]
    for k in ch["keys"]:
        out.append(f"| {k} |  |")
    out.append("")

    if next_:
        np_, nc, _ = next_
        out += [
            "---",
            "",
            f'**다음**: [{nc["no"]} {nc["title"]}]({rel_link(part, np_, nc)}) — {nc["sub"]}',
            "",
        ]
    if prev_:
        pp, pc, _ = prev_
        out.append(f'**이전**: [{pc["no"]} {pc["title"]}]({rel_link(part, pp, pc)})')
        out.append("")

    return "\n".join(out)


def write_stubs(dry_run):
    created, updated, skipped = [], [], []
    for part, ch, idx in ALL:
        path = qmd_path(part, ch)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = build_stub(part, ch, idx)

        if not path.exists():
            created.append(path)
        elif STUB_MARK in path.read_text(encoding="utf-8"):
            if path.read_text(encoding="utf-8") == content:
                continue
            updated.append(path)
        else:
            skipped.append(path)
            continue

        if not dry_run:
            path.write_text(content, encoding="utf-8")
    return created, updated, skipped


# ──────────────────────────────────────────────────────────────────────
# 3. PLAN.md
# ──────────────────────────────────────────────────────────────────────

def is_written(part, ch):
    """본문이 채워진 챕터인가 (STUB 주석이 지워졌는가)."""
    path = qmd_path(part, ch)
    return path.exists() and STUB_MARK not in path.read_text(encoding="utf-8")


def build_plan():
    total = len(ALL)
    n_graphics = sum(len(ch["graphics"]) for _, ch, _ in ALL)
    done = [(p, c) for p, c, _ in ALL if is_written(p, c)]

    out = [
        "# AI/ML Study — 상세 플래닝 문서",
        "",
        "> 이 파일은 `scripts/curriculum.py` 에서 자동 생성됩니다.",
        "> 커리큘럼을 고칠 때는 `curriculum.py` 를 수정하고 `python scripts/gen_scaffold.py` 를 다시 실행하세요.",
        "",
        "## 개요",
        "",
        "데이터 사이언스부터 최신 AI 연구까지, 각 개념마다 **인터랙티브 시각화**를 직접 구현하며",
        "기초에서 고급까지 이어지는 한 권의 책을 만드는 프로젝트.",
        "",
        f"- **파트** {len(PARTS)}개 · **챕터** {total}개 · **계획된 그래픽** {n_graphics}개",
        "- **출력 형식**: Quarto → HTML 정적 사이트 (책)",
        "- **계산·그래픽**: NumPy / SciPy / Plotly (렌더 타임 의존성은 이 셋으로 제한)",
        "- **딥러닝 프레임워크 코드**: 실행하지 않는 예시 블록으로 제시 (빌드 재현성 우선)",
        "",
        f"- **집필 완료** {len(done)} / {total} 챕터",
        "",
        "## 전체 구성",
        "",
        "| 파트 | 주제 | 챕터 | 완료 | 이 파트를 마치면 |",
        "|------|------|------|------|------------------|",
    ]
    for part in PARTS:
        n_done = sum(1 for ch in part["chapters"] if is_written(part, ch))
        out.append(f'| `{part["dir"]}` | {part["title"]} | {len(part["chapters"])} '
                   f'| {n_done} | {part["goal"]} |')

    out += [
        "",
        "### 집필 완료된 장",
        "",
    ]
    if done:
        for p, c in done:
            out.append(f'- ✅ **{c["no"]} {c["title"]}** — `chapters/{p["dir"]}/{c["file"]}.qmd`')
    else:
        out.append("- (아직 없음)")

    out += ["", "---", ""]

    for part in PARTS:
        out += [
            f'## {part["title"]}',
            "",
            f'> {part["goal"]}',
            "",
        ]
        for ch in part["chapters"]:
            badge = " ✅" if is_written(part, ch) else ""
            out += [
                "---",
                "",
                f'### {ch["no"]} {ch["title"]}{badge}',
                "",
                f'`chapters/{part["dir"]}/{ch["file"]}.qmd` — *{ch["sub"]}*',
                "",
                f'{ch["why"]}',
                "",
                "**이론 내용**",
                "",
            ]
            for t in ch["theory"]:
                out.append(f"- {t}")
            out += [
                "",
                "**그래픽 목록**",
                "",
                "| # | 제목 | 인터랙션 | 설명 |",
                "|---|------|----------|------|",
            ]
            for i, (gtitle, interaction, desc) in enumerate(ch["graphics"]):
                out.append(f"| {graphic_id(ch, i)} | {gtitle} | {interaction} | {desc} |")
            out += [
                "",
                f'**핵심 용어**: {", ".join(ch["keys"])}',
                "",
            ]

    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────

def main():
    dry_run = "--dry-run" in sys.argv

    yml = build_quarto_yml()
    plan = build_plan()
    if not dry_run:
        (ROOT / "_quarto.yml").write_text(yml, encoding="utf-8")
        (ROOT / "PLAN.md").write_text(plan, encoding="utf-8")

    created, updated, skipped = write_stubs(dry_run)

    print(f"{'[dry-run] ' if dry_run else ''}파트 {len(PARTS)} · 챕터 {len(ALL)} · "
          f"그래픽 {sum(len(c['graphics']) for _, c, _ in ALL)}")
    print(f"  _quarto.yml  {len(yml.splitlines())} 줄")
    print(f"  PLAN.md      {len(plan.splitlines())} 줄")
    print(f"  스텁 생성 {len(created)} · 갱신 {len(updated)} · 본문 보존 {len(skipped)}")
    for p in skipped:
        print(f"    보존: {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
