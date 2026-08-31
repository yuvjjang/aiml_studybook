# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 콘텐츠 수정 규칙 (ai-ml-study 전용)

- 이 프로젝트는 Quarto book이다. 콘텐츠 수정 대상은 `chapters/**/*.qmd`.
- "노트로 추가해줘" → 해당 섹션 뒤에 `::: {.callout-note}` 블록을 append. 기존 본문은 건드리지 않는다.
- "본문에 병합해줘" → 기존 문체를 유지하며 해당 섹션을 재작성한다.
- 그래프가 필요하면 반드시 실행 가능한 python 코드 청크로 생성하고, `src/viz.py`, `src/datasets.py` 등 기존 모듈을 최대한 재사용한다 (`plotly_dark` 템플릿, Pretendard 폰트, `viz.ACCENT` 팔레트 — `src/viz.py`의 관례 참고).
- 콘텐츠 수정 후에는 git commit 하나를 생성한다.

### 이 프로젝트의 추가 제약

- **렌더 타임 의존성은 numpy / scipy / plotly 로 제한한다.** 새 의존성을 `requirements.txt` 에 추가하지 말 것.
  - 인터랙티브 그래픽은 NumPy 로 직접 계산한다.
  - PyTorch·Hugging Face 코드는 **실행하지 않는** ` ```python ` 블록으로만 제시한다 (` ```{python} ` 이 아니다).
- **합성 데이터는 항상 시드를 고정한다.** `src/datasets.py` 의 생성기를 쓰거나 같은 관례를 따를 것. 빌드마다 그림이 바뀌면 안 된다.
- **커리큘럼(목차·챕터 구성)은 `scripts/curriculum.py` 가 단일 진실 원천이다.**
  - `_quarto.yml`, `PLAN.md`, 챕터 스텁을 직접 손으로 고치지 말 것 — `curriculum.py` 를 고치고 `python scripts/gen_scaffold.py` 를 실행한다.
  - 단, **본문이 채워진 `.qmd` 는 예외**다. 집필을 시작할 때 `<!-- STUB: ... -->` 주석 줄을 지우면 생성기가 그 파일을 건드리지 않는다.
- **hex 색에 알파를 붙일 때는 반드시 `viz.hex_rgba()` 를 쓴다.** 문자열 연결(`col + '1a'`)이나 replace 는 Plotly 가 거부한다.
- 챕터 수정 후에는 그 챕터만 렌더링해 에러를 확인한다:
  - Windows: `.\scripts\render-check.ps1 <수정한 챕터>`
  - macOS/Linux: `./scripts/render-check.sh <수정한 챕터>`