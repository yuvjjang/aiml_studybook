// 책 본문을 CommonMark(markdown-it = VSCode 미리보기 엔진)로 파싱해
// bold 가 깨진 곳을 찾는다. 강조는 줄바꿈을 넘어가므로 문단 단위로 본다.
const md = require('markdown-it')({ html: true });
const fs = require('fs'), path = require('path');

function walk(d, out = []) {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (e.name.endsWith('.qmd')) out.push(p);
  }
  return out;
}

let bad = 0, checked = 0;
for (const f of walk('chapters')) {
  const lines = fs.readFileSync(f, 'utf8').split('\n');
  let fence = false, yaml = lines[0].trim() === '---', seen = 0;
  let para = [], start = 0;

  const flush = () => {
    if (!para.length) { para = []; return; }
    const text = para.join('\n');
    if (text.includes('**')) {
      checked++;
      const stripped = text.replace(/\$\$?[^$]*?\$\$?/g, 'MATH')
                           .replace(/`[^`]*`/g, 'CODE');
      if (md.render(stripped).includes('**')) {
        bad++;
        console.log(`${f}:${start}\n    ${text.replace(/\n/g, ' ').slice(0, 110)}`);
      }
    }
    para = [];
  };

  lines.forEach((ln, i) => {
    if (yaml) { if (ln.trim() === '---' && ++seen === 2) yaml = false; return; }
    if (ln.trimStart().startsWith('```')) { flush(); fence = !fence; return; }
    if (fence) return;
    if (ln.trim() === '') { flush(); return; }
    if (!para.length) start = i + 1;
    para.push(ln);
  });
  flush();
}
console.log(`\n** 포함 문단 ${checked}개 검사 → 깨진 문단 ${bad}개`);
