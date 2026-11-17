// Self-test injected by tests (NOT shipped to annotators). Runs only with ?selftest=1.
// ---------- self test (?selftest=1): exercised by headless Chrome in the project tests ----------
window.W7 = { S, buildCsv, parseCsv, importCsv, loadState, setLabel, go, exportCsv, TASKS };
if (SELFTEST){
  const out = [], ok = (n, c) => out.push((c ? 'PASS ' : 'FAIL ') + n);
  try {
    $('annot').value = 'annotator_01'; $('setSel').value = 'pilot'; loadState();
    const n = S.order.length; ok('pilot task count > 0', n > 0 && n === TASKS.filter(t => t.s === 'pilot').length);
    ok('no main pair in pilot order', S.order.every(t => t.s === 'pilot'));
    const o1 = S.order.map(t => t.p).join(); $('annot').value = 'annotator_02'; loadState();
    ok('annotator order differs', S.order.map(t => t.p).join() !== o1 || n < 3);
    $('annot').value = 'annotator_01'; loadState();
    const key = e => document.dispatchEvent(new KeyboardEvent('keydown', {key:e, bubbles:true}));
    key('2'); ok('key 2 sets label and advances', S.ans[S.order[0].p].label === '2' && S.idx === 1);
    key('0'); key('1'); key('u');
    ok('keys 0/1/U', S.ans[S.order[1].p].label === '0' && S.ans[S.order[2].p].label === '1' && S.ans[S.order[3].p].label === 'U');
    const before = S.idx; key('ArrowLeft'); ok('previous', S.idx === before - 1); key('ArrowRight'); ok('next', S.idx === before);
    go(0); $('note').value = 'ghi chú, có "dấu phẩy"\nvà xuống dòng'; $('note').dispatchEvent(new Event('input'));
    key('c'); ok('uncertain toggle recorded', S.ans[S.order[0].p].unc === true);
    ok('note stored', S.ans[S.order[0].p].note.includes('dấu phẩy'));
    const csv = buildCsv(); const rows = parseCsv(csv);
    ok('csv header', rows[0].join() === CSV_HEAD.join());
    ok('csv has one row per pair', rows.length === n + 1);
    ok('csv roundtrip note', rows[1][5].includes('xuống dòng') && rows[1][5].includes('"dấu phẩy"'));
    ok('csv annotator/set columns', rows[1][2] === 'annotator_01' && rows[1][1] === 'pilot');
    const saved = JSON.stringify(S.ans);
    S.ans = {}; S.idx = 0; loadState();   // simulates reload: state must come back from storage
    ok('state restored after reload', JSON.stringify(S.ans) === saved);
    const first = S.order[0].p; S.ans = {}; persist();
    const r1 = importCsv(csv); ok('import ok', r1.ok && r1.imported === 4);
    ok('import label restored', S.ans[first].label === '2' && S.ans[first].unc === true);
    ok('import rejects other annotator', !(S.annot = 'annotator_02', importCsv(csv)).ok);
    S.annot = 'annotator_01';
    ok('import rejects bad label', (() => { const bad = csv.replace(',2,', ',7,'); const before = JSON.stringify(S.ans); const r = importCsv(bad); return r.skipped >= 1; })());
    ok('export name', exportCsv() === 'annotator_01_pilot.csv');
    $('setSel').value = 'main'; loadState(); ok('main set separate and empty', Object.keys(S.ans).length === 0 && S.order.every(t => t.s === 'main'));
    ok('no method/rank/score words in task data', !/visual|metadata|rank|score|cosine|method|anchor/i.test(JSON.stringify(TASKS)));
    ok('no method words in page text', !/visual|metadata baseline|cosine|resnet/i.test(document.body.innerText));
  } catch (e) { out.push('FAIL exception ' + e.message); }
  try { for (const k of Object.keys(localStorage)) if (k.startsWith(KEY_PREFIX)) localStorage.removeItem(k); } catch (e) {}
  const el = $('selftest'); el.hidden = false; el.textContent = 'SELFTEST_RESULT\n' + out.join('\n') + '\nSELFTEST_END ' + (out.some(x => x.startsWith('FAIL')) ? 'FAIL' : 'PASS');
}
