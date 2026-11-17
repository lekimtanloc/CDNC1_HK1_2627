// Tests-only browser self-test for the subset interfaces (NOT shipped). Injected before </body>; runs with ?selftest=1.
(function () {
  if (!SELFTEST) return;
  const out = [], ok = (n, c) => out.push((c ? 'PASS ' : 'FAIL ') + n);
  const LEAK = /visual|metadata|cosine|resnet|anchor|top-?5|rank|score|method|baseline|\bcnn\b|embedding|tf-?idf|evaluator|amendment/i;
  try {
    const K = e => document.dispatchEvent(new KeyboardEvent('keydown', {key: e, bubbles: true}));
    const snap = () => JSON.stringify(S.ans);
    for (const bad of ['annotator_01', 'annotator_02', 'annotator_3', 'x', 'annotator_00']) {
      $('annot').value = bad; loadState();
      ok('id ' + bad + ' rejected', S.order.length === 0 && $('work').style.display === 'none' && setLabel('0') === false);
    }
    $('annot').value = 'annotator_03'; loadState();
    const n = S.order.length;
    ok('tasks loaded', n >= 2 && n === TASKS.length);
    ok('csv header', parseCsv(buildCsv())[0].join() === CSV_HEAD.join());
    ok('no method words in page', !LEAK.test(document.body.innerText) && !LEAK.test(JSON.stringify(TASKS)));
    const first = S.order[0];
    if (MODE === 'second_judgment') {
      ok('task keys are only p,q,c,o (no label)', TASKS.every(t => Object.keys(t).sort().join() === 'c,o,p,q'));
      ok('no label words in config', !/opts|"label"/.test(document.getElementById('cfg-json').textContent));
      ok('key 2 labels and advances', setLabel('2') === true && S.ans[first.p].label === '2' && (n < 2 || S.idx === 1));
      if (n >= 3) { K('0'); ok('key 0', S.ans[S.order[1].p].label === '0'); }
      if (n >= 4) { K('u'); ok('key U', S.ans[S.order[2].p].label === 'U'); }
      const before = snap(); K('x'); K('9'); ok('other keys ignored', snap() === before);
    } else {
      const t0 = first, [a, b] = t0.opts;
      ok('two options ascending and distinct', a < b && TASKS.every(t => t.opts.length === 2 && t.opts[0] < t.opts[1]));
      ok('both labels shown without attribution', $('twoLabels').textContent === a + ' và ' + b && !/annotator|người chấm số/i.test($('twoBox').textContent));
      const hidden = ['0', '1', '2'].filter(x => !t0.opts.map(String).includes(x));
      ok('foreign label button hidden', hidden.every(x => document.querySelector('.lab[data-label="' + x + '"]').style.display === 'none'));
      const before = snap();
      ok('setLabel outside the two options refused', hidden.every(x => setLabel(x) === false) && snap() === before);
      K(hidden[0]); ok('key outside options ignored', snap() === before);
      ok('choose one existing label', setLabel(String(b)) === true && S.ans[t0.p].label === String(b));
      if (n >= 2) { go(1); ok('U allowed (no decision)', setLabel('U') === true && S.ans[S.order[1].p].label === 'U'); }
    }
    go(0); $('note').value = 'ghi chú, có "dấu phẩy"\nvà xuống dòng'; $('note').dispatchEvent(new Event('input'));
    K('c'); ok('uncertain + note stored', S.ans[first.p].unc === true && S.ans[first.p].note.includes('dấu phẩy'));
    go(1); go(0); ok('navigation keeps answers', S.ans[first.p].label !== '' && S.ans[first.p].note.includes('xuống dòng'));
    const csv = buildCsv(), rows = parseCsv(csv);
    ok('one csv row per task', rows.length === n + 1);
    ok('csv role/annotator/set', rows[1][7] === MODE && rows[1][2] === 'annotator_03' && rows[1][1] === 'main');
    ok('csv note roundtrip', rows[1][5].includes('"dấu phẩy"') && rows[1][5].includes('xuống dòng'));
    const saved = snap(); S.ans = {}; S.idx = 0; loadState();
    ok('state restored after reload', snap() === saved);
    S.ans = {}; persist(); const r = importCsv(csv);
    ok('import valid export', r.ok && snap() === saved);
    const labelled = S.order.find(t => S.ans[t.p] && S.ans[t.p].label !== '');
    const lab = S.ans[labelled.p].label;
    const reject = (name, text) => { const b0 = snap(); const rr = importCsv(text); ok('import rejected: ' + name + ' (state untouched)', rr.ok === false && snap() === b0); };
    reject('unknown pair_id', csv.replace(first.p, 'Pdeadbeef00'));
    reject('wrong role', csv.split(',' + MODE).join(',' + (MODE === 'second_judgment' ? 'adjudication' : 'second_judgment')));
    reject('other annotator', csv.split(',annotator_03,').join(',annotator_04,'));
    reject('invalid label', csv.replace(',annotator_03,' + lab + ',', ',annotator_03,7,'));
    reject('duplicate pair', csv.trimEnd() + '\r\n' + rows[1].map(csvEsc).join(',') + '\r\n');
    reject('missing role column', csv.replace(',role', ',rol').replace(',' + MODE + '\r\n', ',' + MODE + '\r\n'));
    if (MODE === 'adjudication') {
      const bad = ['0', '1', '2'].find(x => !labelled.opts.map(String).includes(x));
      reject('label outside the two options', csv.replace(',annotator_03,' + lab + ',', ',annotator_03,' + bad + ','));
    }
    ok('export file name', exportCsv() === 'annotator_03_' + MODE + '.csv');
    ok('saved answers not lost after failed imports', snap() === saved);
  } catch (e) { out.push('FAIL exception ' + e.message); }
  try { for (const k of Object.keys(localStorage)) if (k.startsWith(KEY_PREFIX)) localStorage.removeItem(k); } catch (e) {}
  const el = $('selftest'); el.hidden = false;
  el.textContent = 'SELFTEST_RESULT\n' + out.join('\n') + '\nSELFTEST_END ' + (out.some(x => x.startsWith('FAIL')) ? 'FAIL' : 'PASS');
})();
