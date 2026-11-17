// Tests-only: labels pairs from window.AUTO ({pair_id: label}) through the real interface code, then prints the exported CSV into #csvout.
(function () {
  if (!SELFTEST) return;
  $('annot').value = window.AUTO_ANNOT; loadState();
  const failures = [];
  S.order.forEach((t, i) => {
    const lab = window.AUTO[t.p];
    if (lab !== undefined) { go(i); if (setLabel(String(lab)) !== true) failures.push(t.p + ':' + lab); }
  });
  const pre = document.createElement('pre'); pre.id = 'csvout';
  pre.textContent = 'CSV_BEGIN\n' + buildCsv() + 'CSV_END\nREFUSED ' + JSON.stringify(failures);
  document.body.appendChild(pre);
  try { for (const k of Object.keys(localStorage)) if (k.startsWith(KEY_PREFIX)) localStorage.removeItem(k); } catch (e) {}
})();
