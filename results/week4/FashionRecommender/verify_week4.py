"""Read-only validation of completed Week 4 artifacts and frozen input hashes."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]); manifest=json.loads((out/'manifest.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
for r in manifest['inputs']: assert sha(r['path'])==r['sha256'],r['path']
for r in manifest['outputs']: assert sha(out/r['path'])==r['sha256'],r['path']
audit=[r for r in read(out/'crop_audit.csv') if r['mode']=='fixed_all']
assert {r['image_path'] for r in audit if r['all_black']=='1'}=={r['image_path'] for r in audit if r['empty_garment_mask']=='1'}
idx=read(out/'fixed_all_index.csv'); a=np.load(out/'fixed_all.npy',mmap_mode='r')
assert a.shape==(13752,2048) and np.isfinite(a).all()
assert np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5)
test=[r for r in idx if r['split']=='test']; queries={r['image_path'] for r in test if r['role']=='query'}; candidates={r['image_path']:r['item_id'] for r in test if r['role']=='gallery'}
protocol=next(Path(r['path']).parent for r in manifest['inputs'] if r['path'].endswith('/ground_truth_test.csv'))
gt={(r['query_image_path'],r['relevant_gallery_path']) for r in read(protocol/'ground_truth_test.csv')}
for name in ('visual_before','visual_only','popularity_only'):
    rr=read(out/f'{name}_top10.csv'); assert len(rr)==5860
    assert list(rr[0])==['query_id','rank','candidate_id','product_id','score','is_correct']
    assert {r['query_id'] for r in rr}==queries
    for q in queries:
        group=[r for r in rr if r['query_id']==q]
        assert [int(r['rank']) for r in group]==list(range(1,11))
        assert len({r['candidate_id'] for r in group})==10
        assert all(float(group[i]['score'])>=float(group[i+1]['score']) for i in range(9))
    for r in rr:
        assert r['candidate_id']!=r['query_id']
        assert candidates[r['candidate_id']]==r['product_id']
        assert int(r['is_correct'])==int((r['query_id'],r['candidate_id']) in gt)
pop={r['product_id']:float(r['popularity']) for r in read(out/'popularity.csv')}
expected=sorted(candidates,key=lambda p:(-pop[candidates[p]],hashlib.sha256(f'42:{p}'.encode()).hexdigest(),p))[:10]
for q in queries:
    group=[r for r in read(out/'popularity_only_top10.csv') if r['query_id']==q]
    assert [r['candidate_id'] for r in group]==expected
    assert all(float(r['score'])==pop[r['product_id']] for r in group)
print('PASS: frozen inputs and output hashes; 13,752 finite normalized image vectors; three CSVs, 5,860 rows each; query/gallery, ranks, scores and relevance.')
