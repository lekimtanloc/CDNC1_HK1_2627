import json
from initialize import O,R,sha,dump
from sanity import paths,read
checks={c:{'sha256':sha(f),'rows':len(read(f)),'source':str(f.relative_to(R))} for c,f in paths.items()}
assert all(v['rows']==5860 for v in checks.values());dump(O/'test_top10_source_checks.json',checks)
v=json.loads((R/'results/week2/FashionRecommender/data_protocol/validation_report.json').read_text());checks=[]
for r in v['outputs']:
 f=R/'results/week2/FashionRecommender/data_protocol'/r['path'];checks.append({'path':str(f.relative_to(R)),'sha256':sha(f),'match_week2':sha(f)==r['sha256'],'is_protocol_csv':f.suffix=='.csv'})
assert all(r['match_week2'] for r in checks if r['is_protocol_csv'])
dump(O/'split_source_checks.json',{'checks':checks,'note':'README.md differs from original week2 validation_report hash, but matches hash at start of this task. All CSV split/inventory/ground_truth hashes match original week2; README not used as numeric input.'})
print('PASS all source CSV hashes; README historical difference documented')
