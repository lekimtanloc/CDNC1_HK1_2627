import json,io,zipfile
from collections import defaultdict
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from run_test_day_du import O,R,P,W,read,write,dump,truth,WARN
assert json.loads((O/'independent_result_day_du.json').read_text())['mismatches']==0
v1={r['image_path']:int(r['all_black']) for r in read(W/'artifacts_completed/crop_audit.csv') if r['mode']=='fixed_all'}
v2={r['image_path']:int(r['all_black']) for r in read(W/'v2/artifacts_v2/crop_audit_v2.csv')}
ex=read(W/'v2/artifacts_v2/bbox_extremes.csv');assert len(ex)==40
small={r['image_path'] for r in ex[:20]};large={r['image_path'] for r in ex[20:]}
assert not small&large
membership=[];groups=[];selection=[];allmain={}
axes={'n_answers':['1','2','3+'],'query_black_v1':['black','not_black'],'query_black_v2':['black','not_black'],'bbox_v2':['small_extreme','large_extreme','not_listed'],'gallery_product_black_v1':['has_black','no_black'],'gallery_product_black_v2':['has_black','no_black']}
for split in ('valid','test'):
 gt=truth(split);rr=read(P/f'{split}.csv');ip={r['image_path']:r['item_id'] for r in rr};gp=defaultdict(list)
 for r in rr:
  if r['role']=='gallery':gp[r['item_id']].append(r['image_path'])
 mm=[]
 for q,ans in gt.items():
  r={'split':split,'query_id':q,'n_answers':'3+' if len(ans)>=3 else str(len(ans)),'query_black_v1':'black' if v1[q] else 'not_black','query_black_v2':'black' if v2[q] else 'not_black','bbox_v2':'small_extreme' if q in small else ('large_extreme' if q in large else 'not_listed'),'gallery_product_black_v1':'has_black' if any(v1[x] for x in gp[ip[q]]) else 'no_black','gallery_product_black_v2':'has_black' if any(v2[x] for x in gp[ip[q]]) else 'no_black'};mm.append(r)
 membership+=mm
 data={c:{r['query_id']:r for r in read(O/(f'valid_{c}_per_query.csv' if split=='valid' else f'test_{c}_per_query_day_du.csv'))} for c in ('B2-v1','B2-v2')};allmain[split]=data
 for axis,values in axes.items():
  for value in values:
   qs=[r['query_id'] for r in mm if r[axis]==value];n=len(qs)
   for c in data:
    groups.append({'split':split,'axis':axis,'group':value,'config':c,'n':n,'Top-1':sum(float(data[c][q]['Top-1']) for q in qs)/n if n else '', 'Recall@5':sum(float(data[c][q]['Recall@5']) for q in qs)/n if n else '', 'interpretation':'n nhỏ, không diễn giải' if n<20 else 'chỉ mô tả; chưa kiểm định nhóm, không suy nguyên nhân','semantic_note':WARN})
 if split=='valid':
  for tag,a,b in [('v1_dung_v2_sai',1,0),('v1_sai_v2_dung',0,1)]:
   qs=sorted(q for q in gt if float(data['B2-v1'][q]['Top-1'])==a and float(data['B2-v2'][q]['Top-1'])==b)[:5];assert len(qs)==5
   selection.extend({'query_id':q,'split':'valid','direction':tag,'selection_rule':'first 5 query_id ascending in each transition group; not random'} for q in qs)
write(O/'group_membership_day_du.csv',membership);write(O/'group_metrics_day_du.csv',groups);write(O/'montage_selection_day_du.csv',selection)
dump(O/'group_rules_day_du.json',{'bbox_source':'v2/artifacts_v2/bbox_extremes.csv','small':'first20 rows: 20 smallest bbox area fractions all splits','large':'last20 rows: 20 largest bbox area fractions all splits','not_listed':'not among saved extremes; not proof bbox is normal','query_black':'fixed bbox crop, corresponding v1/v2 audit','gallery_product_black':'any gallery image of query product is black in specified pipeline','group_metrics':'both configs use same query set within each axis/group','small_n_rule':'n<20: n nhỏ, không diễn giải','overlap':'axes overlap; do not sum counts across axes','semantic_note':WARN})
# Independent group aggregation from independently recomputed query scores.
ind=read(O/'independent_scores_day_du.csv');checks=[]
for r in groups:
 qs={m['query_id'] for m in membership if m['split']==r['split'] and m[r['axis']]==r['group']};ss=[v for v in ind if v['split']==r['split'] and v['config']==r['config'] and v['query_id'] in qs];assert len(ss)==r['n']
 for metric in ('Top-1','Recall@5'):
  val=np.mean([float(v[metric]) for v in ss]) if ss else ''
  checks.append({'split':r['split'],'config':r['config'],'axis':r['axis'],'group':r['group'],'n':len(ss),'metric':metric,'main':r[metric],'independent':val,'match':val=='' if not ss else abs(val-r[metric])<1e-12})
write(O/'independent_groups_day_du.csv',checks);assert all(r['match'] for r in checks)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',13);bold=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',15)
canvas=Image.new('RGB',(1560,2310),'white');draw=ImageDraw.Draw(canvas)
draw.text((10,5),'VALIDATION: 5 mẫu mỗi chiều, theo query_id tăng dần; không ngẫu nhiên',font=bold,fill='black')
draw.text((10,26),'B2-v1 KHÔNG đúng ngữ nghĩa (giữ 4,9; bỏ 13,15). Chỉ mô tả retrieval proxy, chưa giải thích nguyên nhân.',font=font,fill='black')
rankings={c:defaultdict(list) for c in ('B2-v1','B2-v2')}
for c in rankings:
 for r in read(O/f'valid_{c}_top10.csv'):
  if int(r['rank'])<=5:rankings[c][r['query_id']].append(r)
with zipfile.ZipFile(R/'archive.zip') as z:
 for j,s in enumerate(selection):
  y=60+j*225;q=s['query_id'];draw.text((5,y),f"{j+1}. {q} — {s['direction']}",font=bold,fill='black')
  for side,c in enumerate(('B2-v1','B2-v2')):
   x=side*780;draw.text((x+430,y),c,font=bold,fill='black');rr=sorted(rankings[c][q],key=lambda r:int(r['rank']));imgs=[(q,'Ảnh hỏi')]+[(r['candidate_id'],f"#{r['rank']} {'ĐÚNG' if r['is_correct']=='1' else 'SAI'} SP {r['product_id']}") for r in rr]
   for col,(path,label) in enumerate(imgs):
    im=Image.open(io.BytesIO(z.read(path))).convert('RGB');im=ImageOps.contain(im,(126,171));canvas.paste(im,(x+col*130+(126-im.width)//2,y+24));draw.text((x+col*130+1,y+198),label,font=font,fill='black')
  draw.line((0,y+223,1560,y+223),fill='#bbbbbb')
canvas.save(O/'figures/hinh_04_mau_v1_dung_v2_sai_day_du.png',dpi=(180,180))
# Twelve reviewed failures; grouping is descriptive and overlapping for the tunic/dress case.
failed=[r for r in read(O/'category_check_100.csv') if r['correct']=='0'];out=[]
for r in failed:
 item=r['item_id'];tags=[]
 if item in {'2360','4822','5396','5391','3289'}:tags.append('bộ đồ cả áo và quần')
 if item in {'4036','1055','3161','3304','4486'}:tags.append('váy/váy liền/playsuit')
 if item=='3304':tags.append('váy liền/áo dài phối quần')
 if item in {'3909','3332'}:tags.append('mask đen')
 assert tags,item
 out.append({'item_id':item,'groups':'; '.join(tags),'original_agent_note':r['note']})
assert len(out)==12;write(O/'category_error_groups_day_du.csv',out)
print('Groups independently verified; validation-only montage and error taxonomy saved')
