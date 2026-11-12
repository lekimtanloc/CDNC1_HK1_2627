"""Independent count/formula checks plus report generation; no new evaluation metrics."""
from collections import Counter,defaultdict
from common import *
ART=BASE/'artifacts_v2'
def unique_stats(a,rows):
 a=np.ascontiguousarray(a);keys=a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).ravel();_,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True);ids=np.where(counts>1)[0];products=set()
 for g in ids:
  pp={rows[i]['item_id'] for i in np.flatnonzero(inverse==g)}
  if len(pp)>1:products.update(pp)
 return int(counts[counts>1].sum()),len(products)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def main():
 rows=read(ART/'fixed_all_v2_index.csv');oldrows=read(OLD/'fixed_all_index.csv');assert rows==oldrows
 a=np.load(ART/'fixed_all_v2.npy');v1=np.load(OLD/'fixed_all.npy');audit=read(ART/'crop_audit_v2.csv');m=json.load(open(ART/'manifest_v2.json'));checks=json.load(open(ART/'crop_checks_v2.json'));san=json.load(open(ART/'sanity_checks_v2.json'))['results'];repro=json.load(open(ART/'reproducibility.json'));mapping=read(BASE/'mapping/mapping_de_xuat.csv');decision=json.load(open(BASE/'mapping/decision.json'));keep=decision['keep_labels'];idx={r['image_path']:i for i,r in enumerate(rows)}
 assert a.shape==(13752,2048) and np.isfinite(a).all() and np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5)
 for f in m['inputs']:assert sha(f['path'])==f['sha256'],f['path']
 for f in m['outputs']:assert sha(BASE/f['path'])==f['sha256'],f['path']
 # Verify raw mask-derived empty flags independently from per-image label pixel counts.
 raw=read(BASE/'mapping/thong_ke_tung_anh.csv');kept=Counter()
 for r in raw:
  if int(r['label']) in keep:kept[r['image_path']]+=int(r['pixels'])
 for r in audit:assert int(r['kept_pixels'])==kept[r['image_path']];assert int(r['empty_keep'])==int(kept[r['image_path']]==0);assert int(r['all_black'])==int(r['empty_keep'])
 truth=defaultdict(set)
 for r in read(PROTOCOL/'ground_truth_test.csv'):truth[r['query_image_path']].add(r['relevant_gallery_path'])
 test=read(PROTOCOL/'test.csv');queries=[r for r in test if r['role']=='query'];gallery=[r for r in test if r['role']=='gallery'];candidates={r['image_path']:r['item_id'] for r in gallery};tidx=[idx[r['image_path']] for r in queries+gallery];meta=read(ROOT/'FashionRecommender/repro_outputs_resnet50_full/metadata.csv');mi=[idx[r['image_path']] for r in meta]
 independent={};top1={};rankings={}
 for name,path in [('v1_visual',OLD/'visual_only_top10.csv'),('v2_visual',ART/'visual_only_v2_top10.csv'),('popularity_only',ART/'popularity_only_top10.csv')]:
  rr=read(path);assert len(rr)==5860 and list(rr[0])==['query_id','rank','candidate_id','product_id','score','is_correct'];groups=defaultdict(list)
  for r in rr:
   assert candidates[r['candidate_id']]==r['product_id'];assert r['query_id']!=r['candidate_id'];assert int(r['is_correct'])==int(r['candidate_id'] in truth[r['query_id']]);groups[r['query_id']].append(r)
  assert set(groups)==set(truth)
  for q,rs in groups.items():assert [int(r['rank']) for r in rs]==list(range(1,11));assert len({r['candidate_id'] for r in rs})==10
  hits={q:int(rs[0]['candidate_id'] in truth[q]) for q,rs in groups.items()};recall=sum(sum(r['candidate_id'] in truth[q] for r in rs[:5])/len(truth[q]) for q,rs in groups.items())/len(groups);independent[name]={'top1':sum(hits.values())/len(groups),'recall_at_5':recall};top1[name]=hits;rankings[name]=groups
 # Recompute complete cosine ranks and ensure the v2 CSV follows stable gallery order.
 sim=a[[idx[r['image_path']] for r in queries]]@a[[idx[r['image_path']] for r in gallery]].T
 for qi,q in enumerate(queries):
  order=np.argsort(-sim[qi],kind='stable')[:10];assert [gallery[int(i)]['image_path'] for i in order]==[r['candidate_id'] for r in rankings['v2_visual'][q['image_path']]];assert np.array_equal(sim[qi,order],np.array([float(r['score']) for r in rankings['v2_visual'][q['image_path']]],dtype=np.float32))
 oldblack={r['image_path'] for r in read(OLD/'crop_audit.csv') if r['mode']=='fixed_all' and r['all_black']=='1'};black={r['image_path'] for r in audit if r['all_black']=='1'};blackq=[r for r in queries if r['image_path'] in black];blackg=[r for r in gallery if r['image_path'] in black]
 counts=[]
 def add(key,reported,actual,source,tol=1e-10):
  reported=reported.item() if isinstance(reported,np.generic) else reported;actual=actual.item() if isinstance(actual,np.generic) else actual
  match=abs(float(reported)-float(actual))<=tol;counts.append({'quantity':key,'reported':reported,'recalculated':actual,'match':match,'source':source});assert match,(key,reported,actual)
 add('images',13752,len(rows),'fixed_all_v2_index.csv');add('dimension',2048,a.shape[1],'fixed_all_v2.npy');add('queries',586,len(queries),'test.csv');add('gallery',1543,len(gallery),'test.csv');add('gt_pairs',1439,sum(map(len,truth.values())),'ground_truth_test.csv');add('top_k',10,len(next(iter(rankings['v2_visual'].values()))),'visual_only_v2_top10.csv');add('csv_rows',5860,sum(map(len,rankings['v2_visual'].values())),'visual_only_v2_top10.csv')
 for sp,num in [('train',3219),('valid',691),('test',690)]:add('products_'+sp,num,len({r['item_id'] for r in rows if r['split']==sp}),'index + frozen split');add('images_'+sp,len([r for r in rows if r['split']==sp]),len(read(PROTOCOL/f'{sp}.csv')),'index vs frozen split')
 add('black_v1_all',checks['v1_black_all'],len(oldblack),'v1 crop_audit.csv');add('black_v1_test',checks['v1_black_test'],sum(r['image_path'] in oldblack for r in test),'v1 audit + test.csv');add('black_v2_all',checks['black_all'],len(black),'audit vs raw mask statistics')
 for sp in ['train','valid','test']:add('black_v2_'+sp,checks['black_by_split'][sp],sum(r['split']==sp and r['image_path'] in black for r in rows),'audit + index')
 add('empty_keep_total',checks['empty_count'],sum(kept[r['image_path']]==0 for r in rows),'raw mask counts vs audit')
 add('black_v2_queries',len(blackq),sum(r['image_path'] in black for r in queries),'audit + test roles');add('black_v2_gallery',len(blackg),sum(r['image_path'] in black for r in gallery),'audit + test roles')
 for typ in ['all_background','outside_keep_only']:add('empty_'+typ,checks['empty_classification'].get(typ,0),sum(r['empty_type']==typ for r in audit),'crop_audit_v2.csv')
 for version,arr in [('v1',v1),('v2',a)]:
  for scope,aa,rr in [('all',arr,rows),('test',arr[tidx],queries+gallery),('representatives',arr[mi],meta)]:
   x,y=unique_stats(aa,rr);key=('v1_' if version=='v1' else '')+'duplicates_'+scope;add(f'{version}_duplicate_vectors_{scope}',checks[key]['images_in_groups'],x,'np.unique byte-row recomputation');add(f'{version}_duplicate_products_{scope}',checks[key]['products_in_cross_product_groups'],y,'np.unique + item_id cross-product groups')
 for name in san:
  for metric in san[name]:add(name+'_'+metric,san[name][metric],independent[name][metric],'manual counting from Top-10 and GT')
 trs=read(BASE/'top1_chuyen_trang_thai_v1_v2.csv');trsaved=json.load(open(ART/'transition_counts.json'))
 for r in trs:assert int(r['v1_top1_correct'])==top1['v1_visual'][r['query_id']];assert int(r['v2_top1_correct'])==top1['v2_visual'][r['query_id']];assert r['transition']==f"{top1['v1_visual'][r['query_id']]}->{top1['v2_visual'][r['query_id']]}"
 for group in trsaved:
  sel=trs if group=='all' else ([r for r in trs if r['v1_black']=='1'] if group=='black_v1' else ([r for r in trs if r['v2_black']=='1'] if group=='black_v2' else [r for r in trs if r['group']==group]));add('transition_'+group+'_n',trsaved[group]['n'],len(sel),'per-query CSV, hits checked independently')
  for key in ['1->0','0->1','1->1','0->0']:add('transition_'+group+'_'+key,trsaved[group]['counts'].get(key,0),sum(r['transition']==key for r in sel),'per-query transitions')
 for ver,arr in [('v1',v1),('v2',a)]:add('cosine_1687_3445_'+ver,checks['special_cosine_'+ver],float(arr[idx['images/1687_011.jpg']]@arr[idx['images/3445_011.jpg']]),'direct dot product')
 b=np.load(ART/'replay_1.npy');c=np.load(ART/'replay_2.npy');assert b.tobytes()==c.tobytes();add('replay_images',repro['images'],b.shape[0],'replay arrays');add('replay_equal_rows',repro['equal_rows'],sum(np.array_equal(x,y) for x,y in zip(b,c)),'replay arrays');add('replay_max_abs_diff',repro['max_abs_diff'],float(np.abs(b-c).max()),'replay arrays')
 bbox=np.array([float(r['bbox_area_fraction']) for r in audit]);quant=np.quantile(bbox,[0,.01,.05,.5,.95,.99,1])
 for key,val in zip(['min','p01','p05','p50','p95','p99','max'],quant):add('bbox_'+key,checks['bbox_quantiles_all'][key],val,'quantiles from bbox audit')
 add('bbox_positive_below_1pct',checks['bbox_positive_below_1pct'],((bbox>0)&(bbox<.01)).sum(),'bbox audit');add('bbox_above_95pct',checks['bbox_above_95pct'],(bbox>.95).sum(),'bbox audit');add('elapsed_seconds',m['elapsed_seconds'],m['elapsed_seconds'],'manifest recorded measurement; not independent historical timing');add('rss_mib',m['peak_rss_bytes']/2**20,m['peak_rss_bytes']/1048576,'manifest bytes conversion; not independent historical measurement')
 for key,val in [('seed',42),('threads',4),('batch_size',8)]:add(key,val,m[key],'run manifest')
 for name in ['popularity_only_top10.csv','popularity.csv','popularity_lineage.json']:assert sha(OLD/name)==sha(ART/name)
 before=json.load(open(BASE/'hash_cu_truoc.json'));after={p:{'sha256':sha(ROOT/p),'bytes':(ROOT/p).stat().st_size} for p in before};assert before==after;dump(BASE/'hash_cu_sau.json',after);add('unchanged_old_files',len(before),sum(before[k]==after[k] for k in before),'before/after SHA-256')
 samples=read(BASE/'mapping/atlas_samples.csv');add('labels_reviewed',16,len(mapping),'mapping_de_xuat.csv');add('atlas_main_cells',384,sum(r['kind']=='random24' for r in samples),'atlas_samples.csv');add('atlas_total_cells',decision['reviewed_total_cells'],len(samples),'atlas_samples.csv');add('keep_label_count',len(keep),sum(r['nhom'] in ['quan-ao','giay-phu-kien'] for r in mapping),'mapping proposal vs config');add('replay_runs',2,len([ART/'replay_1.npy',ART/'replay_2.npy']),'two saved extraction outputs')
 # Inspect equal-score effects from the same full matrix, without changing rankings.
 ties=[]
 for qi,q in enumerate(queries):
  if q['image_path'] in black:
   order=np.argsort(-sim[qi],kind='stable');ties.append({'query_id':q['image_path'],'first_ten':[gallery[int(i)]['image_path'] for i in order[:10]],'black_gallery_ranks':[int(np.flatnonzero(order==j)[0])+1 for j,g in enumerate(gallery) if g['image_path'] in black]})
 blackvec=a[[idx[r['image_path']] for r in blackq+blackg]];tieinfo={'black_queries':len(blackq),'black_gallery':len(blackg),'black_vectors_byte_equal':all(np.array_equal(blackvec[0],v) for v in blackvec) if len(blackvec) else True,'black_queries_top10':ties,'cosine_min':float((blackvec@blackvec.T).min()) if len(blackvec) else None,'cosine_max':float((blackvec@blackvec.T).max()) if len(blackvec) else None};dump(ART/'tie_diagnostics.json',tieinfo)
 write(BASE/'doi_chieu_so_lieu.csv',counts)
 # All numeric experimental results below are sourced from verified entries above.
 verified={r['quantity']:r['recalculated'] for r in counts};dump(BASE/'verified_values.json',verified)
 map_table=table(['Nhãn','Tên vùng đề xuất','Tin cậy','Giữ/bỏ'],[[r['nhan'],r['ten_vung_de_xuat'],r['do_tin_cay'],'giữ' if int(r['nhan']) in keep else 'bỏ'] for r in mapping])
 comp=table(['Đại lượng / cùng phạm vi','v1','v2'],[['Ảnh đen toàn bộ',len(oldblack),len(black)],['Ảnh đen test',checks['v1_black_test'],checks['black_by_split']['test']],['Vector trùng toàn bộ',checks['v1_duplicates_all']['images_in_groups'],checks['duplicates_all']['images_in_groups']],['Sản phẩm trong nhóm trùng khác sản phẩm',checks['v1_duplicates_all']['products_in_cross_product_groups'],checks['duplicates_all']['products_in_cross_product_groups']],['Vector trùng trên ảnh đại diện',checks['v1_duplicates_representatives']['images_in_groups'],checks['duplicates_representatives']['images_in_groups']],['Top-1 (sanity)',f"{100*san['v1_visual']['top1']:.4f}%",f"{100*san['v2_visual']['top1']:.4f}%"],['Recall@5 (sanity)',f"{100*san['v1_visual']['recall_at_5']:.4f}%",f"{100*san['v2_visual']['recall_at_5']:.4f}%"]])
 transition_table=table(['Nhóm query','n','Đúng→sai','Sai→đúng','Đúng cả hai','Sai cả hai'],[[g,trsaved[g]['n'],*[trsaved[g]['counts'].get(k,0) for k in ['1->0','0->1','1->1','0->0']]] for g in ['all','black_v1','black_v2','neither_black','black_both','black_v1_only','black_v2_only']])
 sanity_table=table(['Cấu hình','Top-1','Recall@5'],[[k,f"{100*v['top1']:.4f}%",f"{100*v['recall_at_5']:.4f}%"] for k,v in san.items()])
 text_report=f'''# Kết quả tuần 04 v2 — sửa bộ lọc mask

## 1. Đã làm gì
Đã hoàn tất xác minh thực nghiệm mapping, qua điểm dừng rồi trích lại {len(rows):,} vector ảnh, mỗi vector {a.shape[1]:,} chiều. Giữ split sản phẩm 3.219/691/690, cùng {len(queries)} query, {len(gallery):,} gallery, {sum(map(len,truth.values())):,} cặp đúng. Không huấn luyện, đổi backbone, đổi split hay chạy nhiều seed.

Đã xuất Top-10 visual v2 ({len(queries)*10:,} dòng) và bản tham chiếu popularity giống byte v1. Trích hai lần cùng tập con {repro['images']} ảnh gồm các ảnh đặc biệt; {repro['equal_rows']} vector bằng byte, max |Δ|={repro['max_abs_diff']}. Hash trước/sau xác nhận {len(before)} file cũ không đổi, gồm tuần 1–3, mã/cache cũ, artifacts_completed và addendum.

## 2. Mapping nhãn và cách sửa crop
Mapping là **đề xuất thực nghiệm**, chưa có tài liệu chính thức nối bảng mã với archive adjusted. Đã xem {decision['reviewed_main_cells']} ô mẫu chính và tổng {decision['reviewed_total_cells']} ô kể cả bổ sung. Mẫu chính chọn seed 42, trải đều các split; có thống kê tâm/diện tích/co-xuất hiện toàn bộ mask. Nguồn và giới hạn: [nguon_mapping.md](nguon_mapping.md); atlas, từng mẫu và ngoại lệ: [mapping/nhan_atlas.md](mapping/nhan_atlas.md).

{map_table}
Nhãn có độ tin cậy trung_binh được hiểu là trung bình-cao cho quyết định giữ tất/quần tất, có ngoại lệ nên không ghi cao. Các nhóm còn lại rõ trên nhiều mẫu đã xem. **Qua điểm dừng**; không có nhóm thấp/chưa rõ đáng kể cần chạy v2b. Mức tin cậy không phải độ chính xác segmentation đo được.

Tập giữ: `{keep}`; bỏ nền, mặt, tóc, da (`[0,4,9,14]`). So v1, thêm nhãn 13/15 và bỏ 4/9; không đổi cách resize, pad hay chuẩn hóa:
```python
selected = np.isin(mask, keep_labels)
masked = np.where(selected[..., None], image_rgb, 0)
y, x = np.where(selected)
fixed = Image.new("RGB", (224, 224))
if len(x):
    bbox = (x.min(), y.min(), x.max()+1, y.max()+1)
    fitted = ImageOps.contain(Image.fromarray(masked).crop(bbox),
                              (224, 224), Image.Resampling.BILINEAR)
    fixed.paste(fitted, ((224-fitted.width)//2, (224-fitted.height)//2))
```
Ảnh không có nhãn giữ được để đen, không thay bằng ảnh gốc. Sau đó normalize ImageNet, ResNet-50 IMAGENET1K_V2, L2-normalize. Audit từng ảnh có nhãn, bbox, diện tích và trạng thái đen.

Ảnh đen theo train/valid/test: {checks['black_by_split']['train']}/{checks['black_by_split']['valid']}/{checks['black_by_split']['test']}; tổng {checks['black_all']}. Không có pixel nhãn giữ: {checks['empty_count']}, gồm {checks['empty_classification'].get('all_background',0)} toàn nền và {checks['empty_classification'].get('outside_keep_only',0)} có nhãn ngoài tập giữ. Bbox/diện tích ảnh (tính cả bbox rỗng=0): min={quant[0]:.6f}, p01={quant[1]:.6f}, p05={quant[2]:.6f}, median={quant[3]:.6f}, p95={quant[4]:.6f}, p99={quant[5]:.6f}, max={quant[6]:.6f}. Có {checks['bbox_positive_below_1pct']} bbox dương dưới 1% ảnh và {checks['bbox_above_95pct']} bbox trên 95%; danh sách cực trị ở bbox_extremes.csv, đây là cờ kiểm tra chứ không tự kết luận lỗi.

Cosine hai ảnh 1687_011 và 3445_011: v1={checks['special_cosine_v1']:.8f}, v2={checks['special_cosine_v2']:.8f}. Ảnh gốc, vùng giữ và crop v2 của cả hai cùng 1000_031 ở [special_crop_v2.png](artifacts_v2/special_crop_v2.png).

## 3. Hai baseline
Visual-only xếp ảnh gallery theo cosine giảm dần; `argsort(kind="stable")` giữ thứ tự gallery test.csv khi hòa điểm. Loại chính query, không gộp ảnh về sản phẩm, không lọc category. CSV có `query_id,rank,candidate_id,product_id,score,is_correct`, K=10.

Popularity-only độc lập vector: sao chép nguyên byte CSV xếp hạng, popularity.csv và lineage v1; cả ba hash khớp trong popularity_reference.json. Nguồn là `week3_verify_and_smoke.py::smoke()`: `default_rng(42).integers(0,101,len(metadata))`, chuẩn hóa min–max, giữ thứ tự metadata cũ; inventory.csv/tracking.sqlite3 không chứa số mua hàng. Giá trị vẫn là giả lập; xếp giảm dần, hòa điểm theo SHA-256 của `42:candidate_path`. Không sinh lại giá trị hoặc thêm baseline.

Ảnh đen test v2: {len(blackq)} query và {len(blackg)} gallery (black_test.csv). Các vector đen bằng byte: {tieinfo['black_vectors_byte_equal']}; cosine đen×đen gần 1 (chi tiết số float32 ở tie_diagnostics.json). Thứ hạng các gallery đen theo từng query có trong black_gallery_ranks.csv. Khi query đen, điểm hòa trong nhóm đen không phản ánh độ giống trang phục; stable gallery order quyết định thứ hạng giữa chúng. Không đổi tie-break để làm đẹp kết quả.

## 4. Sanity check và so sánh v1–v2
**Chỉ kiểm tra pipeline, chưa phải bảng đánh giá chính thức.** Dùng nguyên week3_retrieval_metrics.evaluate: Top-1 lấy reciprocal_rank ở K=1; Recall@5 là số ảnh đáp án đúng trong Top-5 / tổng số ảnh đáp án của query, trung bình đều trên query. Cấp ảnh, không phải hit-rate “ít nhất một đúng”. Không báo NDCG/MRR hay metric khác.

{sanity_table}
### So sánh v1 và v2
{comp}
Vector trùng là bằng byte float32; ảnh đại diện là đúng tập metadata cũ, không đổi phạm vi so sánh. Mọi con số bảng được tính lại trong doi_chieu_so_lieu.csv.

### Chuyển trạng thái Top-1
{transition_table}
Hai hàng black_v1/black_v2 có thể giao nhau; bốn nhóm black_both/black_v1_only/black_v2_only/neither_black là phân hoạch. Chỉ đếm, không kiểm định và không kết luận “cải thiện” có ý nghĩa. Dữ liệu từng query: top1_chuyen_trang_thai_v1_v2.csv.

## 5. Hình minh họa
Sáu hình tiếng Việt ở figures/: hinh_01_atlas_nhan.png, hinh_02_crop_v1_v2.png, hinh_03_top5_v1_v2.png, hinh_04_so_do_pipeline.png, hinh_05_so_sanh_sanity.png, hinh_06_anh_den_v2.png. Nguồn, cách chọn mẫu và caption nằm trong [figures/mo_ta_hinh.md](figures/mo_ta_hinh.md). Atlas đầy đủ nằm trong mapping/; không chỉ dựa vào hình tóm tắt.

## 6. Vấn đề / giới hạn
- Chưa xác minh được tài liệu đánh số chính thức của archive. Mapping dựa trên atlas ngẫu nhiên và ảnh phóng vùng; không coi là nhãn do nhà cung cấp xác nhận. Có lỗi mask riêng lẻ, ví dụ vùng mặt bị gán nhãn tất, váy bị gán nhãn da. Phân biệt subtype của chân váy/quần ngắn, trang sức, áo buộc eo không tuyệt đối nhưng quyết định giữ cùng nhóm.
- Category tuần 2 suy từ mapping cũ và được dùng stratify sản phẩm; vì thế phân tầng không phải category đã xác minh. Ground truth chỉ dùng cùng item_id, không lọc theo category; vẫn giữ split để so công bằng.
- Popularity giả lập; bài toán cùng sản phẩm khác góc, không phải hành vi người dùng/phối đồ. Nhiều ảnh cùng sản phẩm có thể cùng chiếm Top-K.
- Lần trích đầu bị ngắt bởi lỗi I/O khi đọc archive, chưa lưu vector; đã đọc lại và kiểm tra SHA-256 khớp trước khi chạy lại thành công. Nhật ký: execution_attempts.json. Thời gian dưới đây chỉ tính lần hoàn tất.
- Không bảo đảm mask đúng tại mọi pixel, vector bằng byte trên mọi máy/phiên bản; kiểm tra tái lập chỉ trong môi trường hiện tại. Thời gian/RSS là số đo của tiến trình lần chạy, không đo toàn máy.

## 7. Minh chứng và tái chạy
- mapping/: mapping_de_xuat.csv, decision.json, nhan_atlas.md, nhan_thong_ke.csv, thong_ke_tung_anh.csv, dong_xuat_hien.csv, on_dinh_giua_split.csv, atlas_samples.csv và PNG atlas.
- artifacts_v2/: fixed_all_v2.npy + index; crop_audit_v2.csv, crop_checks_v2.json, bbox_extremes.csv; visual_only_v2_top10.csv, popularity tham chiếu; sanity_checks_v2.json; replay_1.npy/replay_2.npy + index + reproducibility.json; black_test.csv, black_gallery_ranks.csv, tie_diagnostics.json; manifest_v2.json.
- hash_cu_truoc.json/hash_cu_sau.json: hash trước/sau file cũ; doi_chieu_so_lieu.csv và verified_values.json: tính lại số; verification_v2.txt: log kiểm tra. manifest_ban_giao_v2.json và muc_luc_goi.md kiểm kê bản bàn giao; file_npy_bo_qua.txt ghi NPY để riêng nếu gói quá lớn.

Môi trường giữ như v1: Python {m['versions']['python']}, torch {m['versions']['torch']}, torchvision {m['versions']['torchvision']}, NumPy {m['versions']['numpy']}, Pillow {m['versions']['pillow']}; CPU, batch {m['batch_size']}, {m['threads']} threads, seed {m['seed']}, deterministic algorithms. Weights SHA-256 `{m['weights_sha256']}`. Pipeline {m['elapsed_seconds']:.2f} giây, peak RSS {m['peak_rss_bytes']/2**20:.2f} MiB; gồm extraction chính và hai lần tập con, không gồm tạo atlas/đóng gói.

Lệnh từ .., theo thứ tự (xem atlas trước khi ghi quyết định):
```bash
python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/build_mapping_evidence.py
python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/record_mapping_review.py
{m['command']}
python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/verify_and_report.py
python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/make_figures_v2.py
python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/package_v2.py
```
Mã từ chối ghi đè đầu ra. Tái chạy extraction có thể dùng `--output-dir` thư mục mới, nhưng file chẩn đoán cấp v2 cũng phải nằm trong bản sao cây thư mục mới chưa có đầu ra. Không chạy nguyên chuỗi lên bản bàn giao. Việc gán mapping là bước xem ảnh có ghi nhận, không phải suy tự động từ thống kê.

## 8. Dự kiến tuần 5
Chốt protocol và giới hạn mapping, sau đó mới lập bảng metric đánh giá chính thức hai baseline. Chưa thực hiện tuần 5 trong lần này.
'''
 (BASE/'ket_qua_tuan_04_v2.md').write_text(text_report)
 change=f'''# Thay đổi so với v1\n\nBộ lọc cũ 1–12 → {keep}; bỏ mặt/tóc, thêm quần/giày theo mapping thực nghiệm. Nền/da tiếp tục loại. Cách bbox, resize bilinear giữ tỉ lệ, pad, normalize, backbone, weights, seed, CPU, batch, split, gallery, GT và tie-break visual giữ nguyên. Không huấn luyện hoặc fusion.\n\n{comp}\n\nCác số trên thay thế phần mô tả kết quả chạy chính khi viết báo cáo v2; không xóa số v1 vì vẫn là đối chứng thật. Popularity giữ nguyên byte nên không sửa số của baseline này.\n\nThêm atlas 16 nhãn, thống kê toàn mask, kiểm tra chéo split, bộ audit bbox/ảnh đen/vector trùng, hai lượt replay, hình/caption, hash trước/sau và đối chiếu số độc lập. Giới hạn mới cần nêu: mapping chưa có xác nhận chính thức; lỗi segmentation riêng lẻ; category stratify tuần 2 kế thừa mapping cũ. Các file v1/addendum/tuần trước không bị ghi đè.\n'''
 (BASE/'danh_sach_thay_doi_so_voi_v1.md').write_text(change)
 print(f'PASS: {len(counts)} numeric checks; cosine ranks/schema/GT; two replay files byte-identical; {len(before)} old files unchanged.');print(json.dumps({'sanity':independent,'black_v2':len(black),'black_test':len(blackq)+len(blackg)},indent=2))
if __name__=='__main__':main()
