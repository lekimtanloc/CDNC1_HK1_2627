import os
os.environ['MPLCONFIGDIR']='/private/tmp/week6-mpl';os.environ['OPENBLAS_NUM_THREADS']='4'
import csv,json,math,hashlib,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent

def read(n):return list(csv.DictReader((O/n).open()))
def js(n):return json.loads((O/n).read_text())
def write(n,s):
 with (O/n).open('x') as f:f.write(s)
def csvwrite(n,rows):
 with (O/n).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def find(rows,**keys):return next(r for r in rows if all(str(r[k])==str(v) for k,v in keys.items()))
def cell(r,scale=1):return f"{float(r['value'])*scale:.5f} [{float(r['ci_low'])*scale:.5f}; {float(r['ci_high'])*scale:.5f}]"
def delta(r,scale=1):return f"{float(r['difference'])*scale:+.5f} [{float(r['ci_low'])*scale:+.5f}; {float(r['ci_high'])*scale:+.5f}]"
valid=read('metrics_valid.csv');test=read('metrics_test.csv');pairs=read('paired_test.csv');op=read('operating_points.csv');sens=read('sensitivity_summary.csv');refs=read('reference_valid.csv')+read('reference_test.csv');scale=read('scale_valid.csv');mc=read('mcnemar_bonferroni.csv');bf=read('paired_bonferroni.csv');leak=js('leakage_diagnostics_valid.json')
fig=O/'figures';fig.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
colors={'A':'#1565c0','B':'#d55e00','G2':'#008565'}
# All plotting numbers come from saved CSV tables.
f,ax=plt.subplots(figsize=(8.6,5.8),dpi=180)
for base in ['A','B','G2']:
 for g in [2,4]:
  points=[0.,.1,.2,.3,.4,.5];xx=[];yy=[]
  for p in points:
   keys=dict(base=base,wp=str(p),gamma=str(g if p else 2))
   xx.append(float(find(valid,metric='MAE@10',**keys)['value']));yy.append(100*float(find(valid,metric='Recall@5',**keys)['value']))
  ax.plot(xx,yy,marker='o',ms=3,color=colors[base],ls='-' if g==2 else '--',label=f'{base}, γ={g}')
  if g==2:ax.scatter(xx[3],yy[3],marker='*',s=130,color=colors[base],edgecolor='black',zorder=5)
 r=find(op,base=base,threshold_pp='2.0')
 if r['has_operating_point']=='True':ax.scatter(float(r['mae10']),100*float(r['recall5']),marker='D',s=70,color=colors[base],edgecolor='black',zorder=5)
for name,label,col in [('gallery_product_mean','Nền sản phẩm','#777777'),('closest10_products_including_query','Tối ưu đủ 10 sản phẩm','#aa77aa')]:ax.axvline(float(find(refs,split='valid',reference=name)['value']),color=col,ls=':',label=label)
ax.set(xlabel='Popularity MAE@10 (sản phẩm duy nhất trong Top-10 ảnh)',ylabel='Recall@5 (%)',title='Validation: 1.597 gallery / 568 ảnh hỏi');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
f.text(.02,.015,'★ P đã bỏ category; ◆ điểm vận hành. CI bỏ trên hình để dễ đọc, có trong metrics_valid.csv.\n“Tối ưu đủ 10 sản phẩm” không là chặn dưới tuyệt đối của MAE Top-10 ảnh; G2 vẫn cần mask.',fontsize=8)
f.tight_layout(rect=[0,.08,1,1]);f.savefig(fig/'01_danh_doi_validation.png');plt.close(f)
f,axs=plt.subplots(1,2,figsize=(8.6,4.8),dpi=180)
for ax,m,factor,label in zip(axs,['Recall@5','MAE@10'],[100,1],['Recall@5 (%)','Popularity MAE@10']):
 rr=[find(test,base='A',wp=str(p),gamma='2',metric=m) for p in [0.,.3]];v=np.array([float(r['value'])*factor for r in rr]);lo=np.array([float(r['ci_low'])*factor for r in rr]);hi=np.array([float(r['ci_high'])*factor for r in rr]);ax.bar([0,1],v,color=['#1565c0','#e69f00']);ax.errorbar([0,1],v,yerr=[v-lo,hi-v],fmt='none',color='black',capsize=4);ax.set_xticks([0,1],['Visual-only','P, đã bỏ category']);ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
f.suptitle('Test A: 1.543 gallery / 586 ảnh hỏi — CI bootstrap 95%');f.text(.03,.025,'Không có điểm vận hành A theo luật validation; không tạo cấu hình test thay thế.\nNguồn: metrics_test.csv. MAE chỉ đo phù hợp t giả lập.',fontsize=8);f.tight_layout(rect=[0,.09,1,.93]);f.savefig(fig/'02_test_A.png');plt.close(f)
f,axs=plt.subplots(1,2,figsize=(8.6,4.8),dpi=180)
for ax,g in zip(axs,[2,4]):
 for scenario,label,col in [('official','σ=∞ (t độc lập)','#444444'),('oracle_sigma0.3','σ=0,3 (oracle)','#d55e00'),('oracle_sigma0.1','σ=0,1 (oracle)','#1565c0')]:
  rr=[find(sens,scenario=scenario,wp=str(p),gamma=str(g if p else 2),metric='Recall@5') for p in [0.,.1,.2,.3,.4,.5]];ax.plot([0,.1,.2,.3,.4,.5],[100*float(r['value']) for r in rr],marker='o',color=col,label=label)
 ax.set(xlabel='w_p',ylabel='Recall@5 (%)',title=f'A, γ={g}');ax.grid(alpha=.2);ax.legend(fontsize=7)
f.suptitle('Giả định oracle — validation: 1.597 gallery / 568 ảnh hỏi');f.text(.02,.015,'CI bỏ trên hình để dễ đọc, có trong sensitivity_summary.csv. σ=∞ chỉ là nhãn trường hợp độc lập.\nOracle dùng đáp án có chủ đích; không dùng để chọn cấu hình, không phải bằng chứng người dùng thật.',fontsize=8);f.tight_layout(rect=[0,.09,1,.93]);f.savefig(fig/'03_do_nhay_sigma.png');plt.close(f)
# Main report.
s=['# Kết quả tuần 06 — visual và popularity giả lập','',
'## 1. Đã làm gì','',
'Giữ A=B2-v1 làm baseline chính (mapping mask sai ngữ nghĩa), B=B2-v2 đối chiếu; G2 ảnh gốc trong bbox C chỉ tham chiếu và vẫn cần mask. Chỉ khảo sát visual + popularity, không category, không đổi split/backbone/seed. Validation: **1.597 gallery / 568 ảnh hỏi**; test: **1.543 gallery / 586 ảnh hỏi**. Mọi metric dưới đây có điều kiện trên các tập này. Popularity và t là giả lập, không kết luận lượt mua hay sở thích.',
'', '## 2. Protocol bổ sung và kiểm tra lỗi','',
'Protocol bất biến: `protocol_week6.md`, SHA-256 `'+(O/'protocol_week6.sha256').read_text().split()[0]+'`. Quy tắc người dùng chuyển wc sang wv, giữ wp/gamma; gốc chỉ lưu lịch sử. P (.7,.3,0), γ=2; tuần3 đã bỏ category (.7,.3,0), γ=4; §4.5 đã bỏ category (.8,.2,0), γ=2 theo mặc định §4.2. Cả ba trùng lưới, không chạy hai lần. Paper không công bố wc số ở §4.5; không khẳng định wc=.1 là số nguồn. Nguồn và dòng: `paper_source_week6.md`, bảng protocol.',
'Cosine không rescale; xếp giảm dần stable theo gallery, giữ các góc ảnh riêng. MAE@K gộp sản phẩm duy nhất trong Top-K ảnh rồi macro query, không bù thêm sản phẩm. CI95% bootstrap theo query; hiệu ghép cặp dùng cùng resample. Top1 là Hit1, Recall1 có mẫu số số ảnh đáp án.',
'Kiểm tra wp0 tái tạo chính xác sáu cột từng dòng Top10 và metric/CI nguồn: `baseline_reproduction_valid.json`, `baseline_reproduction_test.json`. G2 không có MAE lịch sử nên MAE G2 chỉ được kiểm độc lập. Mã độc lập kiểm toàn bộ query ở ba điểm/base validation, mọi cấu hình A test; metric/CI/hiệu được kiểm toàn bộ cấu hình chính thức. Oracle và mức tham chiếu được kiểm riêng. Không có tăng Recall rõ rệt theo tiêu chí dừng đặt trước (CI hiệu dương hoàn toàn so wp0 hoặc bước liền kề); xem `monotonicity_valid.csv`. Không coi gần đơn điệu là bằng chứng độc lập tuyệt đối.',
'', '## 3. Lưới validation','',
'**Validation: 1.597 gallery / 568 ảnh hỏi.** Bảng tỉ lệ 0–1, CI95%. Toàn bộ metric ở metrics_valid.csv; γ=4 tại wp0 là alias γ=2.',
'', '| Base | wp | γ | Recall@5 [CI] | MAE@10 [CI] |','|---|---:|---:|---|---|']
for base in ['A','B','G2']:
 for p,g in [(0.,2)]+[(p/10,g) for p in range(1,6) for g in [2,4]]:
  k=dict(base=base,wp=str(p),gamma=str(g));s.append(f'| {base} | {p} | {g} | {cell(find(valid,metric="Recall@5",**k))} | {cell(find(valid,metric="MAE@10",**k))} |')
s+=['','![Đường đánh đổi validation](figures/01_danh_doi_validation.png)','',
'**Thang cosine validation: 1.597 gallery / 568 ảnh hỏi**, lấy ứng viên visual-only. SD popularity trên toàn cặp query–gallery; chi tiết số hạng nhân wp trong scale_valid.csv. Thang khác nhau nên cùng wp không bảo đảm cùng mức tác động giữa các base.',
'','| Base | Cos Top1 min / median / max | Cos Top10 min / median / max | SD popularity γ2 / γ4 |','|---|---|---|---|']
for base in ['A','B','G2']:
 parts=[]
 for k in [1,10]:
  r=find(scale,base=base,quantity=f'cos_visual_top{k}');parts.append(' / '.join(f'{float(r[a]):.5f}' for a in ['min','median','max']))
 stds=[float(find(scale,base=base,quantity=f'pop_term_p3_g{g}')['std']) for g in [2,4]]
 s.append(f'| {base} | {parts[0]} | {parts[1]} | {stds[0]:.5f} / {stds[1]:.5f} |')
s+=['','## 4. Điểm vận hành và độ nhạy ngưỡng','',
'**Validation: 1.597 gallery / 568 ảnh hỏi.** Chọn theo ngưỡng cố định 2 điểm phần trăm; ngưỡng 1/3 chỉ mô tả. Trong các điểm đạt, ưu tiên wp lớn nhất, rồi MAE10 thấp hơn, rồi γ2. Không có điểm nghĩa là không có wp>0 đạt, không tự gọi wp0 là điểm vận hành.',
'', '| Base | Ngưỡng 1 điểm | Ngưỡng 2 điểm (chính) | Ngưỡng 3 điểm |','|---|---|---|---|']
for base in ['A','B','G2']:
 cells=[]
 for th in [1.,2.,3.]:
  r=find(op,base=base,threshold_pp=str(th));cells.append(f"wp={r['wp']}, γ={r['gamma']}" if r['has_operating_point']=='True' else 'không có điểm vận hành')
 s.append('| '+base+' | '+' | '.join(cells)+' |')
s+=['','## 5. Test xác nhận sau freeze','',
'**Test: 1.543 gallery / 586 ảnh hỏi.** Test chạy một lượt cho các cấu hình đóng băng; không dùng test để chọn. Test đã được quan sát ở tuần trước nên không phải một holdout chưa từng thấy. A không có điểm vận hành nên chỉ hai cấu hình test A. B và G2 xem metrics_test.csv.',
'','| A | Top1 [CI] | Recall@5 [CI] | NDCG@5 [CI] | MRR@10 [CI] | MAE@5 [CI] | MAE@10 [CI] |','|---|---|---|---|---|---|---|']
for p,label in [(0.,'Visual-only'),(.3,'P của paper, đã bỏ category')]:s.append('| '+label+' | '+' | '.join(cell(find(test,base='A',wp=str(p),gamma='2',metric=m)) for m in ['Top-1','Recall@5','NDCG@5','MRR@10','MAE@5','MAE@10'])+' |')
s+=['| Điểm vận hành | không có | — | — | — | — | — |','',
'**Hiệu P−visual A, test 1.543 gallery / 586 ảnh hỏi**, CI95% ghép cặp:',
'','| Metric | Hiệu [CI] |','|---|---|']
for m in ['Recall@5','NDCG@5','MRR@10','MAE@5','MAE@10']:s.append(f'| {m} | {delta(find(pairs,base="A",wp="0.3",gamma="2",metric=m))} |')
s+=['','![Test A](figures/02_test_A.png)','',
'**Mức tham chiếu MAE** (không phải cấu hình được chọn). Nền sản phẩm chính tính cả sản phẩm query; nền theo ảnh và bản loại sản phẩm query được báo để tránh lẫn mẫu số. “Tối ưu đủ 10 sản phẩm” là ràng buộc khác MAE Top10 ảnh; bản loại query còn loại ứng viên hợp lệ. Chặn dưới tuyệt đối là lỗi nhỏ nhất trên một sản phẩm khả dụng.',
'','| Tập (gallery / ảnh hỏi) | Tham chiếu | MAE [CI] |','|---|---|---|']
for r in refs:s.append(f"| {r['split']} ({r['gallery']} / {r['queries']}) | {r['reference']} | {cell(r)} |")
s+=['','## 6. Độ nhạy theo giả định oracle','',
'**Chỉ A, validation 1.597 gallery / 568 ảnh hỏi**. Dùng pop đáp án cộng nhiễu hash theo query và σ rồi clip: oracle có chủ đích, không dùng để chọn hoặc chạy test. σ=∞ chỉ là tên trường hợp t chính thức độc lập, không phải giới hạn Gaussian sau clip. Bảng bên dưới giữ điểm P đã định trước để đặt các giả định cạnh nhau, không chọn điểm tốt nhất từ oracle.',
'','| Trường hợp (wp=.3, γ2) | Recall@5 [CI] | Hiệu Recall@5 [CI] | MAE@10 [CI] |','|---|---|---|']
for scenario in ['official','oracle_sigma0.3','oracle_sigma0.1']:
 r=find(sens,scenario=scenario,wp='0.3',gamma='2',metric='Recall@5');m=find(sens,scenario=scenario,wp='0.3',gamma='2',metric='MAE@10');s.append(f'| {scenario} | {cell(r)} | {float(r["difference"]):+.5f} [{float(r["delta_low"]):+.5f}; {float(r["delta_high"]):+.5f}] | {cell(m)} |')
s+=['','Dưới giả định σ=.1, CI hiệu Recall5 dương ở mọi wp>0 của lưới. Dưới giả định σ=.3, γ4 mọi wp>0 và γ2 tại wp=.1/.2/.3 có CI chứa 0 hoặc dương; γ2 tại wp=.4/.5 có CI âm. CI chứa0 chỉ là chưa phát hiện giảm, không chứng minh tương đương/không giảm. Đây là phân tích giả định, không gọi là “popularity giúp ích”. Toàn bộ MAE5/10 và hiệu/CI ở sensitivity_summary.csv; kiểm độc lập ở independent_oracle_checks.csv.',
'','![Độ nhạy oracle](figures/03_do_nhay_sigma.png)','',
'## 7. Phản biện và giới hạn','',
'Đường đánh đổi quan sát được phù hợp kỳ vọng từ popularity/t giả lập độc lập: thêm thông tin về độ dốc và mức chịu đựng của luật vận hành, không xác nhận cơ chế người dùng. Oracle cho thấy kết luận thay đổi theo giả định liên hệ t–đáp án. G2 cần mask, không phải pipeline không mask. B0 so ngẫu nhiên chỉ mô tả, không chạy B0 mới ở tuần6.',
'CI chỉ phản ánh lấy mẫu query có điều kiện trên một gallery/split/seed; không bao phủ biến thiên người dùng hoặc dữ liệu khác. Có nhiều phép so sánh, xem mcnemar_bonferroni.csv và paired_bonferroni.csv; CI Bonferroni dùng đuôi bootstrap rất thưa nên không dựa vào chúng cho khẳng định mạnh. Luật 2 điểm là lựa chọn chủ quan, kết quả A/B đổi khi nới ngưỡng. Xem phan_bien_tuan_06.md và phản biện độc lập kèm xử lý.',
'Chưa kiểm tra được: nhu cầu/lượt mua thật, category chuẩn do người xác nhận, nhiều split/seed/gallery, cơ chế mask, peak RAM toàn đợt. Không suy từ MAE sang sở thích. Không tái lập đầy đủ thí nghiệm paper (dữ liệu, category và mô phỏng khác).',
'', '## 8. Minh chứng và tái chạy','',
'Xem manifest_tuan06.json, doi_chieu_so_lieu_tuan06.csv, verification_tuan06.txt, commands_tuan06.txt và muc_luc_goi_tuan06.md. Mã nguồn chỉ nằm tuần6; không sửa tuần1–5/5b/5c. Không chạy lại pipeline vào thư mục bàn giao: các bước dùng tạo file độc quyền, test có marker ngăn lần chạy thứ hai. Khi tái lập, dùng workspace bản sao giữ cấu trúc và dữ liệu nguồn đúng hash, thư mục output mới. Bootstrap kiểm chứng lại không phải một lần chọn/test mới.',
'Khác/diễn giải prompt: loại trùng gamma tại wp0 và điểm có tên; bổ sung bản tham chiếu theo ảnh/sản phẩm và chặn dưới đúng cho metric thực; §4.5 gamma mặc định được ghi trước khi chạy. Không tạo cấu hình vận hành thay thế khi luật không có điểm. Mâu thuẫn tuần3 đã được người dùng chốt, không còn treo.', '']
write('ket_qua_tuan_06.md','\n'.join(s))
# Reviewer response, concrete citations and reproducible evidence.
timeline=[]
for n,k in [('user_decision_week6.json','recorded_at_utc'),('validation_started_week6.json','timestamp'),('completed_valid.json','timestamp'),('independent_valid_passed.json','timestamp'),('completed_oracle.json','timestamp'),('freeze_week6.json','timestamp'),('test_started_week6.json','timestamp'),('completed_test.json','timestamp'),('independent_test_passed.json','timestamp')]:timeline.append({'file':n,'timestamp_utc':js(n)[k],'mtime_utc':datetime.fromtimestamp((O/n).stat().st_mtime,timezone.utc).isoformat()})
timeline.insert(1,{'file':'protocol_week6.md','timestamp_utc':datetime.fromtimestamp((O/'protocol_week6.md').stat().st_mtime,timezone.utc).isoformat(),'mtime_utc':datetime.fromtimestamp((O/'protocol_week6.md').stat().st_mtime,timezone.utc).isoformat()})
csvwrite('timeline_week6.csv',timeline)
assert timeline[0]['timestamp_utc']<timeline[1]['timestamp_utc']<js('validation_started_week6.json')['timestamp']<js('freeze_week6.json')['timestamp']<js('test_started_week6.json')['timestamp']<js('completed_test.json')['timestamp']
m=int(mc[0]['family_comparisons']);survive=[r for r in mc if r['significant_bonferroni']=='True'];a_mc=find(mc,split='test',base='A',wp='0.3',gamma='2')
r=['# Tự phản biện tuần 06','', 'Các số retrieval validation ứng với G=1.597/Q=568; test G=1.543/Q=586. Các phát hiện không là bằng chứng nhân quả.', '',
'1. **Đã ghi vào giới hạn — gần biết trước.** Chiều giảm Recall/tăng phù hợp t là kỳ vọng định trước; đóng góp là đo đường và luật chịu mất Recall. operating_points.csv: A/G2 không có điểm ở ngưỡng2, B wp=.1 γ4 (validation G1597/Q568). Không nói đây là thuật toán mới hoặc tính hữu ích thương mại.',
'2. **Đã ghi vào giới hạn — rò rỉ.** leakage_diagnostics_valid.json tái tạo chính xác popularity RNG theo metadata; t_q kiểm hash mọi query. Pearson t với pop đáp án = '+f"{leak['t_query_pop_pearson']:.8f}"+' (validation G1597/Q568), chỉ mô tả. Ranking official dùng cos/t/pop; đáp án chỉ chấm metric. Oracle lưu riêng và freeze chỉ cấu hình official; không lấy oracle để chọn. Độc lập do thiết kế không chứng minh không có tương quan ngẫu nhiên trong mẫu.',
'3. **Đã ghi vào giới hạn — thang cos.** scale_valid.csv và bảng mục3 báo min/median/max Top1/10 và SD popularity (validation G1597/Q568). Không chuẩn hóa lại và không so wp như cường độ tương đương giữa base. Khác thang có thể góp vào chênh độ dốc; không quy cho mask.',
'4. **Không áp dụng cho cáo buộc chọn trên test tuần6 — có bằng chứng thứ tự.** timeline_week6.csv ghi quyết định → protocol → validation → freeze → test. Test có marker độc quyền; configs đóng băng chỉ dựa operating_points.csv. Đã ghi giới hạn rằng test các tuần trước đã thấy. Timestamp ghi nhận quyết định không phải timestamp máy chủ tin nhắn.',
'', '| File | Timestamp UTC nội dung (protocol dùng mtime) | mtime UTC |','|---|---|---|']
for t in timeline:r.append('| '+' | '.join(t.values())+' |')
r+=['',f'5. **Đã ghi vào giới hạn — đa kiểm định.** Họ chính {m} tương phản (validation G1597/Q568: 30; test G1543/Q586: 4), McNemar Bonferroni .05/{m}; {len(survive)} còn dưới .05. Họ CI hiệu gồm {m*5} cặp metric; cùng bootstrap chỉ kỳ vọng {10000*.025/(m*5):.3f} mẫu ở mỗi đuôi nên không coi CI điều chỉnh này là chính xác. Oracle tách riêng, mô tả, không dùng để tuyên bố khám phá. A test P: p exact={float(a_mc["p_exact"]):.8g}, p Bonferroni={float(a_mc["p_bonferroni"]):.8g} (G1543/Q586). Mọi CI chính là chưa điều chỉnh; dùng paired, không suy từ CI hai trung bình.',
'6. **Đã ghi vào giới hạn — biến thiên thật.** Một query mỗi sản phẩm nhưng gallery cố định; bootstrap không gồm biến thiên split/gallery/seed popularity hay con người. Chỉ proxy có điều kiện.',
'7. **Đã ghi vào giới hạn — ngưỡng.** operating_points.csv (validation G1597/Q568): A không có ở1/2, có wp=.1 γ4 ở3; B không có ở1, wp=.1 γ4 ở2, wp=.1 γ2 ở3; G2 không có cả1/2/3. Luật chọn nhạy, không tuyên bố tối ưu ổn định.',
'8. **Đã ghi vào giới hạn — câu dễ vượt bằng chứng.** Không dùng “popularity giúp ích”, thay bằng “dưới giả định oracle”; không dùng “không giảm” khi CI chứa0, thay “chưa phát hiện giảm”; không dùng “G2 không dùng mask”, thay “ảnh gốc trong bbox C cần mask”; không dùng “chặn dưới Top10 ảnh” cho tối ưu đủ10 sản phẩm, đã phân biệt ngay trong protocol. Đây là lựa chọn diễn đạt trước bản báo cáo, không có kết quả cũ bị sửa rồi che đi.',
'9. **Đã ghi vào giới hạn — giải thích thay thế.** Validation G1597/Q568: '+f"{leak['gallery_products']} sản phẩm gallery, {leak['unique_normpop_values']} giá trị NormPop khác nhau, nhóm đồng NormPop lớn nhất {leak['largest_pop_tie_products']} sản phẩm; mean={leak['normpop_mean']:.6f}, SD={leak['normpop_std']:.6f}."+' Gallery hữu hạn, số ảnh/sản phẩm khác nhau, lượng tử popularity và margin cosine có thể tạo đoạn phẳng/giảm gấp. Nền theo ảnh khác nền theo sản phẩm (reference_valid.csv); không quy đường cho một cơ chế.',
'10. **Đã ghi vào giới hạn — điều làm đổi chiều.** Kiểm tra rẻ đã định trước là oracle A, không thêm ablation: σ=.1 có CI hiệu Recall5 dương toàn lưới wp>0; σ=.3 phụ thuộc gamma/wp (validation G1597/Q568; sensitivity_summary.csv). Giả định phụ thuộc đáp án có thể đổi chiều; không mang ý nghĩa người dùng thật.',
'11. **Đã ghi vào giới hạn — G2 cần mask.** Bảng vai trò, protocol, báo cáo và caption hình ghi bbox C từ mask; không có kết luận G2 là pipeline không mask.',
'', '## Phiên bản và xử lý sửa lỗi','',
'Bản v1 của số liệu hoàn tất, chưa có sửa số liệu sau test. Mọi sửa do phản biện độc lập phải ghi file trước/sau và chạy lại kiểm chứng liên quan, không chạy lại hoặc chọn lại test. Phản biện độc lập được thực hiện trên hồ sơ tuần6, không có lịch sử hội thoại.', '']
write('phan_bien_tuan_06.md','\n'.join(r))
print('report and figures written')
