import os
os.environ['MPLCONFIGDIR']='/private/tmp/week5_mpl';os.environ['XDG_CACHE_HOME']='/private/tmp/week5_cache'
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_test_day_du import O,read,configs
plt.rcParams.update({'font.family':'Arial','font.size':10})
F=O/'figures';data={'valid':read(O/'metrics_valid.csv'),'test':read(O/'metrics_test_day_du.csv')};pair=json.loads((O/'paired_day_du.json').read_text());refs={r['split']:r for r in read(O/'reference_levels_day_du.csv')}
footer='B2-v1*: bộ lọc KHÔNG đúng ngữ nghĩa (giữ 4,9; bỏ 13,15). Chỉ retrieval proxy; test là xác nhận.'
def metric(s,c,m):return next(r for r in data[s] if r['config']==c and r['metric']==m)
def save(fig,name):
 path=F/name;assert not path.exists();fig.text(.5,.015,footer,ha='center',fontsize=8);fig.tight_layout(rect=(0,.055,1,1));fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8.8,4.1))
for ax,m in zip(axs,('Recall@5','NDCG@5')):
 for j,(s,label) in enumerate([('valid','Validation'),('test','Test xác nhận')]):
  rr=[metric(s,c,m) for c in configs];v=np.array([float(r['value']) for r in rr]);lo=np.array([float(r['ci_low']) for r in rr]);hi=np.array([float(r['ci_high']) for r in rr]);ax.bar(np.arange(4)+(j-.5)*.36,v,.36,yerr=[v-lo,hi-v],capsize=3,label=label)
 ax.set_xticks(range(4),['B0','B2p','B2-v1*','B2-v2']);ax.set_ylim(bottom=0);ax.set_title(m+' — CI 95%');ax.set_ylabel('Tỉ lệ');ax.legend(fontsize=8)
save(fig,'hinh_01_metric_validation_test_day_du.png')
fig,ax=plt.subplots(figsize=(8.8,4))
for j,(s,label) in enumerate([('valid','Validation'),('test','Test xác nhận')]):
 t=pair[s]['B2-v2 minus B2-v1']['McNemar']['table_rows_subtrahend_cols_minuend_0_1'];vals=[t[1][0],t[0][1],t[1][1],t[0][0]];bars=ax.bar(np.arange(4)+(j-.5)*.36,vals,.36,label=label);ax.bar_label(bars,padding=3)
ax.set_xticks(range(4),['Đúng → sai','Sai → đúng','Đúng cả hai','Sai cả hai']);ax.set_ylabel('Số ảnh hỏi');ax.set_title('Chuyển trạng thái Top-1: B2-v1* → B2-v2');ax.legend();save(fig,'hinh_02_chuyen_trang_thai_v1_v2_day_du.png')
fig,axs=plt.subplots(1,2,figsize=(8.8,4.1),sharey=True)
for ax,s,label in zip(axs,('valid','test'),('Validation','Test xác nhận')):
 rr=[metric(s,c,'Popularity MAE') for c in configs];v=np.array([float(r['value']) for r in rr]);lo=np.array([float(r['ci_low']) for r in rr]);hi=np.array([float(r['ci_high']) for r in rr]);ax.bar(range(4),v,yerr=[v-lo,hi-v],capsize=3,color=['gray','#9ca9bf','#4676ac','#d08b42']);ax.axhline(float(refs[s]['popularity_mae_reference']),linestyle='--',color='#b22222',label='Mức nền toàn gallery');ax.set_xticks(range(4),['B0','B2p','B2-v1*','B2-v2']);ax.set_title(label);ax.set_ylim(bottom=0);ax.legend(fontsize=8)
axs[0].set_ylabel('Popularity MAE (thấp hơn: gần t giả lập hơn)');save(fig,'hinh_05_popularity_mae_day_du.png')
with (F/'mo_ta_hinh_day_du.md').open('x') as f:f.write('''# Nguồn và caption hình đầy đủ

B2-v1 dùng nhãn 1–12, giữ 4,9 và bỏ 13,15: KHÔNG đúng ngữ nghĩa. Kết quả chỉ là retrieval proxy. Luật chọn đặt sau khi đã thấy sanity test tuần 4, vì vậy test là xác nhận, không phải bằng chứng độc lập.

Hình 1. Recall@5 và NDCG@5 của B0, B2p, B2-v1, B2-v2 trên validation và test, CI 95% percentile bootstrap. Nguồn metrics_valid.csv, metrics_test_day_du.csv. Trục y từ 0; đơn vị tỉ lệ. B2p và v2 là đối chiếu, không thay baseline đã chấp nhận.

Hình 2. Chuyển trạng thái Top-1 v1 sang v2 trên validation và test. Nguồn paired_day_du.json, bảng McNemar. Cột là số query, không diễn giải nguyên nhân.

Hình 3. Giữ nguyên hinh_03_category_cu_moi.png và caption cũ; không vẽ lại.

Hình 4. Top-5 của hai cấu hình cạnh nhau cho 5 query validation v1 đúng/v2 sai và 5 query chiều ngược lại. Nguồn montage_selection_day_du.csv, valid_B2-v1_top10.csv, valid_B2-v2_top10.csv, ảnh gốc archive.zip. Trong từng nhóm lấy 5 query_id đầu theo thứ tự tăng dần, không chọn tay, không ngẫu nhiên. Trái: v1; phải: v2. Mỗi nửa gồm ảnh hỏi và 5 kết quả. ĐÚNG/SAI dựa trên ground truth cùng sản phẩm, không theo category. Không dùng query test để chọn mẫu. Chỉ minh họa, chưa kiểm định nguyên nhân.

Hình 5. Popularity MAE và CI cho bốn cấu hình trên validation/test; đường đứt là lỗi trung bình trên toàn bộ sản phẩm gallery sau khi bỏ sản phẩm query. Nguồn reference_levels_day_du.csv và các bảng metric. Popularity và t đều giả lập; không biểu thị hiệu quả mua hàng. Chưa kiểm định khác biệt so với đường tham chiếu.

Các PNG mới nền trắng, 180 dpi, rộng không quá 1600 pixel. Montage cao hơn biểu đồ để giữ đủ mẫu.
''')
print('Figures created without touching old figures')
