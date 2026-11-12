from prepare import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
figdir=O/'figures';figdir.mkdir(exist_ok=True)
plt.rcParams['font.family']='Arial'
report=json.loads((P/'validation_report.json').read_text());cats=read(O/'category_v2.csv');rows=[]
mapname={'1':'Áo thân trên','2':'Chân váy','5':'Váy liền','8':'Áo khoác','13':'Quần','accessory_or_unknown':'Phụ kiện/không rõ'}
fig,axs=plt.subplots(1,2,figsize=(8.8,4.8))
old=Counter();new=Counter()
for s in ('train','valid','test'):
 check=Counter(r['item_category'] for r in read(P/f'{s}.csv'));assert dict(check)==report['split_stats'][s]['item_category_image_counts']
 old.update(check)
 for c,n in sorted(check.items()):rows.append({'split':s,'version':'old','category':c,'image_count':n})
 cc=Counter()
 for r in cats:
  if r['split']==s:cc[mapname[r['category_v2']]]+=int(r['n_images'])
 new.update(cc)
 for c,n in sorted(cc.items()):rows.append({'split':s,'version':'v2','category':c,'image_count':n})
write(O/'category_distribution_comparison.csv',rows)
for ax,c,title in zip(axs,(old,new),('Category cũ (không đáng tin)','Category v2 (chưa đạt kiểm tra)')):
 ax.barh(list(c),list(c.values()));ax.set_title(title);ax.set_xlabel('Số ảnh (category sản phẩm)');ax.set_xlim(left=0)
fig.tight_layout();fig.savefig(figdir/'hinh_03_category_cu_moi.png',dpi=180,facecolor='white');plt.close(fig)
