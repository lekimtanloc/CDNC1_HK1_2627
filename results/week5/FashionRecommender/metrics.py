import math

def score(ranked, relevant):
 if len(ranked)!=len(set(ranked)): raise ValueError('Duplicate image')
 hits=[int(x in relevant) for x in ranked]; n=len(relevant)
 out={'Top-1':float(bool(hits and hits[0]))}
 for k in (1,5,10):out[f'Recall@{k}']=sum(hits[:k])/n if n else 0.
 for k in (5,10):
  dcg=sum(h/math.log2(i+2) for i,h in enumerate(hits[:k]))
  ideal=sum(1/math.log2(i+2) for i in range(min(k,n)))
  out[f'NDCG@{k}']=dcg/ideal if ideal else 0.
 out['MRR@10']=next((1/i for i,h in enumerate(hits[:10],1) if h),0.)
 return out
