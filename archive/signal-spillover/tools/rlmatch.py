import sys
sys.path.insert(0, '/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code')
import pandas as pd
from sig_func import effect
pair, ref, others = sys.argv[1], sys.argv[2], sys.argv[3:]
md = 'ref_findata' if pair.endswith('_fin') else 'ref_meddata'
r = effect(pair, ref)
def sub(v, s): return v[v.index.get_level_values('set') == s]
print(ref, 'MD %.1f EM %.1f' % (sub(r, md).mean(), sub(r, 'ref_cartoon120').mean()))
for t in others:
    v = effect(pair, t)
    print(f"{t:45s} MD {sub(v, md).mean():6.1f} EM {sub(v, 'ref_cartoon120').mean():6.1f}  r_EM {sub(v, 'ref_cartoon120').corr(sub(r, 'ref_cartoon120')):.2f}  r_MD {sub(v, md).corr(sub(r, md)):.2f}")
