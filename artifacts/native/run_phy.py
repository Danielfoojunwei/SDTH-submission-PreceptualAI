import json, time
from wirelessjev.native.phy import layered_study
t=time.time(); out={}
out["awgn"]=layered_study(fading="awgn", snr_db=tuple(range(0,21)), batch=64, batches=8)
out["rayleigh"]=layered_study(fading="rayleigh", snr_db=tuple(range(0,42,2)), batch=64, batches=16)
out["elapsed_s"]=time.time()-t
open("artifacts/native/phy_layered.json","w").write(json.dumps(out,indent=1))
print(json.dumps({k:v["snr_for_1pct_bler"] for k,v in out.items() if isinstance(v,dict)},indent=1))
