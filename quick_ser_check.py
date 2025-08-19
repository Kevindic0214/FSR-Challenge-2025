# quick_ser_check.py
import json
from pathlib import Path

def ser(preds, refs):
    def ed(a,b):
        dp=[[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(len(a)+1): dp[i][0]=i
        for j in range(len(b)+1): dp[0][j]=j
        for i in range(1,len(a)+1):
            for j in range(1,len(b)+1):
                dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(a[i-1]!=b[j-1]))
        return dp[-1][-1]
    e=r=0
    for h,g in zip(preds,refs):
        H=h.split(); G=g.split(); e+=ed(H,G); r+=len(G)
    return 100.0*e/max(1,r)

jsonl = Path("HAT-Vol2/manifests/dev.jsonl")
tsv   = Path("decode_track2.tsv")

ref = {}
for l in jsonl.read_text(encoding="utf-8").splitlines():
    ex=json.loads(l); ref[ex.get("utt_id", Path(ex["audio"]).stem)] = ex["text"].strip()

pred=[]; gold=[]
for l in tsv.read_text(encoding="utf-8").splitlines()[1:]:
    utt, txt, _ = l.rstrip("\n").split("\t", 2)
    if utt in ref:
        pred.append(txt.strip()); gold.append(ref[utt])

print(f"DEV SER = {ser(pred, gold):.2f}%  (N={len(pred)})")
