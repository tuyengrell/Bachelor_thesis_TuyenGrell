"""
Cued AAT - Trial alignment diagnostic

Compare ET dwell side with behavioral cueSide for offsets 0–20.
Report aggregate match percentages and the best offset per column.
A dominant offset across columns suggests a consistent trial shift.
Read-only: no input files are modified and no results are saved.
"""
import numpy as np
import scipy.io as sio
from collections import Counter
from pathlib import Path

# === Settings ===
CT = Path(__file__).resolve().parents[2] / "CuedAAT"

SCREEN_CENTER_PX=960; CROSS_ROI_HALF=100; LEAVE_SAMPLES=50; VEL_THRESHOLD=5
MIN_PIC_STABLE=20; MAX_VAR_PX=50; POST_ONSET_MS=3000
MAX_OFFSET = 20

# === Load MATLAB arrays and cell arrays ===
def load_mat(name):
    d=sio.loadmat(str(CT/f"{name}.mat")); 
    k=[x for x in d if not x.startswith("_")][0]; 
    return d[k]

def load_cell(name):
    d=sio.loadmat(str(CT/f"{name}.mat"),
                  squeeze_me=False,
                  struct_as_record=False)
    k=[x for x in d if not x.startswith("_")][0]; 
    return d[k]

# === Estimate fixation center before picture onset ===
def find_center(x):
    n=len(x)
    if n<5: return float(SCREEN_CENTER_PX)
    roi=(~np.isnan(x))&(np.abs(x-SCREEN_CENTER_PX)<=CROSS_ROI_HALF)
    
    # Fall back to the median gaze position if no samples lie in the central ROI
    if not roi.any():
        v=x[~np.isnan(x)]
        if len(v)==0: return float(SCREEN_CENTER_PX)
        fb=float(np.median(v)); r2=(~np.isnan(x))&(np.abs(x-fb)<=CROSS_ROI_HALF)
        return float(np.nanmedian(x[r2])) if r2.any() else fb
    
    # Find the longest ROI period, allowing brief excursions outside the band
    bs=be=0;bl=0;cur=None;out=0
    
    for i in range(n):
        if roi[i]:
            if cur is None: cur=i
            out=0
        else:
            if cur is not None:
                out+=1
                if out>LEAVE_SAMPLES:
                    e=i-out
                    if e-cur>bl: bl=e-cur; bs,be=cur,e
                    cur=None;out=0
    if cur is not None:
        e=n-1
        if e-cur+1>bl: bs,be=cur,e
    return float(np.nanmedian(x[bs:be+1]))

# === Detect the first qualifying lateral dwell after picture onset ===
def dwell_side(x,t,onset,ctr):
    m=(t>=onset)&(t<onset+POST_ONSET_MS)
    if m.sum()<MIN_PIC_STABLE+2: 
        return None
    
    xw=x[m]
    n=len(xw)
    
    for i in range(n-MIN_PIC_STABLE-1):
        if np.isnan(xw[i]) or np.isnan(xw[i+1]): 
            continue
        
        if abs(xw[i+1]-xw[i])>VEL_THRESHOLD:
            px=xw[i+1]
            if px<ctr-CROSS_ROI_HALF: side="left"
            elif px>ctr+CROSS_ROI_HALF: side="right"
            else: continue
            
            # Require sufficient valid samples and limited gaze variation.
            seg=xw[i+1:i+1+MIN_PIC_STABLE];v=seg[~np.isnan(seg)]
            if len(v)>=int(MIN_PIC_STABLE*0.8) and np.max(v)-np.min(v)<=MAX_VAR_PX:
                return side
    return None

# === Load behavioral and gaze data ===
cue=load_mat("cueSide")
xl=load_cell("xPositionLeftContinuos"); xr=load_cell("xPositionRightContinuos")
tv=load_cell("timeVectorContinuos"); pic=load_cell("pictureTimeIdx"); fix=load_cell("fixationTimeIdx")
NC=xl.shape[1]

# === Extract ET dwell side for each trial and column ===
et_side=[]
for col in range(NC):
    # Exclude off-screen gaze positions and combine available eye signals
    _xl=xl[0,col].flatten().astype(float); _xr=xr[0,col].flatten().astype(float)
    _xl=np.where((_xl<0)|(_xl>1920),np.nan,_xl); _xr=np.where((_xr<0)|(_xr>1920),np.nan,_xr)
    xa=np.where(~np.isnan(_xl)&~np.isnan(_xr),(_xl+_xr)/2,np.where(~np.isnan(_xl),_xl,_xr))
    t=tv[0,col].flatten().astype(float); p=pic[0,col].flatten().astype(float); f=fix[0,col].flatten().astype(float)
    sides=[]
    for k in range(len(p)):
        pi=int(p[k]); fi=int(f[k]) if k<len(f) else -1
        if pi>=len(t) or fi<0 or fi>=len(t): sides.append(None); continue
        ctr=find_center(xa[(t>=t[fi])&(t<t[pi])])
        sides.append(dwell_side(xa,t,t[pi],ctr))
    et_side.append(sides)

# === Compare ET dwell side with behavioral cue side at a given offset ===
def col_match(col, off):
    sides=et_side[col]; m=tot=0
    
    # Compare behavioral trials 5–88 with ET trials shifted by the offset
    for k in range(4,88):
        exp="left" if int(cue[k,col])==0 else "right"
        j=k+off
        if j<len(sides) and sides[j] in ("left","right"):
            tot+=1
            if sides[j]==exp: m+=1
    return m,tot

# === Report aggregate match percentages across offsets ===
print("Aggregate match% by offset:")

for off in range(0,MAX_OFFSET+1):
    m=tot=0
    for col in range(NC):
        a,b=col_match(col,off); m+=a; tot+=b
    print(f"  offset {off:2d}: {100*m/tot if tot else 0:5.1f}%  (n={tot})")

# === Select the best offset per column and count votes ===
votes=Counter()
for col in range(NC):
    best=(-1,-1.0)
    for off in range(0,MAX_OFFSET+1):
        a,b=col_match(col,off)
        pct=100*a/b if b else 0
        # Require at least 40 matched pairs; ties retain the smaller offset
        if b>=40 and pct>best[1]: best=(off,pct)
    # An offset of -1 indicates that no offset met the minimum pair count
    votes[best[0]]+=1
print("\nPer-column best offset (votes across 146 columns):")
for off,c in sorted(votes.items(), key=lambda kv:-kv[1]):
    print(f"  offset {off:2d}: {c} columns")
    
# === Report the dominant offset and interpretation ===
top=votes.most_common(1)[0]
print(f"\n=> Dominant offset = {top[0]} ({top[1]}/{NC} columns).")
