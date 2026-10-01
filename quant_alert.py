"""quant_alert.py — compact research/paper-trading engine.
Closed Binance candles -> robust Z-score -> features -> HMM/GARCH/
sentiment/correlation -> confluence -> volatility barriers -> risk sizing
-> FFC -> paper trader -> Telegram. Live trading is intentionally disabled.
"""
from __future__ import annotations
import html, inspect, logging, math, os, time
from typing import Any
import numpy as np
import pandas as pd
import requests

# ---------- config ----------
def B(k, d):
    v=os.getenv(k)
    return d if v is None else v.strip().lower() in {"1","true","yes","on","y"}
def I(k,d,m=0):
    try:v=int(os.getenv(k,d))
    except: raise ValueError(f"{k} must be integer")
    return max(v,m)
def F(k,d,m=None):
    try:v=float(os.getenv(k,d))
    except: raise ValueError(f"{k} must be number")
    if not math.isfinite(v) or (m is not None and v<m): raise ValueError(f"bad {k}={v}")
    return v

TOKEN=os.getenv("TG_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
CHAT=os.getenv("TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
SYMBOLS=[x.strip().upper() for x in os.getenv("SYMBOLS","BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT").split(",") if x.strip()]
INTERVAL=os.getenv("INTERVAL","15m"); LOOKBACK=I("LOOKBACK",40,10); LIMIT=I("CANDLE_LIMIT",250,LOOKBACK+20)
FAST=I("FAST_POLL_SECONDS",300,15); FULL=I("POLL_SECONDS",1800,FAST); REPORT=I("REPORT_INTERVAL_SECONDS",86400,60)
ZTH=F("Z_THRESHOLD",2.0,.1); ZSTR=F("Z_STRONG_THRESHOLD",2.75,ZTH); ROBUST=B("ROBUST_ZSCORE",True)
HMM_ON=B("USE_HMM_FILTER",True); HMM_INT=os.getenv("HMM_INTERVAL","4h"); HMM_LIMIT=I("HMM_LIMIT",300,50); HMM_STATES=I("HMM_STATES",2,2)
GARCH_ON=B("USE_GARCH_FILTER",True); GARCH_INT=os.getenv("GARCH_INTERVAL","4h"); GARCH_LIMIT=I("GARCH_LIMIT",500,50); MAX_VOL_REGIME=os.getenv("GARCH_MAX_ENTRY_REGIME","HIGH").upper()
SENT_ON=B("USE_SENTIMENT_FILTER",True); CORR_ON=B("USE_CORRELATION_FILTER",True); META_ON=B("USE_META_LABEL_FILTER",False); META_MIN=F("META_LABEL_MIN_PROB",.60,0)
SCORE_TH=F("CONFLUENCE_THRESHOLD",68,0); RSI_MAX=F("RSI_BUY_MAX",45,0); BB_MAX=F("BB_PERCENT_B_MAX",.20,-10); EFF_MAX=F("EFFICIENCY_MAX_FOR_MEAN_REVERSION",.75,0)
RISK=F("RISK_PER_TRADE",.005,.0001); MAX_POS=F("MAX_POSITION_PCT",.20,.001); SLV=F("STOP_LOSS_VOL_MULT",1.0,.1); TPV=F("TAKE_PROFIT_VOL_MULT",2.0,.1); MAX_BARS=I("MAX_HOLDING_BARS",20,2)
FEE=F("FEE_RATE",.001,0); SLIP=F("SLIPPAGE_RATE",.0005,0); FAIL_CLOSED=B("FAIL_CLOSED_ON_FILTER_ERROR",True); PAPER_DURING_HALT=B("PAPER_CONTINUE_WHEN_FFC_HALTED",True)
SIZE_FLOOR=F("SCORE_SIZE_FLOOR",.70,.1); SENT_FLOOR=F("SENTIMENT_SIZE_FLOOR",.50,0)
API="https://data-api.binance.vision/api/v3/klines"; S=requests.Session(); S.headers["User-Agent"]="compact-quant-alert/5"
LOG=logging.getLogger("quant_alert"); logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- optional project modules ----------
MOD={}
def M(name):
    try:
        mod=__import__(name); MOD[name]=mod; return mod
    except Exception as e: MOD[name]=None; LOG.warning("%s unavailable: %s",name,e); return None
paper=M("paper_trader"); hmm=M("hmm_regime") if HMM_ON else None; garch=M("garch_model") if GARCH_ON else None
corr=M("correlation_filter") if CORR_ON else None; sent=M("sentiment_filter") if SENT_ON else None; feat=M("feature_engineer")
riskmod=M("risk_manager"); ffcmod=M("ffc"); dsrmod=M("deflated_sharpe")
ffc=ffcmod.get_ffc() if ffcmod and callable(getattr(ffcmod,"get_ffc",None)) else None
corr_obj=corr.get_correlation_filter() if corr and callable(getattr(corr,"get_correlation_filter",None)) else None
sent_obj=sent.get_sentiment_filter() if sent and callable(getattr(sent,"get_sentiment_filter",None)) else None

# ---------- helpers ----------
def call(fn, vals):
    try:
        p=inspect.signature(fn).parameters; kw={k:v for k,v in vals.items() if k in p or any(x.kind==inspect.Parameter.VAR_KEYWORD for x in p.values())}
        return fn(**kw) if kw else fn(*list(vals.values())[:len([x for x in p.values() if x.default is inspect.Parameter.empty])])
    except (TypeError,ValueError): return fn(**vals)
def sf(x,d=0.):
    try:x=float(x); return x if math.isfinite(x) else d
    except:return d
def esc(x):return html.escape(str(x),quote=False)
def price(x):
    x=sf(x); return f"{x:,.2f}" if x>=1000 else f"{x:.4f}" if x>=1 else f"{x:.8f}"
def fn(mod,names):
    if not mod:return None
    for n in names:
        f=getattr(mod,n,None)
        if callable(f): return f

def tg(msg):
    if not TOKEN or not CHAT:return False
    try:
        r=S.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",json={"chat_id":CHAT,"text":msg,"parse_mode":"HTML"},timeout=15); r.raise_for_status(); return True
    except Exception as e: LOG.warning("Telegram: %s",e); return False

# ---------- data ----------
def candles(symbol,interval=INTERVAL,limit=LIMIT):
    r=S.get(API,params={"symbol":symbol,"interval":interval,"limit":limit},timeout=15); r.raise_for_status(); now=int(time.time()*1000); out=[]
    for k in r.json():
        try:
            if int(k[6])>now: continue
            o,h,l,c,v=map(float,k[1:6])
            if min(o,h,l,c)<=0 or v<0 or h<max(o,c) or l>min(o,c): continue
            out.append([o,h,l,c,v,int(k[6])])
        except: pass
    a=np.asarray(out,float)
    if len(a)<max(LOOKBACK,100): raise ValueError(f"{symbol}: insufficient closed candles ({len(a)})")
    if not np.isfinite(a).all(): raise ValueError(f"{symbol}: non-finite candles")
    return a

def fetch_candles(symbol, interval=INTERVAL, limit=LIMIT): return candles(symbol, interval, limit)
def z_score(closes, period=LOOKBACK):
    x=np.log(np.asarray(closes,dtype=float)[-period:]); center=np.median(x) if ROBUST else np.mean(x); scale=1.4826*np.median(np.abs(x-center)) if ROBUST else np.std(x,ddof=1)
    return 0. if scale<=1e-12 else float((x[-1]-center)/scale)
def classify_signal(z): return "BUY_STATISTICAL" if z<-ZTH else "SELL_STATISTICAL" if z>ZTH else "NO_SIGNAL"
def garman_klass_volatility(a, period=LOOKBACK):
    a=np.asarray(a,dtype=float)[-period:]; o,h,l,c=a[:,0],a[:,1],a[:,2],a[:,3]; q=.5*np.log(h/l)**2-(2*np.log(2)-1)*np.log(c/o)**2
    return float(np.sqrt(max(np.nanmean(q),0)))

def zscore(c):
    x=np.log(c[-LOOKBACK:]); center=np.median(x) if ROBUST else np.mean(x); scale=1.4826*np.median(np.abs(x-center)) if ROBUST else np.std(x,ddof=1)
    return 0. if scale<=1e-12 else float((x[-1]-center)/scale)
def gk(a):
    a=a[-LOOKBACK:]; o,h,l,c=a[:,0],a[:,1],a[:,2],a[:,3]
    v=.5*np.log(h/l)**2-(2*np.log(2)-1)*np.log(c/o)**2
    return float(np.sqrt(max(np.nanmean(v),0)))

# ---------- local features ----------
def features(a):
    o,h,l,c,v=[pd.Series(a[:,i]) for i in range(5)]; d=c.diff(); gain=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rsi=(100-100/(1+gain/loss.replace(0,np.nan))).fillna(50); tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=tr.ewm(alpha=1/14,adjust=False).mean()
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); bb=(c-(mid-2*sd))/(4*sd).replace(0,np.nan); vw=((h+l+c)/3*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
    vz=(v-v.rolling(20).mean())/v.rolling(20).std().replace(0,np.nan); eff=(c-c.shift(20)).abs()/c.diff().abs().rolling(20).sum().replace(0,np.nan)
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); ms=macd.ewm(span=9,adjust=False).mean()
    x=lambda q,d=0.:sf(q.iloc[-1] if hasattr(q,"iloc") else q,d)
    out={"rsi":x(rsi,50),"atr_pct":x(atr/c),"bb":x(bb,.5),"vwap_gap":x(c/vw-1),"volume_z":x(vz),"eff":x(eff,.5),"macd_hist":x((macd-ms)/c),"ret5":x(c.pct_change(5)),"ret20":x(c.pct_change(20))}
    if feat:
        try:
            df=pd.DataFrame({"open":a[:,0],"high":a[:,1],"low":a[:,2],"close":a[:,3],"volume":a[:,4]}); f=fn(feat,["build_features"])
            z=call(f,{"candles":df,"include_target":False}) if f else None
            if isinstance(z,tuple):z=z[0]
            if isinstance(z,pd.DataFrame) and len(z):
                row=z.iloc[-1]
                for k,n in {"rsi14":"rsi","rsi_14":"rsi","bb_percent_b":"bb","price_vs_rvwap_20":"vwap_gap","volume_zscore_20":"volume_z","efficiency_ratio_20":"eff"}.items():
                    if k in z.columns: out[n]=sf(row[k],out[n])
        except Exception as e: LOG.warning("feature_engineer fallback: %s",e)
    return out

# ---------- filters ----------
def regime(symbol):
    if not HMM_ON:return {"ok":True,"name":"OFF","p":1}
    if not hmm:return {"ok":not FAIL_CLOSED,"name":"UNKNOWN","p":0,"reason":"module missing"}
    try:
        fr=fn(hmm,["fetch_returns"]); cg=fn(hmm,["get_current_regime"]); fl=fn(hmm,["filter_signals_by_regime"])
        ret=call(fr,{"symbol":symbol,"interval":HMM_INT,"limit":HMM_LIMIT}); res=call(cg,{"returns":ret,"n_states":HMM_STATES}) or {}; name=str(res.get("current_regime","UNKNOWN")).upper(); p=sf(res.get("regime_prob"),0)
        ok=True if fl is None else bool(call(fl,{"regime":name,"probability":p,"signal":"BUY","min_prob":.55}))
        return {"ok":ok,"name":name,"p":p}
    except Exception as e:return {"ok":not FAIL_CLOSED,"name":"UNKNOWN","p":0,"reason":str(e)}
def volatility(symbol,a):
    if not GARCH_ON:return {"ok":True,"regime":"OFF","mult":1.,"bar":max(gk(a),np.std(np.diff(np.log(a[:,3]))[-LOOKBACK:]))}
    if not garch:return {"ok":not FAIL_CLOSED,"regime":"UNKNOWN","mult":0.,"bar":max(gk(a),1e-5)}
    try:
        retf=fn(garch,["fetch_returns"]); fc=fn(garch,["forecast_volatility","forecast"]); cl=fn(garch,["classify_volatility","classify_regime"]); ad=fn(garch,["adjust_position_size_by_volatility"])
        r=call(retf,{"symbol":symbol,"interval":GARCH_INT,"limit":GARCH_LIMIT}); q=call(fc,{"returns":r,"horizon":1}) or {}; ratio=sf(q.get("vol_ratio"),1.); reg=str(cl(ratio) if cl else ("HIGH" if ratio>=1.5 else "NORMAL")).upper(); mult=sf(ad(1.,ratio) if ad else min(1.,1/max(ratio,1.)),0.)
        local=max(gk(a),np.std(np.diff(np.log(a[:,3]))[-LOOKBACK:]),1e-5); bar=np.clip(max(local,sf(q.get("forecast_vol"),0)),local*.5,local*3)
        return {"ok":reg!=MAX_VOL_REGIME,"regime":reg,"ratio":ratio,"mult":float(np.clip(mult,0,1)),"bar":float(bar)}
    except Exception as e:return {"ok":not FAIL_CLOSED,"regime":"UNKNOWN","mult":0.,"bar":max(gk(a),1e-5),"reason":str(e)}
def sentiment(symbol):
    if not SENT_ON:return {"ok":True,"mult":1.,"score":0,"label":"OFF"}
    if not sent_obj:return {"ok":not FAIL_CLOSED,"mult":0.,"score":0,"label":"UNKNOWN"}
    try:
        f=fn(sent_obj,["filter_signal"]); q=f("BUY",symbol) if f else {}; q=q if isinstance(q,dict) else {}; s=q.get("sentiment",{}) or {}; return {"ok":bool(q.get("allowed",True)),"mult":float(np.clip(sf(q.get("multiplier"),1.),SENT_FLOOR,1)),"score":sf(s.get("score")),"label":str(s.get("label","unknown"))}
    except Exception as e:return {"ok":not FAIL_CLOSED,"mult":0.,"score":0,"label":"ERROR","reason":str(e)}
def correlation(symbol,c):
    if not CORR_ON:return {"ok":True}
    if not corr_obj:return {"ok":not FAIL_CLOSED,"reason":"module missing"}
    try:
        up=getattr(corr_obj,"update_prices",None); can=getattr(corr_obj,"can_open_position",None)
        if callable(up):up(symbol,c)
        opens=[]
        for s in SYMBOLS:
            if s!=symbol and os.path.exists(f"paper_position_{s}.json"):opens.append(s)
        q=can(symbol,opens) if callable(can) else True
        return {"ok":bool(q[0] if isinstance(q,tuple) else q),"reason":str(q[1]) if isinstance(q,tuple) and len(q)>1 else "ok"}
    except Exception as e:return {"ok":not FAIL_CLOSED,"reason":str(e)}
def meta(symbol,z,f):
    if not META_ON:return {"ok":True,"p":None}
    m=M("meta_labeling")
    if not m:return {"ok":not FAIL_CLOSED,"p":None}
    try:
        g=fn(m,["predict_meta_probability","predict_meta_label","meta_label"]); q=call(g,{"features":pd.DataFrame([f]),"feature_vector":f,"symbol":symbol,"z":z,"base_signal":1}) if g else None
        if isinstance(q,dict): p=sf(q.get("probability",q.get("meta_probability",q.get("p1",np.nan))),np.nan); ok=bool(q.get("allowed",p>=META_MIN))
        else:p=sf(q,np.nan);ok=p>=META_MIN
        return {"ok":ok,"p":p}
    except:return {"ok":not FAIL_CLOSED,"p":None}

# ---------- decision/risk ----------
def decision(symbol,a,balance):
    c=a[:,3]; z=zscore(c); f=features(a); r=regime(symbol); v=volatility(symbol,a); s=sentiment(symbol); q=correlation(symbol,c); m=meta(symbol,z,f)
    score=0; score+=35*min(max(-z/ZSTR,0),1); score+=10 if f["rsi"]<=RSI_MAX else 0; score+=10 if f["bb"]<=BB_MAX else 0; score+=10*min(max(-f["vwap_gap"]/0.02,0),1); score+=10*max(0,1-f["eff"]) if f["eff"]<=EFF_MAX else 0; score+=5 if f["volume_z"]>-1 else 0; score+=10*r["p"] if r["name"] in {"BULL","RANGE","SIDEWAYS"} else 0; score+=5 if v["regime"]!="HIGH" else 0; score+=5 if s["score"]>=0 else 0; score=float(np.clip(score,0,100))
    sig=z<-ZTH; hard=sig and r["ok"] and v["ok"] and s["ok"] and q["ok"] and m["ok"]; allowed=hard and score>=SCORE_TH
    bar=max(v["bar"],1e-5); entry=float(c[-1]); stop=entry*math.exp(-SLV*bar); take=entry*math.exp(TPV*bar)
    risk_cash=balance*RISK; unit=(entry-stop)+entry*(FEE+SLIP)+stop*(FEE+SLIP); qty=risk_cash/unit if unit>0 else 0; base=min(qty*entry,balance*MAX_POS); size=min(balance*MAX_POS,base*v.get("mult",1)*s.get("mult",1)*(SIZE_FLOOR+(1-SIZE_FLOOR)*score/100))
    return {"z":z,"score":score,"allowed":allowed,"price":entry,"features":f,"regime":r,"vol":v,"sent":s,"corr":q,"meta":m,"sl":stop,"tp":take,"bars":MAX_BARS,"usd":max(size,0),"qty":max(size/entry,0),"reason":("BUY" if allowed else "blocked")}

# ---------- paper / reporting ----------
def balance():
    f=getattr(paper,"load_balance",None) if paper else None
    if not callable(f):raise RuntimeError("paper_trader.load_balance missing")
    return float(f())
def haspos(symbol):return os.path.exists(f"paper_position_{symbol}.json")
def openpos(symbol,d):
    f=getattr(paper,"open_paper_position",None) if paper else None
    if not callable(f):raise RuntimeError("paper_trader.open_paper_position missing")
    vals={"symbol":symbol,"price":d["price"],"z":d["z"],"position_size_usd":d["usd"],"size_usd":d["usd"],"quantity":d["qty"],"stop_price":d["sl"],"take_profit_price":d["tp"],"vertical_bars":d["bars"],"signal_score":d["score"]}
    return call(f,vals)
def checkpos(symbol,p):
    f=getattr(paper,"check_and_close_position",None) if paper else None
    return f(symbol,p) if callable(f) else None
def ffc_status():return ffc.status() if ffc and callable(getattr(ffc,"status",None)) else {"is_live":True}
def trade_update(tr):
    b=balance();
    if ffc and callable(getattr(ffc,"update_after_trade",None)): ffc.update_after_trade(sf(tr.get("pnl_usd")),b,source="paper")
    return b

def stats():
    f=getattr(paper,"get_stats",None) if paper else None; q=f() if callable(f) else {}; return q if isinstance(q,dict) else {}

def performance():
    q=stats(); n=int(sf(q.get("total_trades"))); wr=sf(q.get("win_rate")); ret=sf(q.get("total_return_pct")); out={"trades":n,"winrate":wr,"return":ret}
    # Optional DSR module consumes closed-trade returns if it exposes the function.
    if dsrmod:
        try:
            f=fn(dsrmod,["deflated_sharpe_ratio"]); hist=q.get("returns") or []
            if f and len(hist)>=20: out["dsr"]=sf(call(f,{"returns":np.asarray(hist)/100.,"n_trials":27,"periods_per_year":1}),np.nan)
        except: pass
    return out

# ---------- cycles ----------
_last_full=0.; _last_report=0.; last_zone={}; last_entry={}
def monitor(symbol,a):
    p=checkpos(symbol,float(a[-1,3]))
    if p:
        b=trade_update(p); tg(f"{'🟢' if sf(p.get('pnl_usd'))>0 else '🔴'} <b>Paper EXIT {esc(symbol)}</b>\nP&amp;L: <code>${sf(p.get('pnl_usd')):+.2f} ({sf(p.get('pnl_pct')):+.3f}%)</code>\nBalance: <code>${b:.2f}</code>\nFFC: <b>{'LIVE' if ffc_status().get('is_live',True) else 'HALTED'}</b>")
        return True
    return False

def run_symbol(symbol,a,b):
    if monitor(symbol,a):return
    d=decision(symbol,a,b); z=d["z"]
    if z<-ZSTR and time.time()-last_entry.get(symbol,0)>1800: tg(f"🚨 <b>Strong statistical setup {esc(symbol)}</b>\nPrice: <code>{price(d['price'])}</code>\nZ: <code>{z:.2f}</code>\nScore: <code>{d['score']:.1f}/100</code>\nPaper only after all gates.")
    zone=z<-ZTH
    if zone and not last_zone.get(symbol,False): tg(f"🟢 <b>{esc(symbol)}</b> entered Z-zone\nZ: <code>{z:.2f}</code>\nScore now: <code>{d['score']:.1f}</code>")
    last_zone[symbol]=zone
    if not d["allowed"] or haspos(symbol): return d
    fs=ffc_status(); paper_ok=PAPER_DURING_HALT or bool(fs.get("is_live",True));
    if not paper_ok or d["usd"]<=0:return d
    try:
        p=openpos(symbol,d); last_entry[symbol]=time.time(); tg(f"🟢 <b>Paper ENTRY {esc(symbol)}</b>\nPrice: <code>{price(d['price'])}</code>\nZ: <code>{d['z']:.2f}</code>\nScore: <code>{d['score']:.1f}/100</code>\nSize: <code>${d['usd']:.2f}</code>\nSL: <code>{price(d['sl'])}</code>\nTP: <code>{price(d['tp'])}</code>\nHMM: <code>{esc(d['regime']['name'])}</code> | GARCH: <code>{esc(d['vol']['regime'])}</code> | Sentiment: <code>{esc(d['sent']['label'])}</code>\nFFC live gate: <b>{'ON' if fs.get('is_live',True) else 'HALTED'}</b>")
        return d
    except Exception as e: LOG.exception("entry %s: %s",symbol,e); return d

def full_cycle():
    b=balance()
    for symbol in SYMBOLS:
        try:
            a=candles(symbol); z=zscore(a[:,3]); LOG.info("%s price=%s z=%.2f GK=%.3f%%",symbol,price(a[-1,3]),z,gk(a)*100); run_symbol(symbol,a,b)
        except Exception as e: LOG.warning("%s: %s",symbol,e)

def report():
    q=performance(); f=ffc_status(); st=stats();
    tg("📊 <b>Paper research report</b>\n"+f"Balance: <code>${sf(st.get('balance',0)):.2f}</code>\nReturn: <code>{sf(st.get('total_return_pct',0)):+.3f}%</code>\nTrades: <code>{int(sf(st.get('total_trades',0)))}</code>\nWin rate: <code>{sf(st.get('win_rate',0)):.2%}</code>\nNet P&amp;L: <code>${sf(st.get('net_pnl_usd',0)):+.4f}</code>\nFFC: <b>{'LIVE' if f.get('is_live',True) else 'HALTED'}</b>\nDSR: <code>{q.get('dsr','n/a')}</code>\nNote: diagnostics do not prove future profitability.")

def main():
    if not TOKEN or not CHAT: raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    if os.getenv("TRADING_MODE","paper").lower()!="paper": raise RuntimeError("Only TRADING_MODE=paper is supported")
    if not paper: raise RuntimeError("paper_trader module is required")
    b=balance(); fs=ffc_status();
    tg(f"🚀 <b>Compact Quant Alert v5 started</b>\nSymbols: <code>{esc(', '.join(SYMBOLS))}</code>\nInterval: <code>{esc(INTERVAL)}</code>\nZ: <code>{'robust' if ROBUST else 'standard'}</code> | threshold <code>{ZTH}</code>\nHMM: <code>{HMM_ON}</code> | GARCH: <code>{GARCH_ON}</code> | Sentiment: <code>{SENT_ON}</code> | Correlation: <code>{CORR_ON}</code>\nBalance: <code>${b:.2f}</code>\nFFC: <b>{'LIVE' if fs.get('is_live',True) else 'HALTED'}</b>\nMode: <b>PAPER ONLY</b>")
    global _last_full,_last_report; full_cycle(); _last_full=time.monotonic(); _last_report=time.monotonic()
    while True:
        t=time.monotonic()
        try:
            if t-_last_full>=FULL: full_cycle(); _last_full=time.monotonic()
            if t-_last_report>=REPORT: report(); _last_report=time.monotonic()
        except Exception: LOG.exception("main cycle")
        time.sleep(max(5,FAST-(time.monotonic()-t)))

if __name__=="__main__": main()
