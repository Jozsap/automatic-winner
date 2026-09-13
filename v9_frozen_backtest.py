from __future__ import annotations

import json, math, os, sys, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from huggingface_hub import hf_hub_download
import yfinance as yf

warnings.filterwarnings('ignore')

START = pd.Timestamp('2014-01-02')
TEST_START = pd.Timestamp('2015-01-01')
END = pd.Timestamp('2026-08-31')
TOP_N = 25
TC = 0.001
FIN_RATE = 0.04
VOL_TARGET = 0.15
SINGLE_CAP = 0.06
SECTOR_CAP = 0.25
CORR_CAP = 0.80

REGIME_CAP = {'STRONG_GREEN':1.10,'GREEN':1.00,'YELLOW':0.80,'RED':0.55}
FACTOR_W = {
    'STRONG_GREEN': {'momentum':.35,'quality':.25,'fund_momentum':.25,'value':.10,'low_risk':.05},
    'GREEN': {'momentum':.30,'quality':.30,'fund_momentum':.20,'value':.10,'low_risk':.10},
    'YELLOW': {'momentum':.20,'quality':.35,'fund_momentum':.15,'value':.10,'low_risk':.20},
    'RED': {'momentum':.10,'quality':.35,'fund_momentum':.10,'value':.20,'low_risk':.25},
}
FOLDS = [
    ('2015-01-01','2017-12-31'),
    ('2018-01-01','2020-12-31'),
    ('2021-01-01','2023-12-31'),
    ('2024-01-01','2026-08-31'),
]

ROOT = Path(__file__).resolve().parent
RF = ROOT/'RFundamentals'
OUT = ROOT/'v9_results'
OUT.mkdir(exist_ok=True)


def canon(s: str) -> str:
    return str(s).upper().replace('-', '.').strip()


def winsor_rank(s: pd.Series, higher=True) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf], np.nan)
    if s.notna().sum() < 3:
        return pd.Series(np.nan,index=s.index)
    lo, hi = s.quantile(.05), s.quantile(.95)
    r = s.clip(lo,hi).rank(pct=True,method='average')
    return r if higher else 1-r


def perf(r: pd.Series) -> dict:
    r=r.dropna()
    if len(r)<2: return {}
    nav=(1+r).cumprod(); yrs=len(r)/252
    cagr=nav.iloc[-1]**(1/yrs)-1
    vol=r.std(ddof=1)*np.sqrt(252)
    sharpe=r.mean()*252/vol if vol>0 else np.nan
    dd=nav/nav.cummax()-1
    return {'CAGR':float(cagr),'Volatility':float(vol),'Sharpe':float(sharpe),'MaxDD':float(dd.min()),'TotalReturn':float(nav.iloc[-1]-1),'Days':int(len(r))}


def get_membership() -> pd.DataFrame:
    p=RF/'data/sp500_constituents_.csv'
    u=pd.read_csv(p)
    u['ticker']=u['ticker'].map(canon)
    u['date_added']=pd.to_datetime(u['date_added'],errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    u['date_removed']=pd.to_datetime(u['date_removed'],errors='coerce').fillna(pd.Timestamp('2100-01-01'))
    return u[['ticker','date_added','date_removed']]


def active_universe(u: pd.DataFrame, d: pd.Timestamp) -> set[str]:
    x=u[(u.date_added<=d)&(u.date_removed>d)]
    return set(x.ticker)


def download_price_panel(membership: pd.DataFrame) -> pd.DataFrame:
    frames=[]
    os.environ.setdefault('HF_HUB_DISABLE_XET','0')
    for year in range(2014,2026):
        fn=f'price_daily/year={year}/part-000.parquet'
        path=hf_hub_download(repo_id='finsaber-team/FINSABER-V2-Data',repo_type='dataset',filename=fn)
        x=pd.read_parquet(path,columns=['date','symbol','adjusted_close'])
        x['date']=pd.to_datetime(x.date); x['ticker']=x.symbol.map(canon)
        x=x[['date','ticker','adjusted_close']].rename(columns={'adjusted_close':'px'})
        frames.append(x)
    px=pd.concat(frames,ignore_index=True).dropna(subset=['px'])
    px=px[(px.date>=START)&(px.date<=pd.Timestamp('2025-12-31'))]

    # For the frozen 2026 fold use Yahoo adjusted prices for the 2026 PIT universe.
    names=sorted(set().union(*[active_universe(membership,pd.Timestamp(d)) for d in ['2026-01-02','2026-04-01','2026-07-01']]))
    yf_names=[t.replace('.','-') for t in names]
    yframes=[]
    for i in range(0,len(yf_names),50):
        batch=yf_names[i:i+50]
        try:
            z=yf.download(batch,start='2024-01-01',end='2026-09-02',auto_adjust=False,actions=False,progress=False,threads=True,group_by='column')
            if z.empty: continue
            if isinstance(z.columns,pd.MultiIndex):
                field='Adj Close' if 'Adj Close' in z.columns.get_level_values(0) else 'Close'
                q=z[field].copy()
                for c in q.columns:
                    a=q[[c]].dropna().reset_index(); a.columns=['date','px']; a['ticker']=canon(c); yframes.append(a)
            else:
                field='Adj Close' if 'Adj Close' in z.columns else 'Close'
                a=z[[field]].dropna().reset_index(); a.columns=['date','px']; a['ticker']=canon(batch[0]); yframes.append(a)
        except Exception as e:
            print('YF batch warning',i,e)
    if yframes:
        y=pd.concat(yframes,ignore_index=True); y['date']=pd.to_datetime(y.date).dt.tz_localize(None)
        y=y[(y.date>=pd.Timestamp('2024-01-01'))&(y.date<=END)]
        # Override 2024+ for 2026-universe names to maintain one adjustment basis across 2026 momentum lookback.
        key=set(y.ticker.unique())
        px=px[~((px.date>=pd.Timestamp('2024-01-01')) & px.ticker.isin(key))]
        px=pd.concat([px,y],ignore_index=True)
    px=px.sort_values(['date','ticker']).drop_duplicates(['date','ticker'],keep='last')
    wide=px.pivot(index='date',columns='ticker',values='px').sort_index()
    wide=wide.loc[(wide.index>=START)&(wide.index<=END)]
    wide.to_parquet(OUT/'prices.parquet')
    return wide


def download_spy() -> pd.Series:
    z=yf.download('SPY',start='2014-01-01',end='2026-09-02',auto_adjust=False,progress=False)
    if isinstance(z.columns,pd.MultiIndex):
        q=z['Adj Close']['SPY'] if 'Adj Close' in z.columns.get_level_values(0) else z['Close']['SPY']
    else:
        q=z['Adj Close'] if 'Adj Close' in z.columns else z['Close']
    q.index=pd.to_datetime(q.index).tz_localize(None)
    q=q.loc[(q.index>=START)&(q.index<=END)].dropna(); q.name='SPY'
    return q


def load_sector_table() -> pd.DataFrame:
    p=RF/'cache/lookups/sector_industry.parquet'
    x=pd.read_parquet(p)
    x['ticker']=x.ticker.map(canon)
    x['valid_from']=pd.to_datetime(x.get('valid_from',pd.Timestamp('2010-01-01')),errors='coerce').fillna(pd.Timestamp('2010-01-01'))
    return x


def sector_asof(sec: pd.DataFrame,ticker:str,d:pd.Timestamp) -> str:
    q=sec[(sec.ticker==ticker)&(sec.valid_from<=d)]
    if q.empty: return 'Unknown'
    return str(q.sort_values('valid_from').iloc[-1].sector)


def load_sparse_fund_data(signal_dates: List[pd.Timestamp], tickers:set[str], sec:pd.DataFrame) -> Dict[pd.Timestamp,pd.DataFrame]:
    tsdir=RF/'cache/timeseries'
    snapshots={d:[] for d in signal_dates}
    wanted_daily=['date','roe','operating_margin','debt_equity','revenue_growth_yoy','eps_growth_yoy','earnings_yield','pfcf']
    for n,t in enumerate(sorted(tickers)):
        dp=tsdir/f'{t}_daily.parquet'; fp=tsdir/f'{t}_fund.parquet'
        if not dp.exists() or not fp.exists(): continue
        try:
            schema=pq.read_schema(dp); cols=[c for c in wanted_daily if c in schema.names]
            ddf=pd.read_parquet(dp,columns=cols); ddf['date']=pd.to_datetime(ddf.date); ddf=ddf.sort_values('date')
            ff=pd.read_parquet(fp); ff['filed_date']=pd.to_datetime(ff.filed_date)
            if 'quarter' in ff.columns: ff=ff[ff.quarter.astype(str)=='FY']
            ff=ff.sort_values(['fiscal_year','filed_date']).drop_duplicates('fiscal_year',keep='last')
            for sd in signal_dates:
                dr=ddf[ddf.date<=sd]
                fr=ff[ff.filed_date<=sd]
                if dr.empty or fr.empty: continue
                a=dr.iloc[-1]; cur=fr.sort_values('fiscal_year').iloc[-1]
                prev=fr[fr.fiscal_year < cur.fiscal_year]
                prev=prev.sort_values('fiscal_year').iloc[-1] if not prev.empty else None
                cfo=float(cur.get('stub_cfo',np.nan)); assets=float(cur.get('stub_assets',np.nan))
                cfo_assets=cfo/assets if np.isfinite(cfo) and np.isfinite(assets) and assets!=0 else np.nan
                cfo_yoy=np.nan; op_margin_yoy=np.nan
                if prev is not None:
                    pc=float(prev.get('stub_cfo',np.nan))
                    if np.isfinite(cfo) and np.isfinite(pc) and pc!=0: cfo_yoy=(cfo-pc)/abs(pc)
                    cm=float(cur.get('operating_margin',np.nan)); pm=float(prev.get('operating_margin',np.nan))
                    if np.isfinite(cm) and np.isfinite(pm): op_margin_yoy=cm-pm
                pfcf=float(a.get('pfcf',np.nan)); fcf_yield=(1/pfcf) if np.isfinite(pfcf) and pfcf!=0 else np.nan
                snapshots[sd].append({
                    'ticker':t,'roe':a.get('roe',np.nan),'op_margin':a.get('operating_margin',np.nan),
                    'debt_equity':a.get('debt_equity',np.nan),'cfo_assets':cfo_assets,
                    'rev_yoy':a.get('revenue_growth_yoy',np.nan),'eps_yoy':a.get('eps_growth_yoy',np.nan),
                    'op_margin_yoy':op_margin_yoy,'cfo_yoy':cfo_yoy,'earnings_yield':a.get('earnings_yield',np.nan),
                    'fcf_yield':fcf_yield,'sector':sector_asof(sec,t,sd)
                })
        except Exception as e:
            print('fund warning',t,e)
        if (n+1)%100==0: print('fund tickers',n+1)
    return {d:pd.DataFrame(rows).set_index('ticker') if rows else pd.DataFrame() for d,rows in snapshots.items()}


def regime(spy:pd.Series,d:pd.Timestamp)->str:
    s=spy.loc[:d].dropna()
    if len(s)<253:return 'GREEN'
    c=s.iloc[-1]; ma50=s.tail(50).mean(); ma200=s.tail(200).mean(); m6=c/s.iloc[-127]-1; m12=c/s.iloc[-253]-1
    if c>ma200 and ma50>ma200 and m6>0 and m12>0:return 'STRONG_GREEN'
    if c>ma200 and m12>0:return 'GREEN'
    if (c>ma200) ^ (m12>0):return 'YELLOW'
    return 'RED'


def technical(px:pd.DataFrame,d:pd.Timestamp,univ:set[str])->pd.DataFrame:
    hist=px.loc[:d,px.columns.intersection(sorted(univ))]
    rows=[]
    for t in hist.columns:
        s=hist[t].dropna()
        if len(s)<253:continue
        ret=s.pct_change().dropna()
        if len(ret)<63:continue
        rows.append({'ticker':t,'m12_1':s.iloc[-22]/s.iloc[-253]-1,'m6_1':s.iloc[-22]/s.iloc[-127]-1,
                     'vol63':ret.tail(63).std(ddof=1)*np.sqrt(252),'dd252':(s.tail(252)/s.tail(252).cummax()-1).min()})
    return pd.DataFrame(rows).set_index('ticker') if rows else pd.DataFrame()


def score_cross(tech:pd.DataFrame,fund:pd.DataFrame,reg:str)->pd.DataFrame:
    if tech.empty or fund.empty:return pd.DataFrame()
    x=tech.join(fund,how='inner')
    sub={}
    sub['mom12']=winsor_rank(x.m12_1,True); sub['mom6']=winsor_rank(x.m6_1,True)
    x['momentum']=.6*sub['mom12']+.4*sub['mom6']
    qparts=pd.concat([winsor_rank(x.roe),winsor_rank(x.op_margin),winsor_rank(x.cfo_assets),winsor_rank(x.debt_equity,False)],axis=1)
    x['quality']=qparts.mean(axis=1,skipna=True)
    fparts=pd.concat([winsor_rank(x.rev_yoy),winsor_rank(x.eps_yoy),winsor_rank(x.op_margin_yoy),winsor_rank(x.cfo_yoy)],axis=1)
    x['fund_momentum']=fparts.mean(axis=1,skipna=True)
    vparts=pd.concat([winsor_rank(x.earnings_yield),winsor_rank(x.fcf_yield)],axis=1)
    x['value']=vparts.mean(axis=1,skipna=True)
    x['low_risk']=pd.concat([winsor_rank(x.vol63,False),winsor_rank(x.dd252,True)],axis=1).mean(axis=1,skipna=True)
    w=FACTOR_W[reg]; factors=list(w)
    observed=pd.DataFrame({k:x[k].notna().astype(float)*w[k] for k in factors})
    x['observed_weight']=observed.sum(axis=1)
    eligible=x.observed_weight>=.80
    num=sum(x[k].fillna(0)*w[k] for k in factors)
    x['score']=np.where(eligible,num/x.observed_weight,np.nan)
    return x.dropna(subset=['score']).sort_values('score',ascending=False)


def constrained_weights(scored:pd.DataFrame,returns126:pd.DataFrame)->Dict[str,float]:
    s=scored.head(TOP_N).copy(); names=list(s.index)
    if len(names)<15:return {}
    base=(1/s.vol63.clip(lower=.05)); base=base/base.sum()
    ub=np.full(len(names),SINGLE_CAP)
    if len(returns126)>20:
        corr=returns126[names].corr(min_periods=30)
        rank={t:i for i,t in enumerate(names)}
        for i,a in enumerate(names):
            for j,b in enumerate(names):
                if j<=i:continue
                if s.loc[a,'sector']==s.loc[b,'sector'] and pd.notna(corr.loc[a,b]) and corr.loc[a,b]>CORR_CAP:
                    lower=b if rank[b]>rank[a] else a
                    k=names.index(lower); ub[k]=min(ub[k],0.5*base.loc[lower])
    x0=np.minimum(base.values,ub); x0=x0/x0.sum()
    cons=[{'type':'eq','fun':lambda w:np.sum(w)-1}]
    for secname in s.sector.fillna('Unknown').unique():
        idx=np.array([i for i,t in enumerate(names) if str(s.loc[t,'sector'])==str(secname)])
        cons.append({'type':'ineq','fun':lambda w,idx=idx:SECTOR_CAP-np.sum(w[idx])})
    res=minimize(lambda w:np.sum((w-base.values)**2),x0,method='SLSQP',bounds=[(0,float(u)) for u in ub],constraints=cons,options={'maxiter':500,'ftol':1e-12})
    if not res.success:
        print('weight optimizer warning',res.message)
        w=x0
    else:w=res.x
    w=np.maximum(w,0); w=w/w.sum()
    return dict(zip(names,w))


def make_rebalance_plan(px,spy,membership,fundsnaps,signal_dates):
    plan={}; audit=[]
    all_dates=px.index
    for sd in signal_dates:
        future=all_dates[all_dates>sd]
        if len(future)==0:continue
        execd=future[0]
        univ=active_universe(membership,sd)
        te=technical(px,sd,univ); reg=regime(spy,sd); sc=score_cross(te,fundsnaps.get(sd,pd.DataFrame()),reg)
        if sc.empty:continue
        r126=px.loc[:sd,px.columns.intersection(sc.head(TOP_N).index)].pct_change().tail(126)
        wt=constrained_weights(sc,r126)
        plan[execd]=wt
        audit.append({'signal_date':sd,'execution_close':execd,'regime':reg,'eligible':len(sc),'holdings':len(wt),'top_score':float(sc.score.iloc[0]) if len(sc) else np.nan})
    pd.DataFrame(audit).to_csv(OUT/'rebalance_audit.csv',index=False)
    return plan


def desired_exposure(reg,base_vol,dd):
    cap=REGIME_CAP[reg]
    vexp=cap if not np.isfinite(base_vol) or base_vol<=0 else float(np.clip(VOL_TARGET/base_vol,.40,cap))
    dcap=10.0
    if dd<=-.20:dcap=.40
    elif dd<=-.15:dcap=.60
    elif dd<=-.10:dcap=.80
    return min(cap,vexp,dcap)


def simulate(px,spy,plan):
    dates=px.index[(px.index>=TEST_START)&(px.index<=END)]
    nav=1.0; peak=1.0; stock_w={}; current_exposure=0.0; rel_target={}
    base_ret_hist=[]; rows=[]; turnover_total=0.0
    prev_px=px.shift(1)
    spy_ret=spy.pct_change()
    spy_nav=1.0
    for d in dates:
        # Realized return from weights held after previous close.
        gross=0.0; basegross=0.0
        if stock_w:
            for t,w in list(stock_w.items()):
                if t in px.columns and pd.notna(px.at[d,t]) and d in prev_px.index and pd.notna(prev_px.at[d,t]) and prev_px.at[d,t]!=0:
                    rr=px.at[d,t]/prev_px.at[d,t]-1
                else: rr=0.0
                gross += w*rr
                if current_exposure>0: basegross += (w/current_exposure)*rr
        financing=max(current_exposure-1,0)*FIN_RATE/252
        nav *= (1+gross-financing)
        if not np.isfinite(nav) or nav<=0: raise RuntimeError('NAV invalid')
        peak=max(peak,nav); dd=nav/peak-1
        base_ret_hist.append(basegross)

        # Drift current weights after close return.
        drift={}
        denom=(1+gross-financing)
        if stock_w and denom!=0:
            for t,w in stock_w.items():
                rr=(px.at[d,t]/prev_px.at[d,t]-1) if (t in px.columns and pd.notna(px.at[d,t]) and pd.notna(prev_px.at[d,t]) and prev_px.at[d,t]!=0) else 0.0
                drift[t]=w*(1+rr)/denom
        # A scheduled portfolio target is executed at this close; it was based on prior-close signal.
        if d in plan and plan[d]: rel_target=plan[d].copy()

        reg=regime(spy,d)
        bv=np.std(base_ret_hist[-21:],ddof=1)*np.sqrt(252) if len(base_ret_hist)>=21 else np.nan
        exp=desired_exposure(reg,bv,dd)
        target={t:exp*w for t,w in rel_target.items()} if rel_target else {}
        names=set(drift)|set(target)
        turn=sum(abs(target.get(t,0)-drift.get(t,0)) for t in names)
        cost=TC*turn
        nav*=max(1-cost,1e-9)
        turnover_total+=turn
        stock_w=target; current_exposure=sum(stock_w.values())

        sr=float(spy_ret.get(d,np.nan));
        if np.isfinite(sr): spy_nav*=1+sr
        rows.append({'date':d,'v9_return':gross-financing-cost,'nav':nav,'drawdown':nav/peak-1,'regime':reg,'base_vol21':bv,'exposure':current_exposure,'turnover':turn,'spy_return':sr,'spy_nav':spy_nav})
    out=pd.DataFrame(rows).set_index('date')
    out.to_csv(OUT/'daily_returns.csv')
    return out,turnover_total


def rolling_3y_excess(df):
    a=df[['v9_return','spy_return']].dropna(); n=756
    vals=[]
    for i in range(n-1,len(a),21):
        z=a.iloc[i-n+1:i+1]
        v=np.prod(1+z.v9_return)**(252/len(z))-1; s=np.prod(1+z.spy_return)**(252/len(z))-1
        vals.append(v-s)
    return float(np.mean(np.array(vals)>0)) if vals else np.nan, vals


def yearly_excess_concentration(df):
    a=df[['v9_return','spy_return']].dropna().copy(); a['year']=a.index.year
    q=a.groupby('year').apply(lambda z:np.log1p(z.v9_return).sum()-np.log1p(z.spy_return).sum())
    total=q.sum(); largest=q.max()
    share=largest/total if total>0 else np.inf
    return float(share),q


def main():
    print('V9-1.0-FROZEN validation start')
    membership=get_membership(); px=download_price_panel(membership); spy=download_spy()
    # Align on SPY trading calendar and retain stock NaNs.
    calendar=spy.index[(spy.index>=START)&(spy.index<=END)]
    px=px.reindex(calendar); spy=spy.reindex(calendar).ffill()
    # Quarterly signals: last trading close before first trading day of Jan/Apr/Jul/Oct.
    exec_candidates=[]; signal_dates=[]
    for y in range(2015,2027):
        for m in (1,4,7,10):
            if pd.Timestamp(y,m,1)>END: continue
            q=calendar[(calendar.year==y)&(calendar.month==m)]
            if len(q)==0:continue
            ex=q[0]; prev=calendar[calendar<ex]
            if len(prev):signal_dates.append(prev[-1]); exec_candidates.append(ex)
    all_tickers=set().union(*[active_universe(membership,d) for d in signal_dates])
    sec=load_sector_table(); fund=load_sparse_fund_data(signal_dates,all_tickers,sec)
    plan=make_rebalance_plan(px,spy,membership,fund,signal_dates)
    df,total_turn=simulate(px,spy,plan)

    stitched=df[['v9_return','spy_return']].dropna()
    vm=perf(stitched.v9_return); sm=perf(stitched.spy_return)
    years=len(stitched)/252; ann_turn=total_turn/years
    roll_share,rollvals=rolling_3y_excess(df); concentration,annual_ex=yearly_excess_concentration(df)
    folds=[]
    for a,b in FOLDS:
        z=stitched.loc[a:b]
        folds.append({'fold':f'{a}:{b}','V9':perf(z.v9_return),'SPY':perf(z.spy_return),'ExcessCAGR':perf(z.v9_return).get('CAGR',np.nan)-perf(z.spy_return).get('CAGR',np.nan)})
    gates={
        'CAGR_ge_15':vm['CAGR']>=.15,
        'Sharpe_ge_080':vm['Sharpe']>=.80,
        'MaxDD_le_25':vm['MaxDD']>=-.25,
        'CAGR_gt_SPY':vm['CAGR']>sm['CAGR'],
        'rolling3y_excess_positive_ge_60pct':roll_share>=.60,
        'single_year_excess_contribution_le_35pct':concentration<=.35,
        'turnover_not_fail':ann_turn<=2.0,
        'turnover_preferred':ann_turn<=1.5,
    }
    result={'strategy':'V9-1.0-FROZEN','period':[str(stitched.index.min().date()),str(stitched.index.max().date())],
            'V9':vm,'SPY':sm,'ExcessCAGR':vm['CAGR']-sm['CAGR'],'annual_one_way_turnover':ann_turn,
            'rolling_3y_positive_excess_share':roll_share,'largest_year_log_excess_share':concentration,
            'final_huf_from_3m':3_000_000*(1+vm['TotalReturn']),'folds':folds,'gates':gates,
            'PASS':all([gates['CAGR_ge_15'],gates['Sharpe_ge_080'],gates['MaxDD_le_25'],gates['CAGR_gt_SPY'],gates['rolling3y_excess_positive_ge_60pct'],gates['single_year_excess_contribution_le_35pct'],gates['turnover_not_fail']])}
    annual_ex.rename('log_excess').to_csv(OUT/'annual_excess.csv')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
