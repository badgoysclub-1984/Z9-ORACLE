import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
         ResponsiveContainer, ReferenceLine } from "recharts";

// ═══════════════════════════════════════════════════════════════
//  CONSTANTS — Z9 Oracle v16.0 "ULTRA"
//  ML Fusion v2 + RL Adaptive Scaling + GARCH Simulation
//  Optimized for Pi 5 Production Environment
// ═══════════════════════════════════════════════════════════════
const SYMS  = ["BTC-USD","ETH-USD","SOL-USD"];
const CLR: Record<string, string>   = {"BTC-USD":"#f7931a","ETH-USD":"#627eea","SOL-USD":"#14F195"};
const SEEDS: Record<string, number> = {"BTC-USD":74311,"ETH-USD":2328,"SOL-USD":84};
const BVOL: Record<string, number>  = {"BTC-USD":6e-4,"ETH-USD":8e-4,"SOL-USD":1.1e-3};
const CORR: Record<string, number>  = {"ETH-USD":0.82,"SOL-USD":0.72};
const STO   = "z9_oracle_v16_ultra_log";

const CFG = {
  // ML fusion
  ML_LR: 0.0072, ML_DECAY: 0.9965, ML_SCALE: 1.92,
  // RL policy
  RL_LR: 0.0085, RL_DECAY: 0.995, RL_DISCOUNT: 0.92,
  // Strategy core
  RR_MULT: { trend:7.2, range:2.9, volatile:5.4, warmup:4.0 } as Record<string, number>,
  SL_MULT: 1.08, TRAIL_MULT: 0.61, TRAIL_ACTIVATE_MULT: 1.9,
  BE_TRIG_MULT: 2.4, PARTIAL_PCT: 0.56, PARTIAL_R: 2.35,
  PROFIT_PROTECTION_R: 1.5,
  ACCEL_GATE: { trend:-0.04, range:-0.14, volatile:-0.10 } as Record<string, number>,
  CONFIDENCE_BOOST_FACTOR: 0.08,
  WARMUP:20, THRESH_BASE:0.59,
  REGIME_THRESH: { trend:0.48, range:0.85, volatile:0.69, warmup:0.92 } as Record<string, number>,
  MIN_ZSCORE:0.81,
  COOLDOWN_BASE:600,
  COOLDOWN_MULT: { trend:0.53, range:1.65, volatile:1.80 } as Record<string, number>,
  MAX_ATR:0.0026,
  TICK_CONFIRM:5,
  MAX_POS:850, LEV_BASE:44, MIN_LEV:19, MAX_LEV:62,
  MAX_OPEN:4, CB_N:4, CB_WAIT:15800,
  MAX_HOLD: { trend:34000, other:14800 } as Record<string, number>,
  BAL0:1000, CHART_LEN:200, TICK_MS:60,
};

// ═══════════════════════════════════════════════════════════════
//  MATH HELPERS
// ═══════════════════════════════════════════════════════════════
const gauss = () => { let s=0; for(let i=0;i<12;i++) s+=Math.random(); return s-6; };

const calcATR = (buf: number[]) => {
  if(buf.length<15) return null;
  const sl=buf.slice(-15);
  return sl.slice(1).reduce((s,p,i)=>s+Math.abs(p-sl[i])/(sl[i]+1e-9),0)/14;
};

const kellySize = (closed: any[], balance: number, regime: string, atr: number | null, streak: number, drawdown: number, conf: number, mlLogit: number) => {
  const n = closed.length;
  if (n < 15) return Math.min(CFG.MAX_POS, balance * 0.26);
  let wWins = 0, wLoss = 0, totalW = 0, w = 1.0;
  for (let i = 0; i < Math.min(60, n); i++) {
    const pnl = closed[i].pnlPct;
    if (pnl > 0) wWins += pnl * w; else wLoss += Math.abs(pnl) * w;
    totalW += w; w *= 0.92;
  }
  const W = wWins / (wWins + wLoss);
  const avgW = wWins / (totalW * W || 1);
  const avgL = wLoss / (totalW * (1 - W) || 1);
  let f = (W - (1 - W) / (avgW / avgL)) * 0.49;
  f = Math.max(0.07, Math.min(0.49, f));
  const regimeMult: Record<string, number> = {trend:1.50, range:0.47, volatile:0.77, warmup:0.60};
  f *= (regimeMult[regime] || 1.0) * Math.max(0.64, Math.min(1.45, conf * 2.3));
  if (streak > 5) f *= 1.11; else if (streak < -4) f *= 0.76;
  if (drawdown > 0.11) f *= 0.20; else if (drawdown > 0.05) f *= 0.42;
  const volAdj = Math.max(0.60, Math.min(1.40, 0.00155 / (atr || 0.001)));
  f *= volAdj;
  const levMult = Math.max(0.84, Math.min(1.36, 1 + mlLogit * 0.19));
  return Math.min(CFG.MAX_POS, balance * f * levMult);
};

const rlScalePosition = (state: any, weights: any) => {
  let score = weights.bias;
  score += state.regimeTrend * weights.regimeTrend;
  score += state.regimeVol * weights.regimeVol;
  score += state.conf * weights.conf;
  score += state.mlLogit * weights.mlLogit;
  score += state.streak * weights.streak;
  score += state.drawdown * weights.drawdown;
  const mult = 0.55 + 1.25 / (1 + Math.exp(-score));
  return Math.max(0.55, Math.min(1.85, mult));
};

const updateRLWeights = (weights: any, state: any, reward: number, lr: number) => {
  const grad = reward > 0 ? 1.45 : -0.68;
  weights.bias = weights.bias * CFG.RL_DECAY + lr * grad * 0.32;
  weights.regimeTrend = weights.regimeTrend * CFG.RL_DECAY + lr * grad * state.regimeTrend;
  weights.regimeVol = weights.regimeVol * CFG.RL_DECAY + lr * grad * state.regimeVol;
  weights.conf = weights.conf * CFG.RL_DECAY + lr * grad * state.conf;
  weights.mlLogit = weights.mlLogit * CFG.RL_DECAY + lr * grad * state.mlLogit;
  weights.streak = weights.streak * CFG.RL_DECAY + lr * grad * state.streak;
  weights.drawdown = weights.drawdown * CFG.RL_DECAY + lr * grad * state.drawdown;
};

// ═══════════════════════════════════════════════════════════════
//  SIGNAL ENGINE
// ═══════════════════════════════════════════════════════════════
const tickConfirmation = (buf: number[], dir: number, n=CFG.TICK_CONFIRM) => {
  if(buf.length<n) return 0;
  const sl=buf.slice(-n); let c=0;
  for(let i=1;i<sl.length;i++) if(dir>0?sl[i]>sl[i-1]:sl[i]<sl[i-1]) c++;
  return c;
};

const zMomentum = (buf: number[]) => {
  const n=buf.length;
  if(n<28) return {dir:0,conf:0,zscore:0,vol:0,tc:0,accel:0};
  const sw=Math.min(14,Math.floor(n/4));
  const mw=Math.min(38,Math.floor(n/2));
  const sma=buf.slice(-sw).reduce((a,b)=>a+b,0)/sw;
  const mbuf=buf.slice(-mw,-sw);
  const mma=mbuf.length>0?mbuf.reduce((a,b)=>a+b,0)/mbuf.length:sma;
  const recent=buf.slice(-Math.min(22,n));
  const mu=recent.reduce((a,b)=>a+b,0)/recent.length;
  const std=Math.sqrt(recent.reduce((a,v)=>a+(v-mu)**2,0)/recent.length)+1e-9;
  const mom=sma-mma;
  const dir=Math.sign(mom);
  const zscore=mom/std;
  const vol=std/(mu+1e-9);
  const prevRecent=buf.slice(-Math.min(30,n)-8,-Math.min(22,n));
  const prevMu=prevRecent.length?prevRecent.reduce((a,b)=>a+b,0)/prevRecent.length:mu;
  const prevStd=Math.sqrt(prevRecent.reduce((a,v)=>a+(v-prevMu)**2,0)/prevRecent.length)+1e-9;
  const prevZ=(sma-mma)/prevStd;
  const accel = zscore - prevZ;
  const rawConf=0.54+0.46*Math.tanh(Math.abs(zscore)*1.10);
  const tc=tickConfirmation(buf,dir);
  const tcBonus=Math.max(0,(tc-3)*0.035);
  const conf=Math.min(0.99,rawConf+tcBonus);
  return {dir,conf,zscore,vol,tc,accel};
};

const classifyRegime = (atr: number | null, zscore: number) => {
  if(atr==null) return "warmup";
  if(atr*100>0.165) return "volatile";
  if(Math.abs(zscore)>1.30) return "trend";
  return "range";
};

const mlFuse = (features: any, weights: any) => {
  let logit = weights.bias;
  logit += features.zscore * weights.zscore;
  logit += features.accel * weights.accel;
  logit += features.tc * weights.tc;
  logit += features.bConf * weights.bConf;
  logit += features.regimeTrend * weights.regimeTrend;
  logit += features.regimeVol * weights.regimeVol;
  logit += features.vol * weights.vol;
  const raw = 0.5 + 0.5 * Math.tanh(logit * CFG.ML_SCALE);
  return {dir: logit > 0 ? 1 : -1, conf: Math.min(0.99, Math.max(0.54, raw)), logit};
};

const ensembleSignal = (sym: string, st: any) => {
  const buf=st.buffers[sym];
  const m = zMomentum(buf);
  let bull=0,bear=0,tw=0;
  if(m.dir!==0){ const w=0.64; m.dir>0?bull+=m.conf*w:bear+=m.conf*w; tw+=w; }
  let bDir=0,bConf=0;
  if(sym!=="BTC-USD"&&st.buffers["BTC-USD"].length>=28){
    const bl=zMomentum(st.buffers["BTC-USD"]);
    bDir=bl.dir; bConf=Math.min(0.93,bl.conf*0.95);
    if(bDir!==0){ const w=0.36; bDir>0?bull+=bConf*w:bear+=bConf*w; tw+=w; }
  }
  const regime=classifyRegime(st.atr[sym],m.zscore);
  if(tw===0||bull===bear) return{dir:0,conf:0,zscore:0,vol:0,tc:0,accel:0,bDir,bConf,regime,mlConf:0,mlLogit:0};
  const regimeTrend = regime==="trend" ? 1.0 : 0;
  const regimeVol = regime==="volatile" ? 1.0 : 0;
  const features = {zscore: m.zscore, accel: m.accel, tc: m.tc, bConf: bConf || 0, regimeTrend, regimeVol, vol: m.vol * -1};
  const fused = mlFuse(features, st.mlWeights);
  let conf=Math.max(bull,bear)/tw;
  const regimeAdj = regime === "trend" ? 1.18 : regime === "range" ? 0.79 : regime === "volatile" ? 0.86 : 1.0;
  conf = Math.min(0.99, conf * regimeAdj * fused.conf);
  const recentClosed = st.closed.slice(0,10);
  const sameRegimeWins = recentClosed.filter((t: any) => t.regime === regime && t.pnlAbs > 0).length;
  if (sameRegimeWins >= 2) conf = Math.min(0.99, conf * (1 + CFG.CONFIDENCE_BOOST_FACTOR));
  const dir = fused.dir;
  return{dir,conf,zscore:m.zscore,vol:m.vol,tc:m.tc,accel:m.accel,bDir,bConf,regime,mlConf:fused.conf,mlLogit:fused.logit};
};

// ═══════════════════════════════════════════════════════════════
//  ORACLE PRICE PROCESSOR
// ═══════════════════════════════════════════════════════════════
const processPrice = (sym: string, price: number, now: number, st: any) => {
  st.latestPrices[sym]=price; st.totalTicks++;
  st.chartData[sym].push({t:st.ticks[sym]++,price});
  if(st.chartData[sym].length>CFG.CHART_LEN) st.chartData[sym].shift();
  const buf=st.buffers[sym]; buf.push(price); if(buf.length>100) buf.shift();
  const atr=calcATR(buf); if(atr!=null) st.atr[sym]=Math.max(3e-4,Math.min(3e-3,atr));

  for(let i=st.openTrades.length-1;i>=0;i--){
    const op=st.openTrades[i]; if(op.sym!==sym) continue;
    let pnlPct=0,exit=false,why="";
    if(op.side==="Long"){
      pnlPct=(price-op.entry)/op.entry;
      if(!op.trailActive&&pnlPct>=op.trailActPct) op.trailActive=true;
      if(!op.be&&pnlPct>=op.beTrigPct){op.trail=op.entry*(1+0.0005);op.be=true;}
      if(op.trailActive&&price>op.peak){
        op.peak=price; let tP = op.trailPct;
        if (pnlPct >= CFG.PROFIT_PROTECTION_R * op.slPct) tP *= 0.6;
        const nt=price*(1-tP); if(nt>op.trail) op.trail=nt;
      }
      if(price<=op.trail){exit=true;why="trail";}
    }else{
      pnlPct=(op.entry-price)/op.entry;
      if(!op.trailActive&&pnlPct>=op.trailActPct) op.trailActive=true;
      if(!op.be&&pnlPct>=op.beTrigPct){op.trail=op.entry*(1-0.0005);op.be=true;}
      if(op.trailActive&&price<op.trough){
        op.trough=price; let tP = op.trailPct;
        if (pnlPct >= CFG.PROFIT_PROTECTION_R * op.slPct) tP *= 0.6;
        const nt=price*(1+tP); if(nt<op.trail) op.trail=nt;
      }
      if(price>=op.trail){exit=true;why="trail";}
    }
    if(pnlPct>=op.tpPct){exit=true;why="TP";}
    if(pnlPct<=-op.slPct){exit=true;why="SL";}
    if(now-op.ts>=op.maxHold){exit=true;why="t/o";}

    if(!op.partialTaken && pnlPct >= op.partialR * op.slPct){
      const partPos = op.pos * CFG.PARTIAL_PCT;
      const partPnl = partPos * (op.partialR * op.slPct) * op.lev;
      st.balance += partPnl;
      st.closed.unshift({sym,side:op.side,entry:op.entry,exitP:price,pnlPct:op.partialR*op.slPct,pnlAbs:partPnl,hold:(now-op.ts)/1000,why:"PARTIAL",conf:op.conf,lev:op.lev,pos:partPos,ts:now,regime:op.regime});
      op.pos *= (1 - CFG.PARTIAL_PCT); op.partialTaken = true;
      op.trail = op.side==="Long" ? op.entry*1.0010 : op.entry*0.9990;
      op.be = true; op.trailActive = true;
    }

    if(exit){
      const hold=(now-op.ts)/1000; const pnlAbs=op.pos*pnlPct*op.lev;
      st.balance+=pnlAbs; st.peak=Math.max(st.peak,st.balance);
      if(!st.firstTs) st.firstTs=now; const dd=(st.peak-st.balance)/st.peak;
      if(dd>st.maxDD) st.maxDD=dd;
      st.closed.unshift({sym,side:op.side,entry:op.entry,exitP:price,pnlPct,pnlAbs,hold,why,conf:op.conf,lev:op.lev,pos:op.pos,ts:now,regime:op.regime});
      if(st.closed.length>2000) st.closed.pop();
      st.openTrades.splice(i,1); st.exitReasons[why]=(st.exitReasons[why]||0)+1;
      if(pnlAbs<0){ if(++st.cbLoss[sym]>=CFG.CB_N){st.cbEnd[sym]=now+CFG.CB_WAIT;st.cbLoss[sym]=0;} }else{st.cbLoss[sym]=0;}
      st.streak=pnlAbs>0?(st.streak<0?1:st.streak+1):(st.streak>0?-1:st.streak-1);
      st.maxWinStreak=Math.max(st.maxWinStreak,st.streak); st.maxLossStreak=Math.min(st.maxLossStreak,st.streak);
      st.balHist.push({t:st.balHist.length,b:st.balance}); if(st.balHist.length>600) st.balHist.shift();
      if(op.featuresAtEntry){
        const reward = pnlAbs > 0 ? 1.4 : -0.65; const lr = CFG.ML_LR * reward;
        Object.keys(st.mlWeights).forEach(k => { if (k !== "bias") st.mlWeights[k] = st.mlWeights[k] * CFG.ML_DECAY + op.featuresAtEntry[k] * lr; });
        st.mlWeights.bias = st.mlWeights.bias * CFG.ML_DECAY + lr * 0.28;
      }
      if(op.rlStateAtEntry){
        const reward = pnlAbs > 0 ? (pnlAbs / op.pos / op.lev) * 100 : (pnlAbs / op.pos / op.lev) * 100;
        updateRLWeights(st.rlWeights, op.rlStateAtEntry, reward, CFG.RL_LR);
      }
    }
  }

  if(buf.length<CFG.WARMUP) return;
  if(now-st.sigTs[sym]<CFG.COOLDOWN_BASE) return;
  if(now<st.cbEnd[sym]) return;
  if(st.openTrades.length>=CFG.MAX_OPEN) return;
  if(st.openTrades.some((o: any)=>o.sym===sym)) return;
  const sig=ensembleSignal(sym,st); st.sigs[sym]=sig;
  if(sig.conf >= (CFG.REGIME_THRESH[sig.regime] || CFG.THRESH_BASE) && sig.dir!==0 && Math.abs(sig.zscore) >= CFG.MIN_ZSCORE){
    const side=sig.dir>0?"Long":"Short"; const regimeLevMult = sig.regime==="trend" ? 1.20 : 1.0;
    const lev=Math.max(CFG.MIN_LEV,Math.min(CFG.MAX_LEV,Math.floor(CFG.LEV_BASE*sig.conf*regimeLevMult)));
    const dd = (st.peak - st.balance) / st.peak || 0;
    let pos = kellySize(st.closed, st.balance, sig.regime, st.atr[sym], st.streak, dd, sig.conf, sig.mlLogit);
    const rlState = { regimeTrend: sig.regime==="trend" ? 1 : 0, regimeVol: sig.regime==="volatile" ? 1 : 0, conf: sig.conf, mlLogit: sig.mlLogit, streak: st.streak, drawdown: dd };
    const rlMult = rlScalePosition(rlState, st.rlWeights); pos = Math.min(CFG.MAX_POS, pos * rlMult);
    const rRR = CFG.RR_MULT[sig.regime] || 4.5; const slPct = Math.max(0.0012, CFG.SL_MULT * (st.atr[sym]||0.001));
    const tpPct = slPct * rRR; const trailPct = slPct * CFG.TRAIL_MULT;
    const tActPct = slPct * CFG.TRAIL_ACTIVATE_MULT; const beTrigPct = slPct * CFG.BE_TRIG_MULT;
    const maxHold = sig.regime==="trend" ? CFG.MAX_HOLD.trend : CFG.MAX_HOLD.other;
    const initTrail=side==="Long"?price*(1-slPct):price*(1+slPct);
    st.openTrades.push({ id:"" + sym + "-" + now, sym, side, entry:price, lev, pos, conf:sig.conf, ts:now, peak:price, trough:price, trail:initTrail, trailActive:false, be:false, partialTaken:false, slPct, tpPct, trailPct, trailActPct:tActPct, beTrigPct, partialR:CFG.PARTIAL_R, maxHold, regime:sig.regime, featuresAtEntry: {zscore:sig.zscore,accel:sig.accel,tc:sig.tc,bConf:sig.bConf||0,regimeTrend:rlState.regimeTrend,regimeVol:rlState.regimeVol,vol:sig.vol*-1}, rlStateAtEntry: rlState });
    st.sigTs[sym]=now;
  }
};

// ═══════════════════════════════════════════════════════════════
//  METRICS & SIM TICK
// ═══════════════════════════════════════════════════════════════
const computeMetrics = (st: any) => {
  const base={balance:st.balance,pnl:0,winPct:0,wl:0,sharpe:0,dd:0,n:0, avgHold:0,ev:0,pf:0,calmar:0,tpm:0,streak:st.streak, maxWS:st.maxWinStreak,maxLS:st.maxLossStreak,rolling20:null, symPnl:Object.fromEntries(SYMS.map(s=>[s,0]))};
  const{closed:tr}=st, n=tr.length; if(!n) return base;
  const pnls=tr.map((t: any)=>t.pnlAbs); const wins=pnls.filter((p: any)=>p>0),loss=pnls.filter((p: any)=>p<=0);
  const total=pnls.reduce((a: any,b: any)=>a+b,0), avg=total/n;
  const std=Math.sqrt(pnls.reduce((a: any,p: any)=>a+(p-avg)**2,0)/n)+1e-9;
  const winRate=wins.length/n; const avgW=wins.length?wins.reduce((a: any,b: any)=>a+b,0)/wins.length:0;
  const avgL=loss.length?Math.abs(loss.reduce((a: any,b: any)=>a+b,0)/loss.length):1e-8;
  const pf=Math.abs(loss.reduce((a: any,b: any)=>a+b,0))>0?wins.reduce((a: any,b: any)=>a+b,0)/Math.abs(loss.reduce((a: any,b: any)=>a+b,0)):wins.reduce((a: any,b: any)=>a+b,0)>0?99:0;
  let sharpe=0; if(n>1&&st.firstTs){const h=(Date.now()-st.firstTs)/3.6e6;if(h>0)sharpe=(avg/std)*Math.sqrt((n/h)*8760);}
  let calmar=0; if(st.maxDD>0&&st.firstTs){const h=(Date.now()-st.firstTs)/3.6e6;if(h>0){const ar=(total/CFG.BAL0)*(8760/h);calmar=ar/st.maxDD;}}
  let tpm=0; if(n>=2&&st.firstTs){const mn=(Date.now()-st.firstTs)/60000;tpm=n/Math.max(0.01,mn);}
  const r20=tr.slice(0,Math.min(20,n));
  const rolling20=r20.length>=3?{n:r20.length,winPct:r20.filter((t: any)=>t.pnlAbs>0).length/r20.length*100, pnl:r20.reduce((a: any,t: any)=>a+t.pnlAbs,0)}:null;
  return{balance:st.balance,pnl:total,n,symPnl:Object.fromEntries(SYMS.map(s=>[s,tr.filter((t: any)=>t.sym===s).reduce((a: any,t: any)=>a+t.pnlAbs,0)])),sharpe,ev:winRate*avgW-(1-winRate)*avgL,pf,calmar,tpm, streak:st.streak,maxWS:st.maxWinStreak,maxLS:st.maxLossStreak,rolling20, winPct:winRate*100,wl:avgW/avgL,dd:st.maxDD,avgHold:tr.reduce((a: any,t: any)=>a+t.hold,0)/n};
};

const simTick = (sym: string, st: any) => {
  const r=st.regimes[sym]; if(--r.ttl<=0){
    r.trend=Math.random()<0.50; r.drift=r.trend?(Math.random()-0.5)*0.0065:0;
    r.volMult=r.trend?0.58:1.42; r.ttl=r.trend?110+Math.floor(Math.random()*270):25+Math.floor(Math.random()*110);
  }
  const anchor=(st.realPrices[sym]>0&&st.latestPrices[sym]>0)?Math.log(st.realPrices[sym]/st.latestPrices[sym])*0.14:0;
  r.vol=0.96*r.vol+0.04*(BVOL[sym]*r.volMult);
  let corrTerm=0; if(sym!=="BTC-USD"&&st.prevBtcReturn!=null) corrTerm=(CORR[sym]||0)*st.prevBtcReturn*0.57;
  const ret=anchor+r.drift+corrTerm+Math.sqrt(Math.max(0,1-(CORR[sym]||0)**2))*r.vol*gauss();
  if(sym==="BTC-USD") st.prevBtcReturn=ret; return st.latestPrices[sym]*Math.exp(ret);
};

const fetchPrices = async () => {
  try{
    const r=await fetch("https://api.anthropic.com/v1/messages",{method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({model:"claude-sonnet-4-6",max_tokens:512, tools:[{type:"web_search_20250305",name:"web_search"}],
        messages:[{role:"user",content:"Search for the current live prices of Bitcoin, Ethereum, Solana in USD. Reply ONLY with this JSON: {\"BTC\":PRICE,\"ETH\":PRICE,\"SOL\":PRICE}"}]})});
    if(r.ok){
      const d=await r.json(); let p=((): any => {
        const text=(d.content||[]).filter((b:any)=>b.type==="text").map((b:any)=>b.text||"").join(" ");
        const clean=text.replace(/\s+/g,"");
        const b=clean.match(/"BTC"\s*:\s*([\d,]+\.?\d*)/)?.[1]?.replace(/,/g,"");
        const e=clean.match(/"ETH"\s*:\s*([\d,]+\.?\d*)/)?.[1]?.replace(/,/g,"");
        const s=clean.match(/"SOL"\s*:\s*([\d,]+\.?\d*)/)?.[1]?.replace(/,/g,"");
        if(b&&e&&s&&+b>100&&+e>1&&+s>0.01) return{"BTC-USD":+b,"ETH-USD":+e,"SOL-USD":+s};
        return null;
      })();
      if(p) return{...p,src:"Claude+Search"};
    }
  }catch{}
  try{
    const ps=await Promise.all([["BTCUSDT","BTC-USD"],["ETHUSDT","ETH-USD"],["SOLUSDT","SOL-USD"]].map(([s,k])=>fetch("https://api.binance.com/api/v3/ticker/price?symbol=" + s, {signal:AbortSignal.timeout(3000)}).then(r=>r.json()).then(d=>({k,p:+d.price}))));
    const pr=Object.fromEntries(ps.map(x=>[x.k,x.p])); if(Object.values(pr).every(p=>p>0)) return{...pr,src:"Binance"};
  }catch{}
  return null;
};

const mkState = () => ({
  latestPrices:{...SEEDS}, realPrices:Object.fromEntries(SYMS.map(s=>[s,0])),
  buffers:Object.fromEntries(SYMS.map(s=>[s,[]])), chartData:Object.fromEntries(SYMS.map(s=>[s,[]])),
  balHist:[{t:0,b:CFG.BAL0}], openTrades:[], closed:[], balance:CFG.BAL0, peak:CFG.BAL0, maxDD:0, firstTs:null, totalTicks:0, prevBtcReturn:null, streak:0, maxWinStreak:0, maxLossStreak:0,
  atr:Object.fromEntries(SYMS.map(s=>[s,CFG.SL_MULT*0.001])), sigTs:Object.fromEntries(SYMS.map(s=>[s,0])),
  cbLoss:Object.fromEntries(SYMS.map(s=>[s,0])), cbEnd:Object.fromEntries(SYMS.map(s=>[s,0])), ticks:Object.fromEntries(SYMS.map(s=>[s,0])),
  sigs:Object.fromEntries(SYMS.map(s=>[s,{dir:0,conf:0,zscore:0,vol:0,tc:0,accel:0,bDir:0,bConf:0,regime:"warmup",mlConf:0,mlLogit:0}])),
  regimes:Object.fromEntries(SYMS.map(s=>[s,{drift:0,vol:BVOL[s],volMult:1,trend:false,ttl:0}])),
  exitReasons:{TP:0,SL:0,trail:0,"t/o":0,PARTIAL:0},
  mlWeights: {zscore:0.48, accel:1.18, tc:0.31, bConf:0.42, regimeTrend:0.71, regimeVol:-0.31, vol:-0.19, bias:-0.09},
  rlWeights: {bias:0.12, regimeTrend:0.68, regimeVol:-0.41, conf:1.25, mlLogit:0.92, streak:0.35, drawdown:-1.18}
});

const fP =(sym: string,p: number)=>!p?"$0.00":sym==="BTC-USD"?"$" + Math.round(p).toLocaleString() : sym==="ETH-USD"?"$" + (p||0).toFixed(2) : "$" + (p||0).toFixed(3);
const fY =(sym: string,v: number)=>sym==="BTC-USD"?"" + (v/1000).toFixed(1) + "k" : sym==="ETH-USD"?v.toFixed(0):v.toFixed(2);
const gc =(v: number,a="#14F195",b="#ff3366")=>v>=0?a:b;
const sgn=(v: number,d=2)=>"" + (v>=0?"+":"") + "$" + Math.abs(v).toFixed(d);
const fTs=(ts: number)=>ts?new Date(ts).toLocaleTimeString():"-";
const pct=(v: number,d=1)=>"" + (v>=0?"+":"") + v.toFixed(d) + "%";

const RB: Record<string, any> = {
  trend:  {bg:"#081a0a",fg:"#14F195",bdr:"#14F195",icon:"📈",lbl:"TREND"},
  range:  {bg:"#0d0d1c",fg:"#627eea",bdr:"#627eea",icon:"📊",lbl:"RANGE"},
  volatile:{bg:"#1a0808",fg:"#ff3366",bdr:"#ff3366",icon:"⚡",lbl:"VOLAT"},
  warmup: {bg:"#0d0d14",fg:"#555",   bdr:"#333",   icon:"◌", lbl:"WARMUP"},
};

export default function Z9OracleSim() {
  const stRef=useRef(mkState()); const simRef=useRef<any[]>([]); const pollRef=useRef<any>(null);
  const saveRef=useRef<any>(null); const selRef=useRef("BTC-USD"); const tickHzRef=useRef(0); const prevTick=useRef(Date.now());
  const [disp,setDisp]=useState<any>(null); const [selSym,setSelSym]=useState("BTC-USD"); const [tab,setTab]=useState("live");
  const [mode,setMode]=useState("STARTING"); const [pSrc,setPSrc]=useState("seed"); const [tickMs,setTickMs]=useState(CFG.TICK_MS); const [fetching,setFetching]=useState(false);

  const chooseSym = useCallback((s: string) => { selRef.current = s; setSelSym(s); }, []);
  const startSim = useCallback((ms: number) => {
    simRef.current.forEach(clearInterval);
    simRef.current = [
      setInterval(() => { try{ const p=simTick("BTC-USD",stRef.current); processPrice("BTC-USD",p,Date.now(),stRef.current); const n=Date.now(),dt=n-prevTick.current; if(dt>0) tickHzRef.current=Math.round(5500/dt); prevTick.current=n; }catch{} }, ms),
      setInterval(() => { try{const p=simTick("ETH-USD",stRef.current);processPrice("ETH-USD",p,Date.now(),stRef.current);}catch{} }, ms+7),
      setInterval(() => { try{const p=simTick("SOL-USD",stRef.current);processPrice("SOL-USD",p,Date.now(),stRef.current);}catch{} }, ms+15),
    ];
  }, []);

  const setSpeed=useCallback((ms:number)=>{setTickMs(ms);startSim(ms);},[startSim]);
  const doReset=useCallback(async()=>{try{await (window as any).storage.delete(STO);}catch{}stRef.current=mkState();startSim(tickMs);setMode("SIM");setPSrc("seed");},[startSim,tickMs]);
  const exportCSV=useCallback(()=>{
    const rows=stRef.current.closed.map((t:any)=>[new Date(t.ts).toISOString(),t.sym,t.side,t.entry.toFixed(4),t.exitP.toFixed(4),(t.pnlPct*100).toFixed(4),t.pnlAbs.toFixed(2),t.hold.toFixed(2),t.why,(t.conf*100).toFixed(1),t.lev,(t.pos||0).toFixed(0)].join(","));
    const a=Object.assign(document.createElement("a"),{href:URL.createObjectURL(new Blob(["ts,sym,side,entry,exit,pnl%,pnl$,hold_s,exit,conf%,lev,pos\n"+rows.join("\n")],{type:"text/csv"})),download:"z9_ultra_trades.csv"});a.click();
  },[]);

  const doFetch=async()=>{setFetching(true);const res=await fetchPrices();setFetching(false);if(!res)return;const{src,...prices}=res;SYMS.forEach(s=>{if((prices as any)[s]>0){stRef.current.realPrices[s]=(prices as any)[s];stRef.current.latestPrices[s]=(prices as any)[s];}});setPSrc(src||"?");setMode("LIVE");};

  useEffect(()=>{
    (async()=>{try{const sv=await (window as any).storage.get(STO);if(sv){const{trades,balance,peak,firstTs,maxDD,streak}=JSON.parse(sv.value);if(trades?.length){const st=stRef.current;st.closed=trades;st.balance=balance??CFG.BAL0;st.peak=peak??CFG.BAL0;st.firstTs=firstTs??null;st.maxDD=maxDD??0;st.streak=streak??0;let b=CFG.BAL0;st.balHist=[{t:0,b}];[...trades].reverse().forEach((t:any)=>{b+=t.pnlAbs;st.balHist.push({t:st.balHist.length,b});});trades.forEach((t:any)=>{st.exitReasons[t.why as keyof typeof st.exitReasons]=(st.exitReasons[t.why as keyof typeof st.exitReasons]||0)+1;});let ks=0,mx=0,mn=0;[...trades].reverse().forEach((t:any)=>{ks=t.pnlAbs>0?(ks<0?1:ks+1):(ks>0?-1:ks-1);mx=Math.max(mx,ks);mn=Math.min(mn,ks);});st.maxWinStreak=mx;st.maxLossStreak=mn;}}}catch{}})();
    saveRef.current=setInterval(async()=>{const st=stRef.current;if(!st.closed.length)return;try{await (window as any).storage.set(STO,JSON.stringify({trades:st.closed.slice(0,800),balance:st.balance,peak:st.peak,firstTs:st.firstTs,maxDD:st.maxDD,streak:st.streak}));}catch{}},5000);return()=>clearInterval(saveRef.current);
  },[]);

  useEffect(()=>{startSim(CFG.TICK_MS);setMode("SIM");doFetch();pollRef.current=setInterval(doFetch,60000);return()=>{simRef.current.forEach(clearInterval);clearInterval(pollRef.current);};},[]);

  useEffect(()=>{
    const t=setInterval(()=>{
      const st=stRef.current; const m=computeMetrics(st);
      const open=st.openTrades.map((op:any)=>{const cur=st.latestPrices[op.sym]||op.entry;const pnlPct=op.side==="Long"?(cur-op.entry)/op.entry:(op.entry-cur)/op.entry;return{...op,cur,pnlPct,pnlAbs:op.pos*pnlPct*op.lev,holdS:(Date.now()-op.ts)/1000};});
      const cRLM=rlScalePosition({regimeTrend:st.sigs[selRef.current]?.regime==="trend"?1:0,regimeVol:st.sigs[selRef.current]?.regime==="volatile"?1:0,conf:st.sigs[selRef.current]?.conf||0.5,mlLogit:st.sigs[selRef.current]?.mlLogit||0,streak:st.streak,drawdown:(st.peak-st.balance)/st.peak||0},st.rlWeights);
      setDisp({prices:{...st.latestPrices},chart:[...(st.chartData[selRef.current]||[])],balHist:st.balHist.slice(-400).map((x,i)=>({...x,t:i})),metrics:m,open,closed:st.closed.slice(0,30),allClosed:st.closed,sigs:{...st.sigs},bufLen:Object.fromEntries(SYMS.map(s=>[s,st.buffers[s].length])),cbActive:Object.fromEntries(SYMS.map(s=>[s,Date.now()<st.cbEnd[s]])),exitReasons:{...st.exitReasons},totalTicks:st.totalTicks,hz:tickHzRef.current,mlWeights:{...st.mlWeights},rlWeights:{...st.rlWeights},currentRLMult:cRLM,currentKelly:kellySize(st.closed,st.balance,st.sigs[selRef.current]?.regime||"warmup",st.atr[selRef.current],st.streak,(st.peak-st.balance)/st.peak,st.sigs[selRef.current]?.conf||0.5,st.sigs[selRef.current]?.mlLogit||0)/st.balance});
    },400);return()=>clearInterval(t);
  },[]);

  if(!disp) return(<div style={{background:"#03030a",color:"#333",height:"100vh",display:"flex",alignItems:"center",justifyContent:"center",fontFamily:"monospace",letterSpacing:4}}>Z9 ORACLE ULTRA v16.0…</div>);
  const{prices,chart,balHist,metrics:m,open,closed,allClosed,sigs,bufLen,cbActive,exitReasons,totalTicks,hz,currentRLMult,currentKelly}=disp;
  const confW = Math.floor((sigs[selSym]?.conf || 0) * 100);
  const cp=chart.map((d:any)=>d.price).filter(Boolean); const yMin=cp.length?Math.min(...cp)*0.9995:SEEDS[selSym]*0.999; const yMax=cp.length?Math.max(...cp)*1.0005:SEEDS[selSym]*1.001; const selC=CLR[selSym];
  const tEx = (Object.values(exitReasons).reduce((a: any, b: any) => (Number(a) || 0) + (Number(b) || 0), 0) as number) || 1;
 const BG="#03030a",CARD="#09090f",BDR="#111224",DIM="#2e2f4a",TXT="#b8b8d0";
  const card=(ex={})=>({background:CARD,border:"1px solid " + BDR,borderRadius:6,padding:"8px 10px",...ex});
  const lbl={color:DIM,fontSize:9,letterSpacing:1.5,marginBottom:3,textTransform:"uppercase" as const};
  const streakC=m.streak>0?"#14F195":m.streak<0?"#ff3366":DIM; const streakLbl=m.streak===0?"─":m.streak>0?"+"+m.streak+"W":Math.abs(m.streak)+"L";

  return(
    <div style={{background:BG,color:TXT,fontFamily:"'Courier New',monospace",fontSize:11,minHeight:"100vh",padding:10,boxSizing:"border-box"}}>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",borderBottom:"1px solid "+BDR,paddingBottom:7,marginBottom:8}}>
        <div><span style={{fontSize:13,fontWeight:900,letterSpacing:3,color:"#4030aa"}}>SKYRMATRON</span><span style={{fontSize:13,fontWeight:900,letterSpacing:3,color:"#fff"}}> Z9</span><span style={{fontSize:8,color:DIM,marginLeft:6}}>v16.0 ULTRA · ML-FUSION v2 · RL ADAPTIVE</span></div>
        <div style={{display:"flex",gap:7,alignItems:"center",flexWrap:"wrap"}}><span style={{color:streakC,fontWeight:700,fontSize:10}}>{streakLbl}</span><span style={{color:totalTicks>0?"#14F195":DIM,fontSize:9}}>⬤ {(totalTicks||0).toLocaleString()}{hz>0?" · "+hz+"Hz":""}</span><span style={{color:"#14F195",fontSize:9}}>● {pSrc}</span><span style={{color:mode==="LIVE"?"#14F195":"#6050cc",fontSize:9,border:"1px solid "+(mode==="LIVE"?"#14F195":"#6050cc"),padding:"1px 6px",borderRadius:3}}>{mode==="LIVE"?"⬤ LIVE":"⬤ GBM"}</span><button onClick={doFetch} style={{background:"transparent",border:"1px solid "+DIM,color:DIM,borderRadius:3,padding:"1px 7px",fontSize:9,cursor:"pointer"}}>↻</button></div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(7,1fr)",gap:4,marginBottom:4}}>
        {[["Balance","$"+m.balance.toFixed(2),gc(m.balance-CFG.BAL0)],["PnL",sgn(m.pnl),gc(m.pnl)],["EV/trade",sgn(m.ev),gc(m.ev)],["Win%",m.winPct.toFixed(1)+"%",m.winPct>20?"#14F195":"#ff3366"],["W/L",m.wl.toFixed(2),gc(m.wl-1)],["Sharpe",m.sharpe.toFixed(2),gc(m.sharpe)],["MaxDD",(m.dd*100).toFixed(2)+"%","#ff3366"]].map(([l,v,c]:any)=>(<div key={l} style={card({padding:"6px 8px"})}><div style={lbl}>{l}</div><div style={{color:c,fontSize:11,fontWeight:700}}>{v}</div></div>))}
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:4,marginBottom:8}}>
        {[["Profit Factor",m.pf>0?m.pf.toFixed(2):"─",m.pf>1?"#14F195":m.pf>0?"#ff9900":"#ff3366"],["Calmar",m.calmar?m.calmar.toFixed(2):"─",gc(m.calmar)],["Trades/min",m.tpm.toFixed(1),"#627eea"],["Streak",streakLbl,streakC],["R20 Win%",m.rolling20?m.rolling20.winPct.toFixed(0)+"% ("+m.rolling20.n+")t":"─",m.rolling20?m.rolling20.winPct>20?"#14F195":"#ff3366":DIM]].map(([l,v,c]:any)=>(<div key={l} style={card({padding:"6px 8px"})}><div style={lbl}>{l}</div><div style={{color:c,fontSize:11,fontWeight:700}}>{v}</div></div>))}
      </div>
      {tab==="live" && (<>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:6,marginBottom:8}}>
          {SYMS.map(s=>{ const sg=sigs[s],rb=RB[sg.regime]||RB.warmup; return(<div key={s} onClick={()=>chooseSym(s)} style={{...card(),cursor:"pointer",borderColor:selSym===s?CLR[s]:BDR,background:selSym===s?"#0c0c1a":CARD}}><div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}><span style={{color:CLR[s],fontSize:9,letterSpacing:1.5}}>{s}</span><span style={{background:rb.bg,color:rb.fg,border:"1px solid "+rb.bdr,padding:"1px 5px",borderRadius:3,fontSize:7}}>{rb.icon} {rb.lbl}</span></div><div style={{color:"#fff",fontSize:15,fontWeight:700,marginBottom:4}}>{fP(s,prices[s])}</div><div style={{display:"flex",justifyContent:"space-between",fontSize:9}}><span style={{color:sg.dir>0?"#14F195":sg.dir<0?"#ff3366":DIM,fontWeight:700}}>{bufLen[s]<CFG.WARMUP?`WU ${bufLen[s]}/${CFG.WARMUP}`:sg.dir>0?"▲LONG":sg.dir<0?"▼SHORT":"─FLAT"}</span><span style={{color:sg.conf>=CFG.THRESH_BASE?"#14F195":DIM}}>z={sg.zscore?.toFixed(2)||"─"} {(sg.conf*100).toFixed(0)}%</span></div>{cbActive[s]&&<div style={{color:"#ff9900",fontSize:8,marginTop:2}}>⚡ CB ACTIVE</div>}</div>);})}
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 340px",gap:4}}>
          <div style={{display:"flex",flexDirection:"column",gap:4}}>
            <div style={card({height:220})}><div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}><span style={{color:selC,fontSize:9}}>{selSym} · {chart.length} ticks</span><div style={{color:selC,fontWeight:700,fontSize:11}}>{fP(selSym,prices[selSym])}</div></div><ResponsiveContainer width="100%" height="90%"><LineChart data={chart}><CartesianGrid strokeDasharray="2 2" stroke="#111"/><XAxis dataKey="t" hide/><YAxis domain={[yMin,yMax]} hide/><Tooltip content={()=><div/>}/><Line type="monotone" dataKey="price" stroke={selC} dot={false} strokeWidth={1.5} isAnimationActive={false}/>{open.filter((o:any)=>o.sym===selSym).map((o:any)=>(<ReferenceLine key={o.id} y={o.entry} stroke={o.side==="Long"?"#14F195":"#ff3366"} strokeDasharray="3 3"/>))}</LineChart></ResponsiveContainer></div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4}}>
              <div style={card({height:140})}><div style={lbl}>Signal: {selSym}</div><div style={{display:"flex",alignItems:"center",gap:10,marginTop:5}}><div style={{width:40,height:40,borderRadius:20,background:RB[sigs[selSym]?.regime]?.bg,display:"flex",alignItems:"center",justifyContent:"center",fontSize:16,border:"2px solid "+RB[sigs[selSym]?.regime]?.bdr}}>{RB[sigs[selSym]?.regime]?.icon}</div><div><div style={{fontSize:12,fontWeight:900,color:RB[sigs[selSym]?.regime]?.fg}}>{RB[sigs[selSym]?.regime]?.lbl}</div><div style={{fontSize:8,color:DIM}}>CONF: {(sigs[selSym]?.conf*100).toFixed(1)}%</div></div><div style={{marginLeft:"auto",textAlign:"right"}}><div style={{fontSize:14,fontWeight:900,color:gc(sigs[selSym]?.dir)}}>{sigs[selSym]?.dir>0?"BUY":sigs[selSym]?.dir<0?"SELL":"WAIT"}</div><div style={{fontSize:8,color:DIM}}>Z:{sigs[selSym]?.zscore.toFixed(2)} TC:{sigs[selSym]?.tc}</div></div></div><div style={{position:"relative",background:"#0d0d18",borderRadius:2,height:3,marginTop:10,marginBottom:4}}><div style={{background:sigs[selSym]?.conf>=CFG.THRESH_BASE?CLR[selSym]:DIM,width:confW+"%",height:"100%",borderRadius:2}}/><div style={{position:"absolute",left:CFG.THRESH_BASE*100+"%",top:-1,width:1,height:5,background:"#ff3366"}}/></div><div style={{fontSize:8,color:DIM}}>BTC-Lead: <span style={{color:gc(sigs[selSym]?.bDir)}}>{sigs[selSym]?.bDir>0?"BULL":"BEAR"}</span> ({(sigs[selSym]?.bConf*100).toFixed(0)}%)</div></div>
              <div style={card({height:140})}><div style={lbl}>Equity Curve</div><ResponsiveContainer width="100%" height="80%"><LineChart data={balHist}><Line type="monotone" dataKey="b" stroke="#6050cc" dot={false} strokeWidth={2} isAnimationActive={false}/><YAxis domain={["auto","auto"]} hide/></LineChart></ResponsiveContainer></div>
            </div>
          </div>
          <div style={{display:"flex",flexDirection:"column",gap:4}}>
            <div style={card()}><div style={lbl}>Ultra Intelligence v2</div><div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,fontSize:9}}><div>RL mult: <span style={{color:"#14F195",fontWeight:700}}>{currentRLMult.toFixed(3)}×</span></div><div>Kelly f: <span style={{color:"#14F195"}}>{currentKelly.toFixed(4)}</span></div><div>ML logit: <span style={{color:"#627eea"}}>{sigs[selSym]?.mlLogit?.toFixed(2)}</span></div><div>ML conf: <span style={{color:"#627eea"}}>{(sigs[selSym]?.mlConf*100).toFixed(0)}%</span></div></div></div>
            <div style={card({flex:1})}><div style={lbl}>Open Positions ({open.length})</div><div style={{marginTop:5,maxHeight:140,overflowY:"auto"}}>{open.map((o:any)=>(<div key={o.id} style={{borderBottom:"1px solid #111",padding:"4px 0",display:"flex",justifyContent:"space-between",alignItems:"center"}}><div><div style={{fontSize:9,fontWeight:700,color:CLR[o.sym]}}>{o.sym} <span style={{color:gc(o.side==="Long"?1:-1)}}>{o.side}</span></div><div style={{fontSize:8,color:DIM}}>{o.lev}x @ {fP(o.sym,o.entry)}</div></div><div style={{textAlign:"right"}}><div style={{fontSize:10,fontWeight:700,color:gc(o.pnlAbs)}}>{sgn(o.pnlAbs)}</div><div style={{fontSize:8,color:DIM}}>{o.holdS.toFixed(0)}s / {pct(o.pnlPct*o.lev)}</div></div></div>))}{!open.length&&<div style={{color:DIM,fontSize:9,textAlign:"center",marginTop:20}}>No Active Trades</div>}</div></div>
            <div style={card({flex:1})}><div style={lbl}>Exit Breakdown</div>{[["TP","#14F195"],["trail","#627eea"],["SL","#ff3366"],["t/o","#666"],["PARTIAL","#4030aa"]].map(([k,c])=>{ const v = Number(exitReasons[k]) || 0; const p = v / tEx; const exitW = Math.floor(p * 100); return(<div key={k} style={{marginBottom:5}}><div style={{display:"flex",justifyContent:"space-between",marginBottom:1}}><span style={{color:c,fontSize:8}}>{k}</span><span style={{color:DIM,fontSize:8}}>{v} · {(p*100).toFixed(0)}%</span></div><div style={{background:"#0d0d18",borderRadius:2,height:2}}><div style={{background:c,width:exitW+"%",height:"100%",borderRadius:2}}/></div></div>);})}</div>
          </div>
        </div>
      </>)}
      {tab==="log" && (<div style={card({padding:"10px"})}><div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}><div style={lbl}>Trade History Log ({allClosed.length})</div><div style={{display:"flex",gap:10,alignItems:"center"}}><span style={{color:gc(m.pnl),fontWeight:700}}>{sgn(m.pnl)}</span><button onClick={exportCSV} style={{background:"#111224",border:"1px solid "+DIM,color:TXT,fontSize:8,padding:"2px 8px",borderRadius:3,cursor:"pointer"}}>EXPORT CSV</button></div></div><div style={{overflowY:"auto",maxHeight:"calc(100vh - 280px)"}}><table style={{width:"100%",borderCollapse:"collapse",fontSize:9}}><thead><tr style={{textAlign:"left",color:DIM,borderBottom:"1px solid "+BDR}}><th style={{padding:4}}>TIME</th><th style={{padding:4}}>SYM</th><th style={{padding:4}}>SIDE</th><th style={{padding:4}}>ENTRY</th><th style={{padding:4}}>EXIT</th><th style={{padding:4}}>LEV</th><th style={{padding:4}}>PnL%</th><th style={{padding:4}}>PnL$</th><th style={{padding:4}}>WHY</th></tr></thead><tbody>{allClosed.slice(0,100).map((t:any,i:number)=>(<tr key={i} style={{borderBottom:"1px solid #080810"}}><td style={{padding:4,color:DIM}}>{new Date(t.ts).toLocaleTimeString()}</td><td style={{padding:4,color:CLR[t.sym],fontWeight:700}}>{t.sym.split("-")[0]}</td><td style={{padding:4,color:gc(t.side==="Long"?1:-1)}}>{t.side}</td><td style={{padding:4}}>{fP(t.sym,t.entry)}</td><td style={{padding:4}}>{fP(t.sym,t.exitP)}</td><td style={{padding:4}}>{t.lev}x</td><td style={{padding:4,color:gc(t.pnlPct)}}>{pct(t.pnlPct*t.lev)}</td><td style={{padding:4,color:gc(t.pnlAbs),fontWeight:700}}>{sgn(t.pnlAbs)}</td><td style={{padding:4,color:DIM}}>{t.why}</td></tr>))}</tbody></table></div></div>)}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr auto",gap:5,alignItems:"center",marginTop:8}}>{SYMS.map(s=>(<div key={s} style={{...card(),display:"flex",justifyContent:"space-between"}}><span style={{color:CLR[s],fontSize:9}}>{s}</span><span style={{color:gc(m.symPnl[s]||0),fontWeight:700,fontSize:10}}>{sgn(m.symPnl[s]||0)}</span></div>))}<div style={{display:"flex",gap:4,marginLeft:10}}>{[[60,"1×"],[30,"2×"],[12,"5×"],[6,"10×"]].map(([ms,l]:any)=>(<button key={ms} onClick={()=>setSpeed(ms)} style={{background:tickMs===ms?"#111224":"transparent",border:"1px solid "+(tickMs===ms?"#6050cc":BDR),color:tickMs===ms?"#9080ee":DIM,borderRadius:4,padding:"4px 7px",fontSize:9,cursor:"pointer"}}>{l}</button>))}<button onClick={doReset} style={{background:"transparent",border:"1px solid #ff3366",color:"#ff3366",borderRadius:4,padding:"4px 8px",fontSize:9,cursor:"pointer",marginLeft:10}}>RESET</button></div></div>
      <div style={{display:"flex",gap:4,marginTop:8}}>{[["live","LIVE VIEW"],["log","HISTORY LOG"]].map(([t,l])=>(<button key={t} onClick={()=>setTab(t)} style={{background:tab===t?"#111224":"transparent",border:"1px solid "+(tab===t?"#6050cc":BDR),color:tab===t?"#9080ee":DIM,borderRadius:4,padding:"4px 14px",fontSize:9,cursor:"pointer",letterSpacing:1}}>{l}</button>))}</div>
    </div>
  );
}
