import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
         ResponsiveContainer, ReferenceLine } from "recharts";

// ══════════════════════════════════════════════════════════════════════════════
//  Z9 Oracle v16.5 — HFT Real-Time Build (Simulator Loop Removed)
// ══════════════════════════════════════════════════════════════════════════════

const SYMS  = ["BTC-USD","ETH-USD","SOL-USD"];
const CLR: Record<string, string> = {"BTC-USD":"#f7931a","ETH-USD":"#627eea","SOL-USD":"#14F195"};

// ══════════════════════════════════════════════════════════════════════════════
//  FORMATTERS + UI CONSTANTS
// ══════════════════════════════════════════════════════════════════════════════
const fP  =(sym: string,p: number)=>!p?"$0.00":sym==="BTC-USD"?`$${p.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`:sym==="ETH-USD"?`$${p.toFixed(2)}`:`$${p.toFixed(3)}`;
const fY  =(sym: string,v: number)=>sym==="BTC-USD"?`${(v/1000).toFixed(1)}k`:sym==="ETH-USD"?v.toFixed(0):v.toFixed(2);
const gc  =(v: number,a="#14F195",b="#ff3366")=>v>=0?a:b;
const sgn =(v: number,d=2)=>`${v>=0?"+":""}${v.toFixed(d)}`;
const fTs =(ts: string)=>ts?new Date(ts).toLocaleTimeString():"-";

const RB: Record<string, any>  ={
  trend:  {bg:"#081a0a",fg:"#14F195",bdr:"#14F195",icon:"↑",lbl:"TREND"},
  range:  {bg:"#0d0d1c",fg:"#627eea",bdr:"#627eea",icon:"≈",lbl:"RANGE"},
  volatile:{bg:"#1a0808",fg:"#ff3366",bdr:"#ff3366",icon:"!",lbl:"VOLAT"},
  warmup: {bg:"#0d0d14",fg:"#555",   bdr:"#333",   icon:"◌",lbl:"WARMUP"},
};

const BG="#03030a",CARD="#09090f",BDR="#111224",DIM="#2e2f4a",TXT="#b8b8d0";

export default function Z9OracleDashboard(){
  const [metrics, setMetrics] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [selSym,  setSelSym]  = useState("BTC-USD");
  const [chartData, setChartData] = useState<Record<string, any[]>>({
    "BTC-USD": [], "ETH-USD": [], "SOL-USD": []
  });

  const fetchSync = useCallback(async () => {
    try {
      const res = await fetch('/real_time_metrics.json?' + Date.now());
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
        
        // Update historical chart data from live feed
        if (data.live_prices) {
          setChartData(prev => {
            const next = {...prev};
            Object.entries(data.live_prices).forEach(([s, p]) => {
              if (!next[s]) next[s] = [];
              // Only add if price changed or every 1s
              const lastP = next[s][next[s].length-1]?.price;
              if (p !== lastP || next[s].length === 0) {
                next[s] = [...next[s], {t: Date.now(), price: p as number}].slice(-120);
              }
            });
            return next;
          });
        }
      }
      
      const resLog = await fetch('/trade_history.csv?' + Date.now());
      if (resLog.ok) {
        const text = await resLog.text();
        const rows = text.trim().split('\n').slice(1).reverse();
        const trades = rows.map(r => {
          const c = r.split(',');
          return {
            ts: c[0], sym: c[1], side: c[2], lev: c[3],
            entry: parseFloat(c[4]), exit: parseFloat(c[5]),
            pnlAbs: parseFloat(c[6]), pnlRaw: parseFloat(c[7]),
            hold: parseFloat(c[9]), why: c[10], regime: c[12]
          };
        });
        setHistory(trades);
      }
    } catch (e) {
      console.warn("Sync error:", e);
    }
  }, []);

  useEffect(() => {
    const t = setInterval(fetchSync, 1000);
    fetchSync();
    return () => clearInterval(t);
  }, [fetchSync]);

  if(!metrics) return (
    <div style={{background:BG,color:"#333",height:"100vh",display:"flex",
      alignItems:"center",justifyContent:"center",fontFamily:"monospace",letterSpacing:4}}>
      CONNECTING TO Z9 ENGINE...
    </div>
  );

  const card=(ex={})=>({background:CARD,border:`1px solid ${BDR}`,borderRadius:6,padding:"8px 10px",...ex});
  const lbl={color:DIM,fontSize:9,letterSpacing:1.5,marginBottom:3,textTransform:"uppercase"} as any;

  return(
    <div style={{background:BG,color:TXT,fontFamily:"'Courier New',monospace",
      fontSize:11,minHeight:"100vh",padding:10,boxSizing:"border-box"}}>

      {/* HEADER */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",
        borderBottom:`1px solid ${BDR}`,paddingBottom:7,marginBottom:8}}>
        <div>
          <span style={{fontSize:13,fontWeight:900,letterSpacing:3,color:"#4030aa"}}>SKYRMATRON</span>
          <span style={{fontSize:13,fontWeight:900,letterSpacing:3,color:"#fff"}}> Z9</span>
          <span style={{fontSize:8,color:DIM,marginLeft:6}}>v16.5 · HFT REAL-TIME FEED</span>
        </div>
        <div style={{display:"flex",gap:10,alignItems:"center"}}>
          <div style={{display:"flex", gap:15}}>
             <span style={{color:DIM, fontSize:9}}>CPU: <span style={{color:"#9080ee"}}>ISOLATED [2,3]</span></span>
             <span style={{color:DIM, fontSize:9}}>LATENCY: <span style={{color:"#14F195"}}>BUSY_POLL</span></span>
          </div>
          <span style={{color:"#14F195",fontSize:9}}>⬤ ENGINE LIVE</span>
          <span style={{color:DIM,fontSize:9}}>{new Date(metrics.last_update).toLocaleTimeString()}</span>
        </div>
      </div>

      {/* PRICE GRID + ADAPTIVE LEVERAGE */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:6,marginBottom:8}}>
        {SYMS.map(s=>{
          const p = metrics.live_prices[s];
          const sig = metrics.live_signals?.[s] || {};
          const rb = RB[sig.regime] || RB.warmup;
          const active = selSym === s;
          return(
            <div key={s} onClick={()=>setSelSym(s)} style={{...card(),cursor:"pointer",
              borderColor:active?CLR[s]:BDR,background:active?"#0c0c1a":CARD}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
                <span style={{color:CLR[s],fontSize:9,letterSpacing:1.5}}>{s}</span>
                <span style={{background:rb.bg,color:rb.fg,border:`1px solid ${rb.bdr}`,
                      padding:"1px 5px",borderRadius:3,fontSize:7}}>{rb.icon} {rb.lbl}</span>
              </div>
              <div style={{display:"flex", justifyContent:"space-between", alignItems:"flex-end"}}>
                <div style={{color:"#fff",fontSize:20,fontWeight:700}}>{fP(s,p)}</div>
                <div style={{textAlign:"right"}}>
                   <div style={lbl}>Adaptive Lev</div>
                   <div style={{color:"#9080ee", fontSize:14, fontWeight:900}}>{sig.lev || "─"}x</div>
                </div>
              </div>
              <div style={{marginTop:4, display:"flex", justifyContent:"space-between", fontSize:8, color:DIM}}>
                 <span>CONF: <span style={{color:gc(sig.conf - 0.6)}}>{(sig.conf*100).toFixed(1)}%</span></span>
                 <span>DIR: <span style={{color:sig.dir > 0 ? "#14F195" : sig.dir < 0 ? "#ff3366" : DIM}}>{sig.dir > 0 ? "LONG" : sig.dir < 0 ? "SHORT" : "FLAT"}</span></span>
              </div>
            </div>
          );
        })}
      </div>

      {/* METRICS ROW */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:4,marginBottom:8}}>
        {[["Balance",`$${metrics.balance.toFixed(2)}`,gc(metrics.balance-1000)],
          ["Total PnL",sgn(metrics.total_pnl),gc(metrics.total_pnl)],
          ["Win Rate",`${metrics.win_pct}%`,"#14F195"],
          ["Sharpe",metrics.sharpe.toFixed(2),"#627eea"],
          ["Drawdown",`${(metrics.drawdown*100).toFixed(2)}%`,"#ff3366"],
          ["Trades",metrics.total_trades,TXT],
        ].map(([l,v,c])=>(
          <div key={l} style={card({padding:"6px 8px"})}>
            <div style={lbl}>{l}</div>
            <div style={{color:c,fontSize:11,fontWeight:700}}>{v}</div>
          </div>
        ))}
      </div>

      {/* CHART SECTION */}
      <div style={card({marginBottom:8, padding:10})}>
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
          <span style={{color:CLR[selSym],fontSize:10,fontWeight:700}}>{selSym} HFT REAL-TIME FEED</span>
          <span style={{color:DIM,fontSize:9}}>LIVE TICK HISTORY</span>
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData[selSym]}>
            <CartesianGrid strokeDasharray="2 4" stroke="#0c0c14"/>
            <XAxis dataKey="t" hide/>
            <YAxis domain={['auto', 'auto']} tick={{fill:DIM,fontSize:8}} tickFormatter={v=>fY(selSym,v)} width={40}/>
            <Tooltip contentStyle={{background:CARD,border:`1px solid ${BDR}`,fontSize:9}}
              labelFormatter={()=>""} formatter={(v: any)=>[fP(selSym,v),"Price"]}/>
            <Line type="monotone" dataKey="price" stroke={CLR[selSym]} dot={false} strokeWidth={2} isAnimationActive={false}/>
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* LOG SECTION */}
      <div style={card({flex:1})}>
        <div style={{...lbl,marginBottom:8}}>Trade History (v16.5 Live Audit)</div>
        <div style={{maxHeight:300,overflowY:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:9}}>
            <thead>
              <tr style={{color:DIM,borderBottom:`1px solid ${BDR}`,textAlign:"left"}}>
                <th>Time</th><th>Coin</th><th>Side</th><th>Lev</th><th>Entry</th><th>Exit</th><th>PnL$</th><th>Why</th><th>Regime</th>
              </tr>
            </thead>
            <tbody>
              {history.slice(0,50).map((t,i)=>(
                <tr key={i} style={{borderBottom:`1px solid #090910`,background:i%2===0?"transparent":"#0a0a12"}}>
                  <td style={{padding:"4px 2px",color:DIM}}>{fTs(t.ts)}</td>
                  <td style={{color:CLR[t.sym]}}>{t.sym.split("-")[0]}</td>
                  <td style={{color:t.side==="Long"?"#14F195":"#ff3366"}}>{t.side}</td>
                  <td style={{color:"#9080ee", fontWeight:900}}>{t.lev}x</td>
                  <td>{t.entry.toFixed(2)}</td>
                  <td>{t.exit.toFixed(2)}</td>
                  <td style={{color:gc(t.pnlAbs),fontWeight:700}}>{sgn(t.pnlAbs)}</td>
                  <td style={{color:DIM}}>{t.why}</td>
                  <td style={{color:RB[t.regime]?.fg || DIM}}>{t.regime?.toUpperCase()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
