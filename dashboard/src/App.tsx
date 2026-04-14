import React, { useState, useEffect } from 'react';
import './App.css';

interface Metrics {
  accuracy: number;
  sharpe: number;
  drawdown: number;
  win_loss_ratio: number;
  win_pct: number;
  total_trades: number;
  total_pnl: number;
  current_balance: number;
  live_prices: {
    'BTC-USD': number;
    'ETH-USD': number;
    'SOL-USD': number;
  };
}

interface Trade {
  timestamp: string;
  coin: string;
  type: string;
  leverage: string;
  entry_price: string;
  exit_price: string;
  pnl_abs: string;
  pnl_pct: string;
  status: string;
  z9_confidence: string;
}

const App: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);

  const fetchData = async () => {
    try {
      const metricsRes = await fetch('/real_time_metrics.json?' + new Date().getTime());
      if (metricsRes.ok) {
        const data = await metricsRes.json();
        setMetrics(data);
      }

      const tradesRes = await fetch('/trade_history.csv?' + new Date().getTime());
      if (tradesRes.ok) {
        const csvText = await tradesRes.text();
        const rows = csvText.split('\n').filter(r => r.trim() !== '').slice(1).reverse();
        const parsedTrades: Trade[] = rows.map(row => {
          const fields = row.split(',');
          return { 
            timestamp: fields[0] || '', coin: fields[1] || '', type: fields[2] || '', 
            leverage: fields[3] || '0', entry_price: fields[4] || '0', exit_price: fields[5] || '0', 
            pnl_abs: fields[6] || '0', pnl_pct: fields[7] || '0', status: fields[8] || '', z9_confidence: fields[9] || '0' 
          };
        });
        setTrades(parsedTrades);
      }
    } catch (e) {
      console.error("Error fetching data", e);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  if (!metrics) return <div className="dashboard-container">Loading Oracle Data...</div>;

  return (
    <div className="dashboard-container">
      <header className="header">
        <div className="logo">Skyrmatron Z9 <span style={{color: '#fff'}}>v12.3</span></div>
        <div className="status-badge">LIVE SIMULATION ACTIVE</div>
      </header>

      {/* Live Prices Grid */}
      <div className="metrics-grid" style={{ marginBottom: '20px' }}>
        <div className="metric-card" style={{ borderColor: '#f7931a' }}>
          <div className="metric-label" style={{ color: '#f7931a' }}>BTC/USD (Live)</div>
          <div className="metric-value">${(metrics.live_prices?.['BTC-USD'] || 0).toFixed(2)}</div>
        </div>
        <div className="metric-card" style={{ borderColor: '#627eea' }}>
          <div className="metric-label" style={{ color: '#627eea' }}>ETH/USD (Live)</div>
          <div className="metric-value">${(metrics.live_prices?.['ETH-USD'] || 0).toFixed(2)}</div>
        </div>
        <div className="metric-card" style={{ borderColor: '#14F195' }}>
          <div className="metric-label" style={{ color: '#14F195' }}>SOL/USD (Live)</div>
          <div className="metric-value">${(metrics.live_prices?.['SOL-USD'] || 0).toFixed(2)}</div>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Current Balance</div>
          <div className="metric-value">${(metrics.current_balance || 0).toFixed(2)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Total PnL</div>
          <div className={`metric-value ${(metrics.total_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
            {(metrics.total_pnl || 0) >= 0 ? '+' : ''}${(metrics.total_pnl || 0).toFixed(2)}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Accuracy</div>
          <div className="metric-value">{((metrics.accuracy || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Sharpe Ratio</div>
          <div className="metric-value">{(metrics.sharpe || 0).toFixed(2)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Drawdown</div>
          <div className="metric-value negative">{((metrics.drawdown || 0) * 100).toFixed(2)}%</div>
        </div>
      </div>

      <div className="content-grid">
        <div className="panel">
          <div className="panel-title">Real-Time Trade Feed</div>
          <div className="trade-log">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Coin</th>
                  <th>Type</th>
                  <th>Lev</th>
                  <th>PnL %</th>
                  <th>PnL $</th>
                  <th>Conf</th>
                </tr>
              </thead>
              <tbody>
                {trades.length === 0 ? (
                  <tr><td colSpan={7} style={{textAlign: 'center', padding: '20px'}}>Waiting for first signal...</td></tr>
                ) : trades.slice(0, 20).map((trade, i) => (
                  <tr key={i}>
                    <td>{trade.timestamp ? new Date(trade.timestamp).toLocaleTimeString() : ''}</td>
                    <td>{trade.coin}</td>
                    <td style={{color: trade.type === 'Long' ? '#3dffaf' : '#ff4d4d'}}>{trade.type}</td>
                    <td>{trade.leverage}x</td>
                    <td className={parseFloat(trade.pnl_pct || '0') >= 0 ? 'positive' : 'negative'}>
                      {(parseFloat(trade.pnl_pct || '0') * 100).toFixed(2)}%
                    </td>
                    <td className={parseFloat(trade.pnl_abs || '0') >= 0 ? 'positive' : 'negative'}>
                      ${parseFloat(trade.pnl_abs || '0').toFixed(2)}
                    </td>
                    <td>{parseFloat(trade.z9_confidence || '0').toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">Oracle Diagnostics</div>
          <div className="metric-label">Z9 Confidence Swarm</div>
          <div className="confidence-meter">
            <div className="confidence-fill" style={{width: `${(parseFloat(trades[0]?.z9_confidence || '0.5') / 1.0) * 100}%`}}></div>
            <div className="confidence-text">{(parseFloat(trades[0]?.z9_confidence || '0.5') * 100).toFixed(1)}%</div>
          </div>
          
          <div style={{marginTop: '40px'}}>
            <div className="metric-label">Win/Loss Ratio</div>
            <div className="metric-value" style={{fontSize: '20px'}}>{(metrics.win_loss_ratio || 0).toFixed(2)}</div>
          </div>

          <div style={{marginTop: '20px'}}>
            <div className="metric-label">Total Trades</div>
            <div className="metric-value" style={{fontSize: '20px'}}>{metrics.total_trades || 0}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
