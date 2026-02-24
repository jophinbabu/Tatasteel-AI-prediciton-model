import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  Activity,
  ShieldCheck,
  Cpu,
  RefreshCw,
  ArrowUpRight,
  ArrowDownRight,
  Minus
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

// Mock data for initial chart placeholder
const mockChartData = [
  { name: '09:15', value: 3200 },
  { name: '10:00', value: 3215 },
  { name: '11:00', value: 3195 },
  { name: '12:00', value: 3225 },
  { name: '13:00', value: 3240 },
  { name: '14:00', value: 3230 },
  { name: '15:20', value: 3255 },
];

function App() {
  const [status, setStatus] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      const [statusRes, healthRes] = await Promise.all([
        fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/status`),
        fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/health`)
      ]);

      if (!statusRes.ok || !healthRes.ok) throw new Error('API disconnected');

      const statusData = await statusRes.json();
      const healthData = await healthRes.json();

      setStatus(statusData);
      setHealth(healthData);
      setLoading(false);
      setError(null);
    } catch (err) {
      setError('System Offline');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  const runCycle = async () => {
    try {
      await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/run`, { method: 'POST' });
      fetchData();
    } catch (err) {
      console.error('Failed to run cycle');
    }
  };

  return (
    <div className="app-container" style={{ padding: '20px' }}>
      {/* Header */}
      <header className="glass-panel" style={{ padding: '20px', marginBottom: '30px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: '800', background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            TATA TRADING AI
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Real-time Intelligence Dashboard</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', borderRadius: '20px', background: 'rgba(0,0,0,0.2)' }}>
            <span className={error ? "" : "live-indicator"}></span>
            <span style={{ fontSize: '12px', fontWeight: 'bold' }}>{error ? 'OFFLINE' : 'SYSTEM LIVE'}</span>
          </div>
          <button onClick={runCycle} className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <RefreshCw size={18} />
            RUN CYCLE
          </button>
        </div>
      </header>

      {/* Main Stats */}
      <div className="stats-grid">
        <StatCard
          icon={<TrendingUp color="var(--accent-cyan)" />}
          label="Portfolio Value"
          value={`₹${status?.account_value?.toLocaleString() || '---'}`}
          trend={+2.4}
        />
        <StatCard
          icon={<Activity color="var(--accent-purple)" />}
          label="AI Win Rate"
          value="55.3%"
          trend={+1.2}
        />
        <StatCard
          icon={<ShieldCheck color="var(--accent-green)" />}
          label="Max Drawdown"
          value={`${status?.current_drawdown?.toFixed(2) || '0.00'}%`}
          trend={-0.5}
        />
        <StatCard
          icon={<Cpu color="var(--accent-cyan)" />}
          label="Model Confidence"
          value={`${((health?.avg_confidence || 0) * 100).toFixed(1)}%`}
          active={health?.model_loaded}
        />
      </div>

      {/* Main Content Area */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px', marginTop: '30px' }}>
        {/* Chart View */}
        <div className="glass-panel hero-card animate-fade-in" style={{ height: '450px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
            <h3 style={{ color: 'var(--text-primary)' }}>Performance Analytics</h3>
            <span style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>TATA MOTORS 5M Interval</span>
          </div>
          <div style={{ flex: 1, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockChartData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#58a6ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                <XAxis dataKey="name" stroke="var(--text-secondary)" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="var(--text-secondary)" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1c2128', border: '1px solid var(--border-color)', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Area type="monotone" dataKey="value" stroke="#58a6ff" strokeWidth={3} fillOpacity={1} fill="url(#colorValue)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Signals Feed */}
        <div className="glass-panel hero-card animate-fade-in" style={{ animationDelay: '0.2s', height: '450px', overflowY: 'hidden' }}>
          <h3 style={{ marginBottom: '20px' }}>Live Signal Feed</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            <SignalItem type="BUY" price="3255.40" confidence="72%" time="15:20" />
            <SignalItem type="HOLD" price="3250.10" confidence="45%" time="15:15" />
            <SignalItem type="HOLD" price="3248.80" confidence="41%" time="15:10" />
            <SignalItem type="SELL" price="3252.30" confidence="68%" time="15:05" />
            <SignalItem type="BUY" price="3240.20" confidence="75%" time="15:00" />
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, trend, active }) {
  return (
    <div className="glass-panel hero-card animate-fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ padding: '10px', borderRadius: '12px', background: 'rgba(255,255,255,0.05)' }}>
          {icon}
        </div>
        {trend !== undefined && (
          <span style={{
            fontSize: '12px',
            fontWeight: '600',
            color: trend >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
            display: 'flex',
            alignItems: 'center',
            gap: '4px'
          }}>
            {trend >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
            {Math.abs(trend)}%
          </span>
        )}
        {active !== undefined && (
          <span style={{ fontSize: '10px', color: active ? 'var(--accent-green)' : 'var(--accent-red)' }}>
            {active ? 'LOADED' : 'UNLOADED'}
          </span>
        )}
      </div>
      <div style={{ marginTop: '10px' }}>
        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '4px' }}>{label}</p>
        <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>{value}</h2>
      </div>
    </div>
  );
}

function SignalItem({ type, price, confidence, time }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '10px',
          background: type === 'BUY' ? 'rgba(63, 185, 80, 0.1)' : type === 'SELL' ? 'rgba(248, 81, 73, 0.1)' : 'rgba(88, 166, 255, 0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: type === 'BUY' ? 'var(--accent-green)' : type === 'SELL' ? 'var(--accent-red)' : 'var(--accent-cyan)'
        }}>
          {type === 'BUY' ? <ArrowUpRight size={20} /> : type === 'SELL' ? <ArrowDownRight size={20} /> : <Minus size={20} />}
        </div>
        <div>
          <p style={{ fontWeight: 'bold', fontSize: '14px' }}>{type}</p>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>₹{price}</p>
        </div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <p style={{ fontWeight: '600', fontSize: '14px' }}>{confidence}</p>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{time}</p>
      </div>
    </div>
  );
}

export default App;
