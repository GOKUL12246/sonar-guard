import React, { useState } from 'react';

export default function LoginPage({ onLogin }) {
  const [email, setEmail] = useState('operator@sonarguard.org');
  const [password, setPassword] = useState('sonar-guard-2026');
  const [role, setRole] = useState('Hydrographic Operations Lead');
  const [remember, setRemember] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleLogin = (e) => {
    if (e) e.preventDefault();
    setError(null);
    if (!email || !password) {
      setError('Please provide both Operator Identifier and Security Passcode.');
      return;
    }

    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      const user = {
        name: email.split('@')[0].toUpperCase(),
        email,
        role,
        clearance: 'LEVEL-IV MARITIME DEFENCE',
        loginTime: new Date().toISOString(),
      };
      if (remember) {
        try {
          localStorage.setItem('sonar_guard_auth_user', JSON.stringify(user));
        } catch (_) {}
      }
      onLogin(user);
    }, 600);
  };

  const handleQuickPreset = (presetRole, presetEmail) => {
    setRole(presetRole);
    setEmail(presetEmail);
    setPassword('sonar-guard-2026');
    const user = {
      name: presetEmail.split('@')[0].toUpperCase(),
      email: presetEmail,
      role: presetRole,
      clearance: 'LEVEL-IV MARITIME DEFENCE',
      loginTime: new Date().toISOString(),
    };
    try {
      localStorage.setItem('sonar_guard_auth_user', JSON.stringify(user));
    } catch (_) {}
    onLogin(user);
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'radial-gradient(circle at 50% 20%, #07172c 0%, #030a16 100%)',
      fontFamily: "'Inter', -apple-system, sans-serif",
      padding: '1.5rem',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Background Decorative Grid */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundImage: 'linear-gradient(rgba(2, 132, 199, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(2, 132, 199, 0.05) 1px, transparent 1px)',
        backgroundSize: '40px 40px',
        pointerEvents: 'none',
      }} />

      <div style={{
        width: '100%',
        maxWidth: '480px',
        background: 'linear-gradient(180deg, rgba(11, 25, 46, 0.95) 0%, rgba(7, 18, 36, 0.98) 100%)',
        border: '1px solid rgba(2, 132, 199, 0.3)',
        borderRadius: '20px',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 40px rgba(2, 132, 199, 0.15)',
        padding: '2.5rem 2.2rem',
        position: 'relative',
        zIndex: 2,
        backdropFilter: 'blur(10px)',
      }}>
        {/* Header Branding */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '64px',
            height: '64px',
            borderRadius: '16px',
            background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
            boxShadow: '0 0 24px rgba(2, 132, 199, 0.5)',
            marginBottom: '1rem',
          }}>
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 2a10 10 0 0 1 10 10" />
              <path d="M12 6a6 6 0 0 1 6 6" />
              <path d="M12 10a2 2 0 0 1 2 2" />
              <line x1="12" y1="12" x2="19" y2="5" />
            </svg>
          </div>

          <h2 style={{
            margin: '0 0 6px 0',
            color: '#f8fafc',
            fontSize: '1.45rem',
            fontWeight: 800,
            letterSpacing: '0.5px',
          }}>
            SONAR-GUARD DEFENCE
          </h2>
          <div style={{
            fontSize: '0.78rem',
            color: '#38bdf8',
            fontWeight: 700,
            letterSpacing: '1px',
            textTransform: 'uppercase',
          }}>
            Autonomous Marine Acoustic Intelligence
          </div>
          <p style={{
            margin: '8px 0 0 0',
            color: '#94a3b8',
            fontSize: '0.8rem',
            lineHeight: 1.4,
          }}>
            Classified Underwater Debris &amp; Ghost Net Telemetry Portal
          </p>
        </div>

        {error && (
          <div style={{
            background: 'rgba(220, 38, 38, 0.15)',
            border: '1px solid rgba(220, 38, 38, 0.4)',
            color: '#fca5a5',
            padding: '10px 14px',
            borderRadius: '8px',
            fontSize: '0.82rem',
            marginBottom: '1.2rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <span><b>Security Alert:</b> {error}</span>
          </div>
        )}

        {/* Form Controls */}
        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
          <div>
            <label style={{ display: 'block', color: '#cbd5e1', fontSize: '0.78rem', fontWeight: 600, marginBottom: '6px' }}>
              OPERATOR IDENTIFIER / OFFICIAL EMAIL
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="e.g. operator@sonarguard.org"
              required
              style={{
                width: '100%',
                padding: '11px 14px',
                background: '#071529',
                border: '1px solid #1e3a8a',
                borderRadius: '8px',
                color: '#ffffff',
                fontSize: '0.9rem',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', color: '#cbd5e1', fontSize: '0.78rem', fontWeight: 600, marginBottom: '6px' }}>
              ACCESS PASSCODE / KEY
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              required
              style={{
                width: '100%',
                padding: '11px 14px',
                background: '#071529',
                border: '1px solid #1e3a8a',
                borderRadius: '8px',
                color: '#ffffff',
                fontSize: '0.9rem',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', color: '#cbd5e1', fontSize: '0.78rem', fontWeight: 600, marginBottom: '6px' }}>
              ASSIGNED MISSION ROLE
            </label>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              style={{
                width: '100%',
                padding: '11px 14px',
                background: '#071529',
                border: '1px solid #1e3a8a',
                borderRadius: '8px',
                color: '#38bdf8',
                fontSize: '0.88rem',
                fontWeight: 600,
                outline: 'none',
                boxSizing: 'border-box',
              }}
            >
              <option value="Hydrographic Operations Lead">Hydrographic Operations Lead (Full Admin)</option>
              <option value="Senior Acoustic Analyst">Senior Acoustic Analyst (Inference &amp; Scoring)</option>
              <option value="Marine Robotics / ROV Specialist">Marine Robotics / ROV Specialist (Field Operations)</option>
              <option value="Naval Environmental Officer">Naval Environmental Officer (Audit &amp; Reports)</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.78rem', color: '#94a3b8' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
                style={{ accentColor: '#0284c7' }}
              />
              Remember Clearance on this Console
            </label>
            <span style={{ color: '#0284c7', cursor: 'pointer' }}>AES-256 Encrypted</span>
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              marginTop: '0.5rem',
              padding: '13px',
              background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
              border: 'none',
              borderRadius: '8px',
              color: '#ffffff',
              fontSize: '0.95rem',
              fontWeight: 700,
              cursor: loading ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 14px rgba(2, 132, 199, 0.4)',
              transition: 'all 0.2s ease',
            }}
          >
            {loading ? 'Verifying Acoustic Credentials…' : 'Access Maritime Surveillance Portal'}
          </button>
        </form>

        {/* Quick Demo Access Badges (For Hackathon Judges) */}
        <div style={{ marginTop: '1.8rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '1.2rem' }}>
          <div style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 700, letterSpacing: '0.8px', marginBottom: '8px', textAlign: 'center' }}>
            QUICK EVALUATOR ACCESS (1-CLICK DEMO LOGIN)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <button
              type="button"
              onClick={() => handleQuickPreset('Hydrographic Operations Lead', 'lead.hydrographer@sonarguard.org')}
              style={{
                padding: '8px 10px',
                background: 'rgba(2, 132, 199, 0.1)',
                border: '1px solid rgba(2, 132, 199, 0.3)',
                borderRadius: '6px',
                color: '#e0f2fe',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              <div style={{ color: '#38bdf8', fontWeight: 700 }}>Hydrographer Lead</div>
              <div style={{ color: '#94a3b8', fontSize: '0.68rem' }}>Full Command Access</div>
            </button>

            <button
              type="button"
              onClick={() => handleQuickPreset('Senior Acoustic Analyst', 'analyst@sonarguard.org')}
              style={{
                padding: '8px 10px',
                background: 'rgba(16, 185, 129, 0.1)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: '6px',
                color: '#e0f2fe',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              <div style={{ color: '#34d399', fontWeight: 700 }}>Acoustic Analyst</div>
              <div style={{ color: '#94a3b8', fontSize: '0.68rem' }}>Inference &amp; Scoring</div>
            </button>
          </div>
        </div>

        {/* Footer Security Notice */}
        <div style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.68rem', color: '#475569' }}>
          Official Maritime Telemetry &bull; WGS-84 GIS Protocol Active &bull; MongoDB Atlas Secured
        </div>
      </div>
    </div>
  );
}
