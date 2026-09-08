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
      background: 'linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%)',
      backgroundImage: 'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(2, 132, 199, 0.08), transparent), linear-gradient(rgba(15, 62, 120, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(15, 62, 120, 0.03) 1px, transparent 1px)',
      backgroundSize: '100% 100%, 36px 36px, 36px 36px',
      fontFamily: "'Inter', -apple-system, sans-serif",
      padding: '1.5rem',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Top subtle ambient glow */}
      <div style={{
        position: 'absolute',
        top: '-120px',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '600px',
        height: '300px',
        background: 'radial-gradient(circle, rgba(2, 132, 199, 0.12) 0%, transparent 70%)',
        filter: 'blur(50px)',
        pointerEvents: 'none',
      }} />

      <div style={{
        width: '100%',
        maxWidth: '460px',
        background: '#ffffff',
        border: '1px solid #e2e8f0',
        borderTop: '5px solid #0f3e78',
        borderRadius: '16px',
        boxShadow: '0 20px 45px rgba(15, 62, 120, 0.1), 0 4px 12px rgba(0, 0, 0, 0.04)',
        padding: '2.4rem 2.2rem',
        position: 'relative',
        zIndex: 2,
      }}>
        {/* Header Branding */}
        <div style={{ textAlign: 'center', marginBottom: '1.8rem' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '60px',
            height: '60px',
            borderRadius: '14px',
            background: 'linear-gradient(135deg, #07172c 0%, #0f3e78 100%)',
            border: '2px solid rgba(212, 175, 55, 0.4)',
            boxShadow: '0 8px 20px rgba(15, 62, 120, 0.25)',
            marginBottom: '1rem',
          }}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 2a10 10 0 0 1 10 10" />
              <path d="M12 6a6 6 0 0 1 6 6" />
              <path d="M12 10a2 2 0 0 1 2 2" />
              <line x1="12" y1="12" x2="19" y2="5" />
            </svg>
          </div>

          <h2 style={{
            margin: '0 0 4px 0',
            color: '#0f172a',
            fontSize: '1.4rem',
            fontWeight: 800,
            letterSpacing: '-0.3px',
          }}>
            SONAR-GUARD
          </h2>
          <div style={{
            display: 'inline-block',
            padding: '3px 10px',
            borderRadius: '20px',
            background: '#e0f2fe',
            border: '1px solid #bae6fd',
            fontSize: '0.72rem',
            color: '#0369a1',
            fontWeight: 700,
            letterSpacing: '0.6px',
            textTransform: 'uppercase',
            marginBottom: '6px',
          }}>
            Marine Acoustic Intelligence
          </div>
          <p style={{
            margin: '4px 0 0 0',
            color: '#64748b',
            fontSize: '0.82rem',
            lineHeight: 1.4,
          }}>
            Autonomous Underwater Debris &amp; Ghost Net Telemetry
          </p>
        </div>

        {error && (
          <div style={{
            background: '#fef2f2',
            border: '1px solid #fecaca',
            color: '#dc2626',
            padding: '10px 14px',
            borderRadius: '8px',
            fontSize: '0.82rem',
            marginBottom: '1.2rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <span><b>Alert:</b> {error}</span>
          </div>
        )}

        {/* Form Controls */}
        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
          <div>
            <label style={{ display: 'block', color: '#334155', fontSize: '0.78rem', fontWeight: 600, marginBottom: '6px', letterSpacing: '0.2px' }}>
              OPERATOR IDENTIFIER / EMAIL
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
                background: '#ffffff',
                border: '1px solid #cbd5e1',
                borderRadius: '8px',
                color: '#0f172a',
                fontSize: '0.9rem',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'border-color 0.2s',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', color: '#334155', fontSize: '0.78rem', fontWeight: 600, marginBottom: '6px', letterSpacing: '0.2px' }}>
              SECURITY PASSCODE / KEY
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
                background: '#ffffff',
                border: '1px solid #cbd5e1',
                borderRadius: '8px',
                color: '#0f172a',
                fontSize: '0.9rem',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'border-color 0.2s',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', color: '#334155', fontSize: '0.78rem', fontWeight: 600, marginBottom: '6px', letterSpacing: '0.2px' }}>
              ASSIGNED MISSION ROLE
            </label>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              style={{
                width: '100%',
                padding: '11px 14px',
                background: '#ffffff',
                border: '1px solid #cbd5e1',
                borderRadius: '8px',
                color: '#0f3e78',
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

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.78rem', color: '#64748b' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
                style={{ accentColor: '#0f3e78' }}
              />
              Remember Session on this Console
            </label>
            <span style={{ color: '#0369a1', fontWeight: 600 }}>AES-256 Protocol</span>
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              marginTop: '0.6rem',
              padding: '13px',
              background: 'linear-gradient(135deg, #07172c 0%, #0f3e78 100%)',
              border: 'none',
              borderRadius: '8px',
              color: '#ffffff',
              fontSize: '0.95rem',
              fontWeight: 700,
              cursor: loading ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 14px rgba(15, 62, 120, 0.3)',
              transition: 'all 0.2s ease',
            }}
          >
            {loading ? 'Authenticating Credentials…' : 'Sign In to Portal'}
          </button>
        </form>

        {/* Quick Demo Access Badges (For Hackathon Judges) */}
        <div style={{ marginTop: '1.8rem', borderTop: '1px solid #e2e8f0', paddingTop: '1.2rem' }}>
          <div style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 700, letterSpacing: '0.8px', marginBottom: '8px', textAlign: 'center' }}>
            QUICK EVALUATOR ACCESS (1-CLICK DEMO LOGIN)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <button
              type="button"
              onClick={() => handleQuickPreset('Hydrographic Operations Lead', 'lead.hydrographer@sonarguard.org')}
              style={{
                padding: '9px 12px',
                background: '#f8fafc',
                border: '1px solid #cbd5e1',
                borderRadius: '8px',
                color: '#0f172a',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.2s',
              }}
            >
              <div style={{ color: '#0f3e78', fontWeight: 700 }}>Hydrographer Lead</div>
              <div style={{ color: '#64748b', fontSize: '0.68rem' }}>Full Command Access</div>
            </button>

            <button
              type="button"
              onClick={() => handleQuickPreset('Senior Acoustic Analyst', 'analyst@sonarguard.org')}
              style={{
                padding: '9px 12px',
                background: '#f8fafc',
                border: '1px solid #cbd5e1',
                borderRadius: '8px',
                color: '#0f172a',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.2s',
              }}
            >
              <div style={{ color: '#059669', fontWeight: 700 }}>Acoustic Analyst</div>
              <div style={{ color: '#64748b', fontSize: '0.68rem' }}>Inference &amp; Scoring</div>
            </button>
          </div>
        </div>

        {/* Footer Security Notice */}
        <div style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.68rem', color: '#94a3b8' }}>
          Official Marine Acoustic Telemetry &bull; WGS-84 GIS Standard &bull; MongoDB Atlas Cloud
        </div>
      </div>
    </div>
  );
}
