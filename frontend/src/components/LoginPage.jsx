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
      background: '#09090b',
      backgroundImage: 'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(255, 255, 255, 0.12), transparent), linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px)',
      backgroundSize: '100% 100%, 32px 32px, 32px 32px',
      fontFamily: "'Inter', -apple-system, sans-serif",
      padding: '1.5rem',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Top subtle white ambient spotlight glow */}
      <div style={{
        position: 'absolute',
        top: '-150px',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '600px',
        height: '350px',
        background: 'radial-gradient(circle, rgba(255, 255, 255, 0.08) 0%, transparent 70%)',
        filter: 'blur(50px)',
        pointerEvents: 'none',
      }} />

      <div style={{
        width: '100%',
        maxWidth: '460px',
        background: 'rgba(18, 18, 20, 0.88)',
        border: '1px solid rgba(255, 255, 255, 0.14)',
        borderRadius: '20px',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.9), 0 0 0 1px rgba(255, 255, 255, 0.05)',
        padding: '2.4rem 2.2rem',
        position: 'relative',
        zIndex: 2,
        backdropFilter: 'blur(20px)',
      }}>
        {/* Header Branding */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '58px',
            height: '58px',
            borderRadius: '14px',
            background: '#ffffff',
            boxShadow: '0 0 30px rgba(255, 255, 255, 0.25)',
            marginBottom: '1.2rem',
          }}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#09090b" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 2a10 10 0 0 1 10 10" />
              <path d="M12 6a6 6 0 0 1 6 6" />
              <path d="M12 10a2 2 0 0 1 2 2" />
              <line x1="12" y1="12" x2="19" y2="5" />
            </svg>
          </div>

          <h2 style={{
            margin: '0 0 6px 0',
            color: '#ffffff',
            fontSize: '1.45rem',
            fontWeight: 800,
            letterSpacing: '-0.3px',
          }}>
            SONAR-GUARD
          </h2>
          <div style={{
            display: 'inline-block',
            padding: '3px 10px',
            borderRadius: '20px',
            background: 'rgba(255, 255, 255, 0.08)',
            border: '1px solid rgba(255, 255, 255, 0.16)',
            fontSize: '0.7rem',
            color: '#e4e4e7',
            fontWeight: 700,
            letterSpacing: '0.8px',
            textTransform: 'uppercase',
            marginBottom: '8px',
          }}>
            Marine Acoustic Intelligence
          </div>
          <p style={{
            margin: '4px 0 0 0',
            color: '#a1a1aa',
            fontSize: '0.82rem',
            lineHeight: 1.4,
          }}>
            Autonomous Underwater Debris &amp; Ghost Net Telemetry
          </p>
        </div>

        {error && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(239, 68, 68, 0.5)',
            color: '#f87171',
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
            <label style={{ display: 'block', color: '#d4d4d8', fontSize: '0.78rem', fontWeight: 600, marginBottom: '6px', letterSpacing: '0.3px' }}>
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
                background: '#09090b',
                border: '1px solid #27272a',
                borderRadius: '8px',
                color: '#ffffff',
                fontSize: '0.9rem',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'border-color 0.2s',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', color: '#d4d4d8', fontSize: '0.78rem', fontWeight: 600, marginBottom: '6px', letterSpacing: '0.3px' }}>
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
                background: '#09090b',
                border: '1px solid #27272a',
                borderRadius: '8px',
                color: '#ffffff',
                fontSize: '0.9rem',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'border-color 0.2s',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', color: '#d4d4d8', fontSize: '0.78rem', fontWeight: 600, marginBottom: '6px', letterSpacing: '0.3px' }}>
              ASSIGNED MISSION ROLE
            </label>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              style={{
                width: '100%',
                padding: '11px 14px',
                background: '#09090b',
                border: '1px solid #27272a',
                borderRadius: '8px',
                color: '#ffffff',
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

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.78rem', color: '#a1a1aa' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
                style={{ accentColor: '#ffffff' }}
              />
              Remember Session on this Console
            </label>
            <span style={{ color: '#d4d4d8', fontWeight: 600 }}>AES-256 Protocol</span>
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              marginTop: '0.6rem',
              padding: '13px',
              background: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              color: '#09090b',
              fontSize: '0.95rem',
              fontWeight: 800,
              cursor: loading ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 20px rgba(255, 255, 255, 0.2)',
              transition: 'all 0.2s ease',
            }}
          >
            {loading ? 'Authenticating Credentials…' : 'Sign In to Portal'}
          </button>
        </form>

        {/* Quick Demo Access Badges (For Hackathon Judges) */}
        <div style={{ marginTop: '1.8rem', borderTop: '1px solid rgba(255, 255, 255, 0.1)', paddingTop: '1.2rem' }}>
          <div style={{ fontSize: '0.72rem', color: '#71717a', fontWeight: 700, letterSpacing: '0.8px', marginBottom: '8px', textAlign: 'center' }}>
            QUICK EVALUATOR ACCESS (1-CLICK DEMO LOGIN)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <button
              type="button"
              onClick={() => handleQuickPreset('Hydrographic Operations Lead', 'lead.hydrographer@sonarguard.org')}
              style={{
                padding: '9px 12px',
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                color: '#ffffff',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'background 0.2s',
              }}
            >
              <div style={{ color: '#ffffff', fontWeight: 700 }}>Hydrographer Lead</div>
              <div style={{ color: '#a1a1aa', fontSize: '0.68rem' }}>Full Command Access</div>
            </button>

            <button
              type="button"
              onClick={() => handleQuickPreset('Senior Acoustic Analyst', 'analyst@sonarguard.org')}
              style={{
                padding: '9px 12px',
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                color: '#ffffff',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'background 0.2s',
              }}
            >
              <div style={{ color: '#ffffff', fontWeight: 700 }}>Acoustic Analyst</div>
              <div style={{ color: '#a1a1aa', fontSize: '0.68rem' }}>Inference &amp; Scoring</div>
            </button>
          </div>
        </div>

        {/* Footer Security Notice */}
        <div style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.68rem', color: '#52525b' }}>
          Official Marine Acoustic Telemetry &bull; WGS-84 GIS Standard &bull; MongoDB Atlas Cloud
        </div>
      </div>
    </div>
  );
}
