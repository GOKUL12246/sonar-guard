import { useCallback, useEffect, useMemo, useState } from 'react';
import MapView from './components/MapView.jsx';
import Dashboard from './components/Dashboard.jsx';
import AnalysisStudio from './components/AnalysisStudio.jsx';
import ReviewQueue from './components/ReviewQueue.jsx';
import Reports from './components/Reports.jsx';
import { fetchContacts, reversePlace, searchPlaces, submitVerification } from './api.js';
import { GLOBAL_MARINE_REGIONS } from './constants/regions.js';

const ALL_BANDS = ['Critical', 'High', 'Medium', 'Low'];
const DEFAULT_CENTER = [13.1150, 80.3400];
const TABS = [
  ['dashboard', 'Dashboard'],
  ['analysis', 'Analysis Studio'],
  ['map', 'Marine GIS Map'],
  ['review', 'Review Queue'],
  ['reports', 'Mission Reports'],
];

export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [contacts, setContacts] = useState([]);
  const [places, setPlaces] = useState({});
  const [center, setCenter] = useState(DEFAULT_CENTER);
  const [centerPlace, setCenterPlace] = useState('Offshore Bay of Bengal (Marine Survey Track)');
  const [currentRegionId, setCurrentRegionId] = useState('in-bob');
  const [query, setQuery] = useState('');
  const [latInput, setLatInput] = useState('13.1150');
  const [lonInput, setLonInput] = useState('80.3400');
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [activeBands, setActiveBands] = useState(ALL_BANDS);
  const [verifyingId, setVerifyingId] = useState(null);
  const [toast, setToast] = useState(null);
  const [error, setError] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  const refreshContacts = useCallback(async () => {
    try {
      setContacts(await fetchContacts());
      setError(null);
    } catch {
      setError('Backend unreachable — start it with: uvicorn backend.main:app --port 8000');
    }
  }, []);

  useEffect(() => {
    (async () => {
      await refreshContacts();
      setLoading(false);
    })();
  }, [refreshContacts]);

  // Resolve real place names once per contact list
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const entries = await Promise.all(
        contacts.map(async (c) => {
          if (c.latitude == null || c.longitude == null) return [c.anomaly_id, '—'];
          try {
            return [c.anomaly_id, await reversePlace(c.latitude, c.longitude)];
          } catch {
            return [c.anomaly_id, `${c.latitude.toFixed(4)}, ${c.longitude.toFixed(4)}`];
          }
        }),
      );
      if (!cancelled) setPlaces(Object.fromEntries(entries));
    })();
    return () => {
      cancelled = true;
    };
  }, [contacts]);

  useEffect(() => {
    (async () => {
      try {
        setCenterPlace(await reversePlace(center[0], center[1]));
      } catch {
        setCenterPlace(`${center[0].toFixed(4)}, ${center[1].toFixed(4)}`);
      }
    })();
  }, [center]);

  const visible = useMemo(
    () => contacts.filter((c) => activeBands.includes(c.marine_risk_band)),
    [contacts, activeBands],
  );

  const handleSelectRegion = (region) => {
    if (!region) return;
    setCurrentRegionId(region.id);
    setCenter([region.lat, region.lon]);
    setLatInput(String(region.lat));
    setLonInput(String(region.lon));
    setCenterPlace(region.name);
    setToast({ kind: 'ok', text: `Survey region updated to ${region.name}` });
  };

  const runSearch = useCallback(async () => {
    const q = query.trim();
    if (q.length < 2) return;
    setSearching(true);
    try {
      const results = await searchPlaces(q);
      if (results.length === 0) {
        setToast({ kind: 'error', text: `No place found for “${q}”.` });
      } else {
        setCenter([results[0].lat, results[0].lon]);
        setLatInput(String(results[0].lat));
        setLonInput(String(results[0].lon));
        setToast({ kind: 'ok', text: `Relocated to: ${results[0].display_name}` });
      }
    } catch {
      setToast({ kind: 'error', text: 'Search lookup failed.' });
    } finally {
      setSearching(false);
    }
  }, [query]);

  const handleVerify = useCallback(
    async (anomalyId, decision) => {
      setVerifyingId(anomalyId);
      try {
        await submitVerification(anomalyId, decision);
        setContacts((prev) =>
          prev.map((c) => (c.anomaly_id === anomalyId ? { ...c, verification_status: decision } : c)),
        );
        setToast({ kind: 'ok', text: `Anomaly ${anomalyId.slice(0, 8)} recorded as ${decision}` });
      } catch (err) {
        setToast({ kind: 'error', text: err.message });
      } finally {
        setVerifyingId(null);
      }
    },
    [],
  );

  const handleAnalyzed = useCallback(
    async (out) => {
      setAnalysis(out);
      await refreshContacts();
      setToast({
        kind: 'ok',
        text: `Analysis complete: ${out.num_accepted} of ${out.num_candidates} obstacles accepted (${out.t_det_ms} ms)`,
      });
    },
    [refreshContacts],
  );

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4200);
    return () => clearTimeout(t);
  }, [toast]);

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: '#0f172a' }}>
              <path d="M2 12h20M2 12l5-5m-5 5l5 5M22 12l-5-5m5 5l-5 5" />
            </svg>
          </div>
          <div>
            <h1>
              UNDERWATER GHOST NET &amp; DEBRIS DETECTION<span> PLATFORM</span>
            </h1>
            <p>SIDE-SCAN SONAR ACOUSTIC INTELLIGENCE &amp; MARINE HAZARD MAPPING</p>
          </div>
        </div>
        <div className="live-pill">
          <span className="dot" /> SYSTEM OPERATIONAL · REAL-TIME INFERENCE READY
        </div>
      </header>

      <nav className="tabs">
        {TABS.map(([id, label]) => (
          <button key={id} className={tab === id ? 'tab active' : 'tab'} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </nav>

      {toast && <div className={`toast toast-${toast.kind}`}>{toast.text}</div>}
      {error && <div className="toast toast-error">{error}</div>}
      {loading && <div className="loading">Loading survey contacts…</div>}

      {tab === 'dashboard' && (
        <Dashboard
          contacts={contacts}
          centerPlace={centerPlace}
          currentRegionId={currentRegionId}
          onSelectRegion={handleSelectRegion}
          analysis={analysis}
          onNavigate={setTab}
        />
      )}

      {tab === 'analysis' && (
        <AnalysisStudio onAnalyzed={handleAnalyzed} defaultLat={center[0]} defaultLon={center[1]} />
      )}

      {tab === 'map' && (
        <div>
          <section className="controls">
            <div className="search" style={{ flexWrap: 'wrap', gap: '0.6rem' }}>
              <select
                style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.82rem', fontWeight: 600, background: '#ffffff', minWidth: '240px' }}
                value={currentRegionId}
                onChange={(e) => {
                  const reg = GLOBAL_MARINE_REGIONS.find((r) => r.id === e.target.value);
                  if (reg) handleSelectRegion(reg);
                }}
              >
                {GLOBAL_MARINE_REGIONS.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.country}: {r.name.split('·')[1]?.trim() || r.name}
                  </option>
                ))}
              </select>

              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && runSearch()}
                placeholder="Search marine area (e.g. Bay of Bengal, Gulf of Mannar, Arabian Sea)…"
                style={{ minWidth: '240px', flex: 1 }}
              />
              <button onClick={runSearch} disabled={searching}>
                {searching ? 'Searching…' : 'Locate Marine Area'}
              </button>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginLeft: 'auto' }}>
                <span style={{ fontSize: '0.8rem', color: '#64748b' }}>GPS:</span>
                <input
                  type="number"
                  step="0.0001"
                  value={latInput}
                  onChange={(e) => setLatInput(e.target.value)}
                  placeholder="Lat"
                  style={{ width: '85px', padding: '6px 8px', fontSize: '0.8rem', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                />
                <input
                  type="number"
                  step="0.0001"
                  value={lonInput}
                  onChange={(e) => setLonInput(e.target.value)}
                  placeholder="Lon"
                  style={{ width: '85px', padding: '6px 8px', fontSize: '0.8rem', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                />
                <button
                  onClick={() => {
                    const l1 = parseFloat(latInput);
                    const l2 = parseFloat(lonInput);
                    if (!isNaN(l1) && !isNaN(l2) && l1 >= -90 && l1 <= 90 && l2 >= -180 && l2 <= 180) {
                      setCenter([l1, l2]);
                      setToast({ kind: 'ok', text: `Map relocated to GPS: ${l1.toFixed(4)}°, ${l2.toFixed(4)}°` });
                    }
                  }}
                  style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                >
                  Fly to GPS
                </button>
              </div>
            </div>
            <div className="bands">
              {ALL_BANDS.map((b) => (
                <label key={b} className="band">
                  <input
                    type="checkbox"
                    checked={activeBands.includes(b)}
                    onChange={() =>
                      setActiveBands((prev) =>
                        prev.includes(b) ? prev.filter((x) => x !== b) : [...prev, b],
                      )
                    }
                  />
                  {b}
                </label>
              ))}
            </div>
          </section>

          <p className="center-line">
            Active Survey Zone: <b>{centerPlace}</b> at {center[0].toFixed(5)}° N, {center[1].toFixed(5)}° E
          </p>

          <MapView
            contacts={contacts}
            center={center}
            activeBands={activeBands}
            onVerify={handleVerify}
            verifyingId={verifyingId}
          />

          <section className="card">
            <h3>Georeferenced Obstacle Registry — {centerPlace}</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Contact ID</th>
                    <th>Class</th>
                    <th>Physical Dimensions</th>
                    <th>Location</th>
                    <th>Artificial / Natural</th>
                    <th>Risk Priority</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((c) => (
                    <tr key={c.anomaly_id}>
                      <td>
                        <code>{c.anomaly_id.slice(0, 13)}</code>
                      </td>
                      <td><b>{c.class}</b></td>
                      <td>
                        <span style={{ fontSize: '0.82rem', color: '#0369a1', fontWeight: 600 }}>
                          {c.dimensions_text || (c.length_m ? `${c.length_m}m × ${c.width_m}m` : '2.4m × 1.1m')}
                        </span>
                      </td>
                      <td className="place-cell">{places[c.anomaly_id] ?? 'Open Sea'}</td>
                      <td>
                        {c.artificial_score}% / {c.natural_score}%
                      </td>
                      <td>
                        <span className={`risk risk-${c.marine_risk_band.toLowerCase()}`}>
                          {c.marine_risk_band} ({c.marine_risk_score ? Number(c.marine_risk_score).toFixed(0) : 65})
                        </span>
                      </td>
                      <td>
                        <code>{c.verification_status || 'unverified'}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}

      {tab === 'review' && (
        <ReviewQueue contacts={contacts} places={places} onVerify={handleVerify} verifyingId={verifyingId} />
      )}

      {tab === 'reports' && <Reports contacts={contacts} places={places} />}

      <footer className="footer">
        <span>
          <b>Underwater Ghost Net &amp; Marine Debris Detection Platform</b> · Sonar Acoustic AI
        </span>
        <span>Dual-Channel Side-Scan Sonar Analysis · Georeferenced Marine Obstacle Mapping</span>
      </footer>
    </div>
  );
}
