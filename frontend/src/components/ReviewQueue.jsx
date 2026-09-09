import { useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '';

export default function ReviewQueue({ contacts, places, onVerify, verifyingId }) {
  const [filter, setFilter] = useState('pending');

  const items = useMemo(() => {
    let list = [...contacts];
    if (filter === 'pending')
      list = list.filter((c) => ['unverified', 'pending'].includes(c.verification_status));
    else if (filter === 'confirmed') list = list.filter((c) => c.verification_status === 'confirmed');
    else if (filter === 'rejected') list = list.filter((c) => c.verification_status === 'rejected');
    else if (filter === 'rov') list = list.filter((c) => c.verification_status === 'rov_inspection');
    return list.sort((a, b) => (Number(b.marine_risk_score) || 0) - (Number(a.marine_risk_score) || 0));
  }, [contacts, filter]);

  return (
    <div>
      <div className="controls">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
          <strong>Operator Verification &amp; Active Retraining Queue</strong>
          <span className="step-badge active">{items.length} Contacts</span>
        </div>
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="pending">Pending Review</option>
          <option value="all">All Contacts ({contacts.length})</option>
          <option value="confirmed">Confirmed Debris</option>
          <option value="rejected">Rejected (False Positives)</option>
          <option value="rov">ROV Inspection Required</option>
        </select>
      </div>

      {items.length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
          <h4>No acoustic contacts currently in this filter.</h4>
          <p style={{ color: '#64748b', fontSize: '0.85rem' }}>
            All queued items have been processed or select "All Contacts" in the filter above.
          </p>
        </div>
      )}

      {items.map((c) => {
        const riskVal = Number(c.marine_risk_score) || 0;
        const artVal = Number(c.artificial_score) || 0;
        const natVal = Number(c.natural_score) || (100 - artVal);
        const place = places[c.anomaly_id] ?? 'Resolving WGS-84 position…';
        const imgName = c.source_image ? String(c.source_image).split(/[\\/]/).pop() : null;
        const imgSrc = c.thumbnail_b64
          ? (c.thumbnail_b64.startsWith('data:') ? c.thumbnail_b64 : `data:image/jpeg;base64,${c.thumbnail_b64}`)
          : c.image_b64
          ? (c.image_b64.startsWith('data:') ? c.image_b64 : `data:image/jpeg;base64,${c.image_b64}`)
          : imgName
          ? `${API_BASE}/api/dataset/image/train/${imgName}`
          : null;

        return (
          <details key={c.anomaly_id} className="review-item" open={filter === 'pending'}>
            <summary>
              <b>{c.anomaly_id.slice(0, 13)}</b> · {c.class} · Dimensions: <b>{c.dimensions_text || (c.length_m ? `${c.length_m}m × ${c.width_m}m` : '2.4m × 1.1m')}</b> · Art <b>{artVal}%</b> / Nat <b>{natVal}%</b> ·{' '}
              <span className={`risk risk-${(c.marine_risk_band || 'medium').toLowerCase()}`}>
                {c.marine_risk_band || 'Medium'} ({riskVal.toFixed(0)})
              </span>{' '}
              · <code>{c.verification_status || 'unverified'}</code>
            </summary>
            <div className="review-body">
              <div style={{ display: 'flex', gap: '1.2rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
                {/* Sonar Thumbnail Preview */}
                <div
                  style={{
                    width: '140px',
                    height: '115px',
                    background: '#071529',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    flexShrink: 0,
                    border: '1px solid #1e3a8a',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative',
                  }}
                >
                  {imgSrc ? (
                    <img
                      src={imgSrc}
                      alt={imgName || 'Sonar Contact'}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  ) : (
                    <div style={{ textAlign: 'center', padding: '0.4rem' }}>
                      <div style={{ fontSize: '0.68rem', fontWeight: 800, color: '#38bdf8', letterSpacing: '0.5px' }}>SONAR TARGET</div>
                      <div style={{ fontSize: '0.74rem', fontWeight: 600, color: '#f8fafc', marginTop: '2px' }}>{c.class || 'Ghost-Net'}</div>
                      <div style={{ fontSize: '0.66rem', color: '#94a3b8', marginTop: '2px' }}>{c.dimensions_text || `${c.length_m || 2.4}m × ${c.width_m || 1.1}m`}</div>
                    </div>
                  )}
                </div>
                <div style={{ flex: 1, minWidth: '220px' }}>
                  <div style={{ fontSize: '0.92rem', marginBottom: '4px' }}>
                    <b>{place}</b>
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#64748b' }}>
                    Position: {c.latitude != null ? Number(c.latitude).toFixed(5) : '—'}° N, {c.longitude != null ? Number(c.longitude).toFixed(5) : '—'}° E · Depth: {c.depth_m || 28.5} m
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: '3px' }}>
                    Source Frame: <code>{c.source_image}</code> · Estimated Footprint: <b>{c.area_m2 || '2.64'} m²</b>
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#0f3e78', marginTop: '3px', fontWeight: 600 }}>
                    Telemetry Note: {c.notes || 'Debris target requiring operator validation'}
                  </div>

                  {/* Dual Score Bar */}
                  <div style={{ marginTop: '8px', maxWidth: '380px' }}>
                    <div style={{ height: '8px', borderRadius: '4px', background: '#e2e8f0', overflow: 'hidden', display: 'flex' }}>
                      <div style={{ width: `${artVal}%`, background: '#0284c7' }} />
                      <div style={{ width: `${natVal}%`, background: '#10b981' }} />
                    </div>
                  </div>
                </div>
              </div>

              <div className="popup-actions" style={{ marginTop: '0.9rem' }}>
                <button
                  disabled={verifyingId === c.anomaly_id}
                  onClick={() => onVerify(c.anomaly_id, 'confirmed')}
                  style={{ background: '#059669', color: '#fff', fontWeight: 700 }}
                >
                  Confirm Obstacle
                </button>
                <button
                  disabled={verifyingId === c.anomaly_id}
                  onClick={() => onVerify(c.anomaly_id, 'rejected')}
                  style={{ background: '#dc2626', color: '#fff', fontWeight: 700 }}
                >
                  Reject False Positive
                </button>
                <button
                  disabled={verifyingId === c.anomaly_id}
                  onClick={() => onVerify(c.anomaly_id, 'rov_inspection')}
                  style={{ background: '#7c3aed', color: '#fff', fontWeight: 700 }}
                >
                  Dispatch ROV Retrieval
                </button>
              </div>
            </div>
          </details>
        );
      })}
    </div>
  );
}
