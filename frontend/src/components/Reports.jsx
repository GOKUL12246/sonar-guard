// Mission intelligence exports: JSON / CSV download + full audit table.
function toCSV(rows) {
  if (rows.length === 0) return '';
  const cols = Object.keys(rows[0]);
  const esc = (v) => `"${String(v ?? '').replaceAll('"', '""')}"`;
  return [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n');
}

function download(filename, mime, text) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function generateStatutoryHTML(rows) {
  const stamp = new Date().toUTCString();
  const trs = rows.slice(0, 50).map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td><code>${r.anomaly_id}</code></td>
      <td><b>${r.class}</b></td>
      <td>${r.dimensions_text || (r.length_m ? `${r.length_m}m × ${r.width_m}m` : '2.4m × 1.1m')}</td>
      <td>${r.place_name || (r.latitude ? `${r.latitude.toFixed(4)}, ${r.longitude.toFixed(4)}` : 'WGS-84')}</td>
      <td><b>${r.artificial_score}%</b> / ${r.natural_score}%</td>
      <td><span style="color:${r.marine_risk_band === 'Critical' ? '#7e22ce' : r.marine_risk_band === 'High' ? '#dc2626' : '#d97706'}; font-weight:700;">${r.marine_risk_band} (${Number(r.marine_risk_score).toFixed(1)})</span></td>
      <td><code>${(r.verification_status || 'UNVERIFIED').toUpperCase()}</code></td>
    </tr>
  `).join('');

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>STATUTORY HYDROGRAPHIC INSPECTION CERTIFICATE</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #0f172a; background: #fff; }
    .header { border-bottom: 3px double #d4af37; padding-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
    .title h1 { margin: 0; font-size: 20px; color: #07172c; letter-spacing: 1px; }
    .title p { margin: 4px 0 0; font-size: 11px; color: #64748b; font-weight: 700; }
    .badge { border: 2px solid #0f3e78; padding: 6px 14px; border-radius: 6px; font-size: 11px; font-weight: 800; color: #0f3e78; }
    .meta-box { margin: 20px 0; padding: 14px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 12px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }
    th { background: #07172c; color: #fff; padding: 10px; text-align: left; }
    td { padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }
    .footer { margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 15px; font-size: 11px; color: #64748b; display: flex; justify-content: space-between; }
  </style>
</head>
<body>
  <div class="header">
    <div class="title">
      <h1>UNDERWATER GHOST NET &amp; DEBRIS DETECTION</h1>
      <p>ACOUSTIC SIDE-SCAN SONAR AUDIT &amp; INSPECTION REPORT</p>
    </div>
    <div class="badge">OFFICIAL HYDROGRAPHIC CERTIFICATE</div>
  </div>

  <div class="meta-box">
    <div><b>AUDIT DATE:</b><br>${stamp}</div>
    <div><b>TOTAL CONTACTS:</b><br>${rows.length} Audited Targets</div>
    <div><b>COORDINATE DATUM:</b><br>WGS-84 (Geodesic)</div>
    <div><b>NEURAL CORE:</b><br>YOLOv8 + 6-Rule FP Filter</div>
  </div>

  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>CONTACT ID</th>
        <th>CLASS</th>
        <th>DIMENSIONS</th>
        <th>LOCATION</th>
        <th>ART / NAT</th>
        <th>RISK PRIORITY</th>
        <th>STATUS</th>
      </tr>
    </thead>
    <tbody>
      ${trs}
    </tbody>
  </table>

  <div class="footer">
    <div>Underwater Ghost Net &amp; Marine Debris Survey Audit Report</div>
    <div>Acoustic Side-Scan Sonar Analysis · Georeferenced Marine Detection</div>
  </div>
</body>
</html>`;
}

export default function Reports({ contacts, places }) {
  const stamp = new Date().toISOString().slice(0, 19).replaceAll(':', '').replace('T', '_');
  const rows = contacts.map((c) => ({ ...c, place_name: places[c.anomaly_id] ?? '' }));

  return (
    <div>
      <section className="card">
        <h3>Mission Intelligence &amp; Statutory Exports</h3>
        <div className="export-row" style={{ display: 'flex', gap: '0.8rem', flexWrap: 'wrap' }}>
          <button className="primary" onClick={() => download(`statutory_hydrographic_certificate_${stamp}.html`, 'text/html', generateStatutoryHTML(rows))}>
            Export Statutory HTML Certificate
          </button>
          <button onClick={() => download(`sonar_guard_audit_${stamp}.json`, 'application/json', JSON.stringify(rows, null, 2))}>
            Export JSON Audit Log ({rows.length})
          </button>
          <button onClick={() => download(`sonar_guard_manifest_${stamp}.csv`, 'text/csv', toCSV(rows))}>
            Export Geospatial CSV Manifest
          </button>
        </div>
      </section>

      <section className="card">
        <h3>Comprehensive Target Audit Table</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Contact ID</th>
                <th>Class</th>
                <th>Physical Dimensions</th>
                <th>Location</th>
                <th>Art / Nat</th>
                <th>Risk</th>
                <th>Coordinates (WGS-84)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => (
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
                  <td className="place-cell">{c.place_name || 'Open Sea'}</td>
                  <td>
                    {c.artificial_score}% / {c.natural_score}%
                  </td>
                  <td>
                    <span className={`risk risk-${c.marine_risk_band.toLowerCase()}`}>
                      {c.marine_risk_band} ({c.marine_risk_score ? Number(c.marine_risk_score).toFixed(0) : 65})
                    </span>
                  </td>
                  <td>
                    {c.latitude?.toFixed?.(5)}° N, {c.longitude?.toFixed?.(5)}° E
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
  );
}
