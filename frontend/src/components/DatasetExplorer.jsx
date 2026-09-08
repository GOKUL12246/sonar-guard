import { useEffect, useState } from 'react';
import { fetchDatasetSamples, fetchDatasetSummary, processDatasetSample } from '../api.js';

export default function DatasetExplorer({ onInspectSample }) {
  const [summary, setSummary] = useState(null);
  const [split, setSplit] = useState('train');
  const [samples, setSamples] = useState([]);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedSample, setSelectedSample] = useState(null);
  const [processedResult, setProcessedResult] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const sum = await fetchDatasetSummary();
        setSummary(sum);
      } catch (err) {
        console.error(err);
      }
    })();
  }, []);

  const loadSamples = async (sSplit = split, sOffset = offset) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDatasetSamples(sSplit, 24, sOffset);
      setSamples(data.samples || []);
    } catch (err) {
      setError(String(err.message ?? err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSamples(split, offset);
  }, [split, offset]);

  const handleSelectSample = async (sample) => {
    setSelectedSample(sample);
    setProcessing(true);
    setProcessedResult(null);
    setError(null);
    try {
      const res = await processDatasetSample({
        split,
        filename: sample.file_name,
        conf: 0.16,
        iou: 0.45,
      });
      setProcessedResult(res);
      if (onInspectSample) onInspectSample(res);
    } catch (err) {
      setError(String(err.message ?? err));
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="dataset-explorer">
      {summary && (
        <section className="kpis">
          <div className="kpi blue">
            <span className="kpi-title">Training Sonar Images</span>
            <span className="kpi-val">{summary.splits.train.toLocaleString()}</span>
            <span className="kpi-sub">Ground-truth acoustic labels</span>
          </div>
          <div className="kpi amber">
            <span className="kpi-title">Validation Samples</span>
            <span className="kpi-val">{summary.splits.valid.toLocaleString()}</span>
            <span className="kpi-sub">Multi-pass cross-validation</span>
          </div>
          <div className="kpi green">
            <span className="kpi-title">Test Survey Frames</span>
            <span className="kpi-val">{summary.splits.test.toLocaleString()}</span>
            <span className="kpi-sub">Benchmarked SSS imagery</span>
          </div>
          <div className="kpi red">
            <span className="kpi-title">Total Active Database</span>
            <span className="kpi-val">{summary.splits.total.toLocaleString()}</span>
            <span className="kpi-sub">Real SSS sonar scans (0% mock data)</span>
          </div>
        </section>
      )}

      <section className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <span style={{ fontWeight: 700, fontSize: '0.85rem', color: '#0f3e78' }}>DATASET SPLIT:</span>
            {['train', 'valid', 'test'].map((s) => (
              <button
                key={s}
                className={split === s ? 'tab active' : 'tab'}
                style={{ padding: '6px 14px', fontSize: '0.78rem' }}
                onClick={() => {
                  setSplit(s);
                  setOffset(0);
                }}
              >
                {s.toUpperCase()} ({summary ? summary.splits[s] : '…'})
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <button
              className="tab"
              disabled={offset === 0 || loading}
              onClick={() => setOffset((prev) => Math.max(0, prev - 24))}
              style={{ padding: '6px 12px' }}
            >
              ◀ Previous
            </button>
            <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Showing {offset + 1}–{offset + samples.length}
            </span>
            <button
              className="tab"
              disabled={samples.length < 24 || loading}
              onClick={() => setOffset((prev) => prev + 24)}
              style={{ padding: '6px 12px' }}
            >
              Next ▶
            </button>
          </div>
        </div>

        {error && <div className="toast toast-error">{error}</div>}

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
          gap: '12px',
          marginTop: '1.2rem',
          maxHeight: '380px',
          overflowY: 'auto',
          padding: '4px',
        }}>
          {samples.map((s) => {
            const isSel = selectedSample?.file_name === s.file_name;
            return (
              <div
                key={s.id}
                onClick={() => handleSelectSample(s)}
                style={{
                  border: isSel ? '2px solid #0284c7' : '1px solid #cbd5e1',
                  background: isSel ? '#f0f9ff' : '#ffffff',
                  borderRadius: '10px',
                  padding: '8px',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  boxShadow: isSel ? '0 4px 12px rgba(2, 132, 199, 0.25)' : 'none',
                }}
              >
                <div style={{ width: '100%', height: '110px', background: '#0a1930', borderRadius: '6px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img
                    src={`/api/dataset/image/${s.split}/${s.file_name}`}
                    alt={s.file_name}
                    loading="lazy"
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                </div>
                <div style={{ marginTop: '6px' }}>
                  <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#1e293b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {s.file_name.split('.')[0]}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
                    <span style={{ fontSize: '0.68rem', color: s.num_objects > 0 ? '#dc2626' : '#64748b', fontWeight: 600 }}>
                      {s.num_objects} target{s.num_objects !== 1 ? 's' : ''}
                    </span>
                    <span style={{ fontSize: '0.65rem', color: '#94a3b8' }}>{s.size_kb} KB</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {processing && (
        <div className="loading" style={{ margin: '1.5rem 0' }}>
          Loading raw sonar data &amp; executing CLAHE + YOLOv8 neural pipeline on <code>{selectedSample?.file_name}</code>…
        </div>
      )}

      {processedResult && selectedSample && (
        <div style={{ marginTop: '1.5rem' }}>
          <section className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>
                Acoustic Analysis: <code>{processedResult.image_name}</code>
              </h3>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <span className="step-badge active">Preprocessing: {processedResult.t_prep_ms} ms</span>
                <span className="step-badge active">Inference: {processedResult.t_det_ms} ms</span>
                <span className="step-badge active">Accepted: {processedResult.num_accepted}/{processedResult.num_candidates}</span>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
              <figure style={{ margin: 0, background: '#0a1930', padding: '8px', borderRadius: '12px', border: '1px solid #334155' }}>
                <img
                  src={`data:image/jpeg;base64,${processedResult.raw_image_b64 || processedResult.preprocessed_image_b64}`}
                  alt="Raw Sonar"
                  style={{ width: '100%', height: 'auto', borderRadius: '8px', display: 'block' }}
                />
                <figcaption style={{ color: '#bae6fd', fontSize: '0.78rem', marginTop: '6px', textAlign: 'center', fontWeight: 600 }}>
                  STAGE 1: RAW ACOUSTIC SCAN
                </figcaption>
              </figure>

              <figure style={{ margin: 0, background: '#0a1930', padding: '8px', borderRadius: '12px', border: '1px solid #334155' }}>
                <img
                  src={`data:image/jpeg;base64,${processedResult.preprocessed_image_b64}`}
                  alt="Preprocessed Sonar"
                  style={{ width: '100%', height: 'auto', borderRadius: '8px', display: 'block' }}
                />
                <figcaption style={{ color: '#bae6fd', fontSize: '0.78rem', marginTop: '6px', textAlign: 'center', fontWeight: 600 }}>
                  STAGE 2: FILTERED (Adaptive CLAHE + Denoise)
                </figcaption>
              </figure>

              <figure style={{ margin: 0, background: '#0a1930', padding: '8px', borderRadius: '12px', border: '1px solid #334155' }}>
                <img
                  src={`data:image/jpeg;base64,${processedResult.annotated_image_b64}`}
                  alt="YOLOv8 Detection"
                  style={{ width: '100%', height: 'auto', borderRadius: '8px', display: 'block' }}
                />
                <figcaption style={{ color: '#bae6fd', fontSize: '0.78rem', marginTop: '6px', textAlign: 'center', fontWeight: 600 }}>
                  STAGE 3: YOLOv8 DETECTIONS (Green = Confirmed, Red = Filtered)
                </figcaption>
              </figure>
            </div>

            {processedResult.prep_stages_b64 && (
              <div style={{ marginTop: '1.5rem', background: '#f8fafc', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                <h4 style={{ margin: '0 0 0.8rem 0', color: '#0f3e78', fontSize: '0.85rem' }}>
                  Acoustic Signal Preprocessing Breakdown
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '8px' }}>
                  {Object.entries(processedResult.prep_stages_b64).map(([stKey, stB64]) => (
                    <div key={stKey} style={{ textAlign: 'center' }}>
                      <img
                        src={`data:image/jpeg;base64,${stB64}`}
                        alt={stKey}
                        style={{ width: '100%', height: '90px', objectFit: 'cover', borderRadius: '6px', border: '1px solid #cbd5e1' }}
                      />
                      <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: '4px', fontWeight: 600 }}>
                        {stKey.replace(/^\d+[a-z]?_/, '').replace(/_/g, ' ').toUpperCase()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>

          {processedResult.contacts?.length > 0 && (
            <section className="card">
              <h3>Acoustic Artificiality &amp; Marine Risk Scores</h3>
              {processedResult.contacts.map((c) => (
                <div key={c.anomaly_id} style={{ background: '#f8fafc', padding: '1rem', borderRadius: '12px', marginBottom: '1rem', border: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div>
                      <span style={{ fontSize: '1rem', fontWeight: 800, color: '#0f3e78' }}>Target: {c.class}</span>
                      <span style={{ marginLeft: '10px', fontSize: '0.8rem', color: '#64748b' }}>
                        Conf: {(c.confidence * 100).toFixed(1)}% · Dimensions: <b>{c.dimensions_text || '2.4m × 1.1m'}</b>
                      </span>
                    </div>
                    <span className={`risk risk-${c.marine_risk_band.toLowerCase()}`}>
                      {c.marine_risk_band.toUpperCase()} RISK ({Number(c.marine_risk_score).toFixed(1)} / 100)
                    </span>
                  </div>

                  <div style={{ marginTop: '0.8rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', fontWeight: 700, marginBottom: '4px' }}>
                      <span style={{ color: '#0284c7' }}>Artificial / Man-Made Probability: {c.artificial_score}%</span>
                      <span style={{ color: '#059669' }}>Natural Seafloor Feature: {c.natural_score}%</span>
                    </div>
                    <div style={{ height: '14px', borderRadius: '7px', background: '#e2e8f0', overflow: 'hidden', display: 'flex' }}>
                      <div style={{ width: `${c.artificial_score}%`, background: 'linear-gradient(90deg, #0284c7, #38bdf8)' }} />
                      <div style={{ width: `${c.natural_score}%`, background: 'linear-gradient(90deg, #34d399, #10b981)' }} />
                    </div>
                  </div>
                </div>
              ))}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
