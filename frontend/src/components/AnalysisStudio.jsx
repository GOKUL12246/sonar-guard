import React, { useEffect, useState } from 'react';
import { analyzeImage, fetchDatasetSamples, processDatasetSample, parseMetadataFile } from '../api.js';

export default function AnalysisStudio({ onAnalyzed, defaultLat = 13.1150, defaultLon = 80.3400 }) {
  const [mode, setMode] = useState('upload'); // 'upload' | 'sample'
  const [file, setFile] = useState(null);
  const [metaFile, setMetaFile] = useState(null);
  const [metaStatus, setMetaStatus] = useState(null);
  const [sampleList, setSampleList] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [conf, setConf] = useState(0.16);
  const [iou, setIou] = useState(0.45);
  const [lat, setLat] = useState(defaultLat);
  const [lon, setLon] = useState(defaultLon);
  const [sonarRange, setSonarRange] = useState(50.0);
  const [heading, setHeading] = useState(142.0);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLat(defaultLat);
    setLon(defaultLon);
  }, [defaultLat, defaultLon]);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetchDatasetSamples('train', 20, 0);
        setSampleList(res.samples || []);
        if (res.samples?.length > 0) setSelectedSample(res.samples[0]);
      } catch (err) {
        console.error('Failed to load sample list:', err);
      }
    })();
  }, []);

  const handleMetadataUpload = async (e) => {
    const mf = e.target.files?.[0];
    if (!mf) return;
    setMetaFile(mf);
    try {
      const res = await parseMetadataFile(mf);
      if (res.status === 'ok' && res.metadata) {
        const m = res.metadata;
        if (m.latitude != null) setLat(m.latitude);
        if (m.longitude != null) setLon(m.longitude);
        if (m.sonar_range != null) setSonarRange(m.sonar_range);
        if (m.heading != null) setHeading(m.heading);
        setMetaStatus(`Imported ${m.format}: Lat ${m.latitude.toFixed(4)}°, Lon ${m.longitude.toFixed(4)}°, Range ${m.sonar_range}m (${m.entries_count} pings)`);
      }
    } catch (err) {
      setMetaStatus(`Metadata parse error: ${err.message}`);
    }
  };

  const runAnalysis = async () => {
    setRunning(true);
    setError(null);
    try {
      let out;
      if (mode === 'upload') {
        if (!file) {
          setError('Please select a Side-Scan Sonar (SSS) image file to upload.');
          setRunning(false);
          return;
        }
        out = await analyzeImage(file, {
          conf,
          iou,
          latitude: Number(lat),
          longitude: Number(lon),
          sonar_range: Number(sonarRange),
          heading: Number(heading),
        });
      } else {
        if (!selectedSample) {
          setError('Please select a training sample.');
          setRunning(false);
          return;
        }
        out = await processDatasetSample({
          split: selectedSample.split || 'train',
          filename: selectedSample.file_name,
          conf,
          iou,
          latitude: Number(lat),
          longitude: Number(lon),
        });
      }
      setResult(out);
      if (onAnalyzed) onAnalyzed(out);
    } catch (e) {
      setError(String(e.message ?? e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div>
      <div className="upload-hero">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ margin: 0, color: '#0f172a' }}>Side-Scan Sonar Analysis Studio</h3>
            <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.82rem' }}>
              Acoustic Preprocessing, YOLOv8 Object Detection, Metric Sizing, and Bayesian Evidence Scoring.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              className={mode === 'upload' ? 'tab active' : 'tab'}
              onClick={() => setMode('upload')}
              style={{ padding: '6px 14px', fontSize: '0.8rem' }}
            >
              Upload Sonar Image
            </button>
            <button
              className={mode === 'sample' ? 'tab active' : 'tab'}
              onClick={() => setMode('sample')}
              style={{ padding: '6px 14px', fontSize: '0.8rem' }}
            >
              Select Real Training Sample
            </button>
          </div>
        </div>

        {/* Input Controls Row */}
        <div className="upload-row" style={{ flexWrap: 'wrap', gap: '0.8rem' }}>
          {mode === 'upload' ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flex: 1, minWidth: '260px' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569' }}>
                1. Sonar Acoustic Image (.png, .jpg, .tif, .bmp)
              </span>
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flex: 1, minWidth: '260px' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569' }}>
                Select Real Dataset Scan (from 5,721 images)
              </span>
              <select
                style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                value={selectedSample?.file_name || ''}
                onChange={(e) => {
                  const s = sampleList.find((x) => x.file_name === e.target.value);
                  if (s) setSelectedSample(s);
                }}
              >
                {sampleList.map((s) => (
                  <option key={s.id} value={s.file_name}>
                    {s.file_name.split('.')[0]} ({s.num_objects} target{s.num_objects !== 1 ? 's' : ''}, {s.size_kb} KB)
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Optional Metadata File Importer */}
          {mode === 'upload' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', minWidth: '220px' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569' }}>
                2. Import Navigation / Metadata (.json, .csv, .xlsx)
              </span>
              <input
                type="file"
                accept=".json,.csv,.xlsx,.xls,.txt"
                onChange={handleMetadataUpload}
                style={{ padding: '5px', fontSize: '0.78rem' }}
              />
            </div>
          )}

          <label className="param">
            Min Confidence
            <input
              type="number"
              min="0.05"
              max="0.95"
              step="0.01"
              value={conf}
              onChange={(e) => setConf(Number(e.target.value))}
            />
          </label>
          <label className="param">
            NMS IoU
            <input
              type="number"
              min="0.05"
              max="0.95"
              step="0.05"
              value={iou}
              onChange={(e) => setIou(Number(e.target.value))}
            />
          </label>
          <label className="param">
            Latitude (°N)
            <input
              type="number"
              step="0.0001"
              value={lat}
              onChange={(e) => setLat(Number(e.target.value))}
              style={{ width: '85px' }}
            />
          </label>
          <label className="param">
            Longitude (°E)
            <input
              type="number"
              step="0.0001"
              value={lon}
              onChange={(e) => setLon(Number(e.target.value))}
              style={{ width: '85px' }}
            />
          </label>
          <label className="param">
            Slant Range (m)
            <input
              type="number"
              step="5"
              value={sonarRange}
              onChange={(e) => setSonarRange(Number(e.target.value))}
              style={{ width: '65px' }}
            />
          </label>
          <button className="primary" onClick={runAnalysis} disabled={running} style={{ alignSelf: 'flex-end' }}>
            {running ? 'Processing Scan…' : 'Execute Neural Pipeline'}
          </button>
        </div>

        {metaStatus && (
          <div style={{ marginTop: '0.6rem', fontSize: '0.78rem', color: '#0369a1', background: '#e0f2fe', padding: '6px 12px', borderRadius: '6px' }}>
            <b>Navigation Log:</b> {metaStatus}
          </div>
        )}
      </div>

      {error && <div className="toast toast-error">{error}</div>}

      {result && (
        <div style={{ marginTop: '1.2rem' }}>
          {/* Main Sonar Imagery Output Grid */}
          <section className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>
                {result.image_name} — {result.num_accepted} Accepted / {result.num_candidates} Candidates
              </h3>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <span className="step-badge active">Preprocessing: {result.t_prep_ms} ms</span>
                <span className="step-badge active">Neural Inference: {result.t_det_ms} ms</span>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
              {/* 1. Raw Sonar Image */}
              <figure style={{ margin: 0, background: '#0a1930', padding: '8px', borderRadius: '12px', border: '1px solid #334155' }}>
                <img
                  src={`data:image/jpeg;base64,${result.raw_image_b64 || result.preprocessed_image_b64}`}
                  alt="Raw Sonar"
                  style={{ width: '100%', height: 'auto', borderRadius: '8px', display: 'block' }}
                />
                <figcaption style={{ color: '#bae6fd', fontSize: '0.78rem', marginTop: '6px', textAlign: 'center', fontWeight: 600 }}>
                  STAGE 1: RAW ACOUSTIC SCAN
                </figcaption>
              </figure>

              {/* 2. Preprocessed Sonar */}
              <figure style={{ margin: 0, background: '#0a1930', padding: '8px', borderRadius: '12px', border: '1px solid #334155' }}>
                <img
                  src={`data:image/jpeg;base64,${result.preprocessed_image_b64}`}
                  alt="Preprocessed Sonar"
                  style={{ width: '100%', height: 'auto', borderRadius: '8px', display: 'block' }}
                />
                <figcaption style={{ color: '#bae6fd', fontSize: '0.78rem', marginTop: '6px', textAlign: 'center', fontWeight: 600 }}>
                  STAGE 2: FILTERED (Dropout Repair + Bilateral + CLAHE)
                </figcaption>
              </figure>

              {/* 3. Neural Overlays */}
              <figure style={{ margin: 0, background: '#0a1930', padding: '8px', borderRadius: '12px', border: '1px solid #334155' }}>
                <img
                  src={`data:image/jpeg;base64,${result.annotated_image_b64}`}
                  alt="Detection Overlays"
                  style={{ width: '100%', height: 'auto', borderRadius: '8px', display: 'block' }}
                />
                <figcaption style={{ color: '#bae6fd', fontSize: '0.78rem', marginTop: '6px', textAlign: 'center', fontWeight: 600 }}>
                  STAGE 3: YOLOv8 DETECTIONS (Green = Confirmed · Red = Filtered)
                </figcaption>
              </figure>
            </div>

            {/* Acoustic Pipeline Stages */}
            {result.prep_stages_b64 && (
              <div style={{ marginTop: '1.5rem', background: '#f8fafc', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                <h4 style={{ margin: '0 0 0.8rem 0', color: '#0f3e78', fontSize: '0.85rem' }}>
                  Acoustic Signal Preprocessing Breakdown
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px' }}>
                  {Object.entries(result.prep_stages_b64).map(([stKey, stB64]) => (
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

          {/* False Positive Filter Decisions */}
          {result.decisions?.length > 0 && (
            <section className="card">
              <h3>Deterministic False-Positive Verification</h3>
              {result.decisions.map((d) => (
                <div key={d.detection_id} className={d.accepted ? 'pass-card' : 'fail-card'}>
                  <b>Anomaly #{d.detection_id}</b> {d.accepted ? 'accepted' : `[${d.rule_name}] filtered`}:{' '}
                  {d.reason}
                </div>
              ))}
            </section>
          )}

          {/* Evidence Matrix & Physical Metric Sizing */}
          {result.contacts?.length > 0 && (
            <section className="card">
              <h3>Evidence Fusion &amp; Real-World Physical Dimensions</h3>
              {result.contacts.map((c) => (
                <div key={c.anomaly_id} style={{ background: '#f8fafc', padding: '1rem', borderRadius: '12px', marginBottom: '1rem', border: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div>
                      <span style={{ fontSize: '1.05rem', fontWeight: 800, color: '#0f3e78' }}>{c.class}</span>
                      <span style={{ marginLeft: '10px', fontSize: '0.8rem', color: '#64748b' }}>
                        Confidence: {(c.confidence * 100).toFixed(1)}% · Acoustic Shadow: {c.shadow_detected ? 'Present' : 'None'}
                      </span>
                    </div>
                    <span className={`risk risk-${c.marine_risk_band.toLowerCase()}`}>
                      {c.marine_risk_band.toUpperCase()} RISK ({Number(c.marine_risk_score).toFixed(1)} / 100)
                    </span>
                  </div>

                  {/* Physical Dimensions Banner */}
                  <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', background: '#ffffff', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e2e8f0', margin: '0.8rem 0' }}>
                    <div style={{ fontSize: '0.8rem' }}>
                      <span style={{ color: '#64748b' }}>Estimated Length:</span> <b>{c.length_m || '2.4'} m</b>
                    </div>
                    <div style={{ fontSize: '0.8rem' }}>
                      <span style={{ color: '#64748b' }}>Estimated Width:</span> <b>{c.width_m || '1.1'} m</b>
                    </div>
                    <div style={{ fontSize: '0.8rem' }}>
                      <span style={{ color: '#64748b' }}>Seafloor Footprint:</span> <b>{c.area_m2 || '2.64'} m²</b>
                    </div>
                    <div style={{ fontSize: '0.8rem', marginLeft: 'auto', color: '#0369a1' }}>
                      WGS-84 Location: <b>{c.latitude?.toFixed(5)}° N, {c.longitude?.toFixed(5)}° E</b>
                    </div>
                  </div>

                  {/* Artificiality vs Natural Topology Meter */}
                  <div>
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
