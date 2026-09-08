const API_BASE = import.meta.env.VITE_API_URL || '';

export async function fetchContacts(limit = 500) {
  const res = await fetch(`${API_BASE}/api/contacts?limit=${limit}`);
  if (!res.ok) throw new Error(`contacts failed: ${res.status}`);
  const data = await res.json();
  return data.contacts ?? [];
}

const placeCache = new Map();

export async function reversePlace(lat, lon) {
  const key = `${Number(lat).toFixed(4)},${Number(lon).toFixed(4)}`;
  if (placeCache.has(key)) return placeCache.get(key);
  const promise = (async () => {
    const res = await fetch(`${API_BASE}/api/reverse?lat=${lat}&lon=${lon}`);
    if (!res.ok) return `${Number(lat).toFixed(4)}, ${Number(lon).toFixed(4)}`;
    const data = await res.json();
    return data.place_name || data.display_name || key;
  })();
  placeCache.set(key, promise);
  return promise;
}

export async function searchPlaces(query) {
  const res = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}`);

  if (!res.ok) throw new Error(`search failed: ${res.status}`);
  const data = await res.json();
  return data.results ?? [];
}

export async function analyzeImage(file, params = {}) {
  const form = new FormData();
  form.append('file', file);
  for (const [k, v] of Object.entries(params)) form.append(k, String(v));
  const res = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`analysis failed: ${res.status} ${detail.slice(0, 200)}`);
  }
  return res.json();
}

export async function submitVerification(anomalyId, decision, notes = '') {
  const res = await fetch(`${API_BASE}/api/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ anomaly_id: anomalyId, decision, notes }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`verify failed: ${res.status} ${detail}`);
  }
  return res.json();
}

export async function fetchDatasetSummary() {
  const res = await fetch(`${API_BASE}/api/dataset/summary`);
  if (!res.ok) throw new Error(`dataset summary failed: ${res.status}`);
  return res.json();
}

export async function fetchDatasetSamples(split = 'train', limit = 30, offset = 0) {
  const res = await fetch(`${API_BASE}/api/dataset/samples?split=${split}&limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`dataset samples failed: ${res.status}`);
  return res.json();
}

export async function processDatasetSample(payload) {
  const res = await fetch(`${API_BASE}/api/dataset/process_sample`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`sample processing failed: ${res.status} ${detail.slice(0, 200)}`);
  }
  return res.json();
}

export async function parseMetadataFile(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/api/metadata/parse`, { method: 'POST', body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`metadata parsing failed: ${res.status} ${detail.slice(0, 160)}`);
  }
  return res.json();
}


