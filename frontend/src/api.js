// Intelligent Dual-Target API Caller with Auto-Failover
const isLocalhost = typeof window !== 'undefined' && (
  window.location.hostname === 'localhost' ||
  window.location.hostname === '127.0.0.1'
);

const PRIMARY_API = import.meta.env.VITE_API_URL || (isLocalhost ? '' : 'https://sonar-guard.onrender.com');
const FALLBACK_API = isLocalhost ? 'http://127.0.0.1:8000' : '';

async function smartFetch(path, options = {}) {
  // First attempt with primary target
  try {
    const primaryUrl = `${PRIMARY_API}${path}`;
    const res = await fetch(primaryUrl, options);
    if (res.ok) return res;
  } catch (err) {
    // If primary failed on localhost, retry with explicit direct localhost:8000
    if (FALLBACK_API) {
      try {
        const fallbackUrl = `${FALLBACK_API}${path}`;
        const res2 = await fetch(fallbackUrl, options);
        if (res2.ok) return res2;
      } catch (_) {}
    }
    // If running on cloud and Render is cold, try direct cloud url
    if (!PRIMARY_API.includes('onrender.com')) {
      try {
        const cloudUrl = `https://sonar-guard.onrender.com${path}`;
        const res3 = await fetch(cloudUrl, options);
        if (res3.ok) return res3;
      } catch (_) {}
    }
  }

  // Final direct attempt to throw proper response / error
  const finalUrl = `${PRIMARY_API || 'http://127.0.0.1:8000'}${path}`;
  return await fetch(finalUrl, options);
}

export async function fetchContacts(limit = 500) {
  const res = await smartFetch(`/api/contacts?limit=${limit}`);
  if (!res.ok) throw new Error(`contacts failed: ${res.status}`);
  const data = await res.json();
  return data.contacts ?? [];
}

const placeCache = new Map();

export async function reversePlace(lat, lon) {
  const key = `${Number(lat).toFixed(4)},${Number(lon).toFixed(4)}`;
  if (placeCache.has(key)) return placeCache.get(key);
  const promise = (async () => {
    try {
      const res = await smartFetch(`/api/reverse?lat=${lat}&lon=${lon}`);
      if (!res.ok) return `${Number(lat).toFixed(4)}, ${Number(lon).toFixed(4)}`;
      const data = await res.json();
      return data.place_name || data.display_name || key;
    } catch {
      return `${Number(lat).toFixed(4)}, ${Number(lon).toFixed(4)}`;
    }
  })();
  placeCache.set(key, promise);
  return promise;
}

export async function searchPlaces(query) {
  const res = await smartFetch(`/api/search?q=${encodeURIComponent(query)}`);
  if (!res.ok) throw new Error(`search failed: ${res.status}`);
  const data = await res.json();
  return data.results ?? [];
}

export async function analyzeImage(file, params = {}) {
  const form = new FormData();
  form.append('file', file);
  for (const [k, v] of Object.entries(params)) form.append(k, String(v));
  try {
    const res = await smartFetch('/api/analyze', { method: 'POST', body: form });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Analysis error (${res.status}): ${detail.slice(0, 200)}`);
    }
    return await res.json();
  } catch (err) {
    if (err.message && err.message.includes('Analysis error')) throw err;
    throw new Error(`Network Error: Backend server is unreachable. Please verify local server is running on port 8000 or wait for cloud container warmup.`);
  }
}

export async function submitVerification(anomalyId, decision, notes = '') {
  const res = await smartFetch('/api/verify', {
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
  const res = await smartFetch('/api/dataset/summary');
  if (!res.ok) throw new Error(`dataset summary failed: ${res.status}`);
  return res.json();
}

export async function fetchDatasetSamples(split = 'train', limit = 30, offset = 0) {
  const res = await smartFetch(`/api/dataset/samples?split=${split}&limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`dataset samples failed: ${res.status}`);
  return res.json();
}

export async function processDatasetSample(payload) {
  const res = await smartFetch('/api/dataset/process_sample', {
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
  const res = await smartFetch('/api/metadata/parse', { method: 'POST', body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`metadata parsing failed: ${res.status} ${detail.slice(0, 160)}`);
  }
  return res.json();
}


