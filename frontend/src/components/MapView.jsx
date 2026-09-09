import { useEffect, useMemo, useState } from 'react';
import {
  CircleMarker,
  LayersControl,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
  useMap,
} from 'react-leaflet';
import L from 'leaflet';
import { reversePlace } from '../api.js';

const RISK_COLORS = {
  Critical: '#7e22ce',
  High: '#dc2626',
  Medium: '#d97706',
  Low: '#059669',
};

// Imperatively fly the map when the operator searches a place.
function FlyTo({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) map.flyTo(center, 13, { duration: 1.2 });
  }, [center, map]);
  return null;
}

export default function MapView({ contacts, center, activeBands, onVerify, verifyingId }) {
  const [places, setPlaces] = useState({}); // anomaly_id -> real place name

  // Resolve each visible contact's real place name (backend-cached Nominatim).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const entries = await Promise.all(
        contacts.map(async (c) => {
          if (c.latitude == null || c.longitude == null) return [c.anomaly_id, ''];
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

  const visible = useMemo(
    () => contacts.filter((c) => activeBands.includes(c.marine_risk_band)),
    [contacts, activeBands],
  );

  const track = useMemo(
    () =>
      visible
        .filter((c) => c.latitude != null && c.longitude != null)
        .map((c) => [c.latitude, c.longitude]),
    [visible],
  );

  const mapCenter = center ?? [13.1150, 80.3400];

  return (
    <MapContainer
      center={mapCenter}
      zoom={13}
      scrollWheelZoom
      style={{ height: '560px', width: '100%', borderRadius: '16px' }}
    >
      <FlyTo center={center} />
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="OpenStreetMap Standard">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Esri Satellite Imagery">
          <TileLayer
            attribution="Esri World Imagery · Maxar · Earthstar Geographics"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.BaseLayer>
        <LayersControl.Overlay checked name="Place &amp; Nautical Labels">
          <TileLayer
            attribution="Esri Reference — boundaries & places"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.Overlay>
      </LayersControl>

      {track.length > 1 && (
        <Polyline positions={track} pathOptions={{ color: '#0ea5e9', weight: 3, dashArray: '6 10' }} />
      )}

      {visible.map((c) => {
        if (c.latitude == null || c.longitude == null) return null;
        const color = RISK_COLORS[c.marine_risk_band] ?? '#0284c7';
        const place = places[c.anomaly_id] ?? 'resolving place…';
        return (
          <CircleMarker
            key={c.anomaly_id}
            center={[c.latitude, c.longitude]}
            radius={c.marine_risk_band === 'Critical' ? 12 : 9}
            pathOptions={{ color, fillColor: color, fillOpacity: 0.55, weight: 2 }}
          >
            <Tooltip direction="top" offset={[0, -12]} opacity={0.95}>
              <span className="marker-label">{place}</span>
            </Tooltip>
            <Popup>
              <div className="popup">
                <strong>
                  {c.anomaly_id.slice(0, 13)} — {c.class}
                </strong>
                {c.thumbnail_b64 && (
                  <div style={{ margin: '6px 0', borderRadius: '6px', overflow: 'hidden', border: '1px solid #0f3e78', maxHeight: '110px' }}>
                    <img
                      src={c.thumbnail_b64.startsWith('data:') ? c.thumbnail_b64 : `data:image/jpeg;base64,${c.thumbnail_b64}`}
                      alt={c.class || 'SSS Acoustic Contact'}
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    />
                  </div>
                )}
                <div>
                  Dimensions: <b>{c.dimensions_text || (c.length_m ? `${c.length_m}m × ${c.width_m}m` : '2.4m × 1.1m')}</b>
                </div>
                <div>
                  Artificial <b>{c.artificial_score}%</b> · Natural <b>{c.natural_score}%</b>
                </div>
                <div>
                  Risk <b>{c.marine_risk_band}</b> ({Number(c.marine_risk_score).toFixed(1)} / 100)
                </div>
                <div>
                  Coords: {Number(c.latitude).toFixed(5)}° N, {Number(c.longitude).toFixed(5)}° E · Depth {c.depth_m} m
                </div>
                <div>
                  Status: <code>{c.verification_status || 'unverified'}</code>
                </div>
                <div className="popup-actions">
                  <button
                    disabled={verifyingId === c.anomaly_id}
                    onClick={() => onVerify(c.anomaly_id, 'confirmed')}
                  >
                    ✅ Confirm
                  </button>
                  <button
                    disabled={verifyingId === c.anomaly_id}
                    onClick={() => onVerify(c.anomaly_id, 'rejected')}
                  >
                    ❌ Reject
                  </button>
                  <button
                    disabled={verifyingId === c.anomaly_id}
                    onClick={() => onVerify(c.anomaly_id, 'rov_inspection')}
                  >
                    🤿 ROV
                  </button>
                </div>
              </div>
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}

// Silence the missing default-icon warning: we only use vector CircleMarkers.
delete L.Icon.Default.prototype._getIconUrl;
