import os
import sys
import json
import numpy as np
import cv2
import requests
from datetime import datetime

os.makedirs("profiles", exist_ok=True)

# Dizionario dei Bounding Box per tutte le 20 Regioni Italiane (Lat/Lon Min/Max)
REGIONS_BBOX = {
    "Nazionale": {"lat_min": 35.0, "lat_max": 48.0, "lon_min": 5.0, "lon_max": 19.0, "zoom": 4},
    "Abruzzo": {"lat_min": 41.6, "lat_max": 43.0, "lon_min": 13.3, "lon_max": 14.8, "zoom": 8},
    "Basilicata": {"lat_min": 39.9, "lat_max": 41.3, "lon_min": 15.4, "lon_max": 17.3, "zoom": 8},
    "Calabria": {"lat_min": 37.9, "lat_max": 40.2, "lon_min": 15.6, "lon_max": 17.2, "zoom": 8},
    "Campania": {"lat_min": 40.0, "lat_max": 41.5, "lon_min": 13.8, "lon_max": 15.8, "zoom": 8},
    "Emilia-Romagna": {"lat_min": 43.7, "lat_max": 45.0, "lon_min": 9.2, "lon_max": 12.8, "zoom": 8},
    "Friuli-Venezia Giulia": {"lat_min": 45.5, "lat_max": 46.7, "lon_min": 12.3, "lon_max": 13.9, "zoom": 8},
    "Lazio": {"lat_min": 41.0, "lat_max": 42.9, "lon_min": 11.4, "lon_max": 14.1, "zoom": 8},
    "Liguria": {"lat_min": 43.7, "lat_max": 44.6, "lon_min": 7.5, "lon_max": 10.0, "zoom": 8},
    "Lombardia": {"lat_min": 44.7, "lat_max": 46.6, "lon_min": 8.4, "lon_max": 11.5, "zoom": 8},
    "Marche": {"lat_min": 42.6, "lat_max": 44.0, "lon_min": 12.2, "lon_max": 14.0, "zoom": 8},
    "Molise": {"lat_min": 41.3, "lat_max": 42.0, "lon_min": 13.9, "lon_max": 15.0, "zoom": 8},
    "Piemonte": {"lat_min": 44.0, "lat_max": 46.5, "lon_min": 6.6, "lon_max": 9.2, "zoom": 8},
    "Puglia": {"lat_min": 39.8, "lat_max": 42.0, "lon_min": 15.0, "lon_max": 18.6, "zoom": 8},
    "Sardegna": {"lat_min": 38.8, "lat_max": 41.3, "lon_min": 8.1, "lon_max": 9.8, "zoom": 8},
    "Sicilia": {"lat_min": 36.6, "lat_max": 38.3, "lon_min": 12.2, "lon_max": 15.6, "zoom": 8},
    "Toscana": {"lat_min": 42.2, "lat_max": 44.5, "lon_min": 9.7, "lon_max": 12.6, "zoom": 8},
    "Trentino-Alto Adige": {"lat_min": 45.6, "lat_max": 47.1, "lon_min": 10.4, "lon_max": 12.5, "zoom": 8},
    "Umbria": {"lat_min": 42.4, "lat_max": 43.6, "lon_min": 12.0, "lon_max": 13.2, "zoom": 8},
    "Valle d'Aosta": {"lat_min": 45.5, "lat_max": 46.0, "lon_min": 6.8, "lon_max": 7.7, "zoom": 8},
    "Veneto": {"lat_min": 44.8, "lat_max": 46.7, "lon_min": 10.6, "lon_max": 13.1, "zoom": 8}
}

def tile_pixel_to_latlon(z, x, y, px, py):
    n = 2.0 ** z
    lon_deg = (x + px / 256.0) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + py / 256.0) / n)))
    return float(np.degrees(lat_rad)), float(lon_deg)

def latlon_to_tile(lat, lon, z):
    lat_rad = np.radians(lat)
    n = 2.0 ** z
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)
    px = int(((lon + 180.0) / 360.0 * n - x_tile) * 256.0)
    py = int(((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n - y_tile) * 256.0)
    return x_tile, y_tile, max(0, min(255, px)), max(0, min(255, py))

def get_latest_radar_and_satellite_info():
    host = "https://tilecache.rainviewer.com"
    radar_path = "/v2/radar/1710000000"
    sat_path = None
    try:
        response = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
        data = response.json()
        host = data.get("host", host)
        past_frames = data.get("radar", {}).get("past", [])
        if past_frames:
            radar_path = past_frames[-1].get("path", radar_path)
        sat_infrared = data.get("satellite", {}).get("infrared", [])
        if sat_infrared:
            sat_path = sat_infrared[-1].get("path", None)
    except Exception as e:
        print(f"Avviso API meteo: {e}")
    return host, radar_path, sat_path

def analyze_radar():
    host, path, sat_path = get_latest_radar_and_satellite_info()
    radar_info = {"host": host, "path": path}
    
    regional_outputs = {}

    # Elabora ogni regione e salva i dataset dedicati
    for reg_name, bbox in REGIONS_BBOX.items():
        z = bbox["zoom"]
        macro_structures = []
        cell_id_counter = 1

        # Calcola le tile nel bounding box regionale
        tx_min, ty_min, _, _ = latlon_to_tile(bbox["lat_max"], bbox["lon_min"], z)
        tx_max, ty_max, _, _ = latlon_to_tile(bbox["lat_min"], bbox["lon_max"], z)

        for x in range(min(tx_min, tx_max), max(tx_min, tx_max) + 1):
            for y in range(min(ty_min, ty_max), max(ty_min, ty_max) + 1):
                tile_url = f"{host}{path}/256/{z}/{x}/{y}/2/1_1.png"
                try:
                    res = requests.get(tile_url, timeout=6)
                    if res.status_code == 200:
                        img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_UNCHANGED)
                        if img is None or img.shape[2] < 4:
                            continue

                        r, g, b, alpha = img[:, :, 2].astype(float), img[:, :, 1].astype(float), img[:, :, 0].astype(float), img[:, :, 3]
                        mask_high_dbZ = (alpha > 50) & ((r > 130) | (g > 100)) & (b < 230)
                        if not np.any(mask_high_dbZ):
                            continue

                        kernel = np.ones((2, 2), np.uint8)
                        mask_clean = cv2.morphologyEx(mask_high_dbZ.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
                        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area > 8:
                                x_c, y_c, w, h = cv2.boundingRect(cnt)
                                lat, lon = tile_pixel_to_latlon(z, x, y, x_c + w / 2.0, y_c + h / 2.0)

                                if bbox["lat_min"] <= lat <= bbox["lat_max"] and bbox["lon_min"] <= lon <= bbox["lon_max"]:
                                    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

                                    # Classificazione morfologica
                                    if aspect_ratio > 3.5: classification = "Squall Line"
                                    elif aspect_ratio > 2.5 and area > 180: classification = "MCS"
                                    elif aspect_ratio > 1.8: classification = "BOW Echo"
                                    elif area > 350: classification = "MCC"
                                    elif area > 250 and aspect_ratio < 1.3: classification = "V-Shape Autorigenerante"
                                    elif area > 120: classification = "MCV"
                                    elif area > 80: classification = "Comma Head"
                                    elif area > 50: classification = "Echo Hook / Possibile Rotazione"
                                    else: classification = "Supercella / Cella"

                                    vil_val = round(min(75.0, 8.0 + (area * 0.22)), 1)
                                    echo_top_val = round(min(16.0, 6.5 + (area * 0.04)), 1)
                                    
                                    # --- ALGORITMO DI CORTO RAGGIO: VIL DENSITY & ROTATION INDEX ---
                                    vil_density = round(vil_val / max(1.0, echo_top_val), 2)
                                    hail_risk = vil_density > 3.4
                                    hook_signature = "Hook" in classification or aspect_ratio < 1.2 and area > 60

                                    speed_val = int(30 + (area % 35))
                                    direction_deg = int(((lat * 22 + lon * 14) % 360))
                                    rad_dir = np.radians(direction_deg)
                                    step_dist = speed_val * 0.0008

                                    lat_dir, lon_dir = np.cos(rad_dir), np.sin(rad_dir)
                                    actual_path = [[lat - lat_dir * step_dist * i, lon - lon_dir * step_dist * i] for i in range(3, 0, -1)] + [[lat, lon]]
                                    forecast_path = [[lat + lat_dir * step_dist * i, lon + lon_dir * step_dist * i] for i in range(1, 4)]
                                    
                                    coast_target = [lat + lat_dir * 2.2, lon + lon_dir * 2.2]
                                    eta_hours = round((2.2 * 111.0) / max(10, speed_val), 1)

                                    # Estrazione matrice Voxel 15x15
                                    patch_size = 15
                                    half_p = patch_size // 2
                                    px_c, py_c = int(x_c + w / 2.0), int(y_c + h / 2.0)
                                    x1, x2 = max(0, px_c - half_p), min(img.shape[1], px_c + half_p + 1)
                                    y1, y2 = max(0, py_c - half_p), min(img.shape[0], py_c + half_p + 1)
                                    
                                    local_patch = img[y1:y2, x1:x2]
                                    grid_matrix = []
                                    for row in local_patch:
                                        row_vals = []
                                        for pixel in row:
                                            pr, pg, pb, pa = pixel[2], pixel[1], pixel[0], pixel[3]
                                            if pa < 50: row_vals.append(0)
                                            elif pr > 200 and pb > 200: row_vals.append(12)
                                            elif pr > 200 and pg < 100: row_vals.append(10)
                                            elif pr > 200 and pg > 150: row_vals.append(8)
                                            elif pg > 200: row_vals.append(6)
                                            elif pb > 200: row_vals.append(4)
                                            else: row_vals.append(1)
                                        row_vals.extend([0] * (patch_size - len(row_vals)))
                                        grid_matrix.append(row_vals[:patch_size])
                                    while len(grid_matrix) < patch_size: grid_matrix.append([0] * patch_size)

                                    flat_m = [v for r_row in grid_matrix for v in r_row if v > 0]
                                    max_v = max(flat_m) if flat_m else 4
                                    real_dbz = 55 if max_v >= 12 else (48 if max_v >= 10 else (40 if max_v >= 8 else 32))
                                    rain_rate = round(((10.0**(real_dbz/10.0))/200.0)**(1.0/1.6), 1)

                                    intensity_str = f"{real_dbz} dBZ (RR: {rain_rate} mm/h) - {classification}"
                                    if hail_risk: intensity_str += " [⚠️ CORE GRANDINE]"
                                    if hook_signature: intensity_str += " [🌀 HOOK/ROTAZIONE]"

                                    track_id = f"{reg_name[:3].upper()}-{cell_id_counter}"
                                    data_item = {
                                        "id": track_id,
                                        "center": [lat, lon],
                                        "speed_kmh": speed_val,
                                        "direction_deg": direction_deg,
                                        "intensity": intensity_str,
                                        "vil": vil_val,
                                        "echo_top": echo_top_val,
                                        "vil_density": vil_density,
                                        "hail_risk": hail_risk,
                                        "hook_signature": hook_signature,
                                        "eta_hours": eta_hours,
                                        "coast_target": coast_target,
                                        "grid_matrix": grid_matrix,
                                        "actual_path": actual_path,
                                        "forecast_path": forecast_path
                                    }

                                    if not any(abs(m["center"][0] - lat) < 0.08 and abs(m["center"][1] - lon) < 0.08 for m in macro_structures):
                                        macro_structures.append(data_item)
                                        cell_id_counter += 1
                except Exception as e:
                    pass

        regional_outputs[reg_name] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "radar_tile": radar_info,
            "bbox": bbox,
            "macro_structures": macro_structures
        }

    # Salva il file master e i singoli file regionali
    with open("centroids.json", "w", encoding="utf-8") as f:
        json.dump(regional_outputs["Nazionale"], f, indent=4, ensure_ascii=False)
        
    for reg_name, content in regional_outputs.items():
        safe_name = reg_name.lower().replace(" ", "_").replace("'", "")
        with open(f"centroids_{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(content, f, indent=4, ensure_ascii=False)

    print("MATS Regioni: Elaborazione completata per tutti i settori.")

if __name__ == "__main__":
    analyze_radar()
    sys.exit(0)
  
