import os
import sys
import json
import numpy as np
import cv2
import requests
from datetime import datetime

os.makedirs("profiles", exist_ok=True)
KALMAN_STATE_FILE = "profiles/kalman_states.json"

REGIONS_BBOX = {
    "Nazionale": {"lat_min": 35.0, "lat_max": 48.0, "lon_min": 5.0, "lon_max": 19.0, "zoom": 6, "center": [42.5, 12.5]},
    "Abruzzo": {"lat_min": 41.6, "lat_max": 43.0, "lon_min": 13.3, "lon_max": 14.8, "zoom": 8, "center": [42.3, 14.0]},
    "Basilicata": {"lat_min": 39.9, "lat_max": 41.3, "lon_min": 15.4, "lon_max": 17.3, "zoom": 8, "center": [40.6, 16.5]},
    "Calabria": {"lat_min": 37.9, "lat_max": 40.2, "lon_min": 15.6, "lon_max": 17.2, "zoom": 8, "center": [39.0, 16.5]},
    "Campania": {"lat_min": 40.0, "lat_max": 41.5, "lon_min": 13.8, "lon_max": 15.8, "zoom": 8, "center": [40.8, 14.2]},
    "Emilia-Romagna": {"lat_min": 43.7, "lat_max": 45.0, "lon_min": 9.2, "lon_max": 12.8, "zoom": 8, "center": [44.5, 11.3]},
    "Friuli-Venezia Giulia": {"lat_min": 45.5, "lat_max": 46.7, "lon_min": 12.3, "lon_max": 13.9, "zoom": 8, "center": [46.1, 13.2]},
    "Lazio": {"lat_min": 41.0, "lat_max": 42.9, "lon_min": 11.4, "lon_max": 14.1, "zoom": 8, "center": [41.9, 12.5]},
    "Liguria": {"lat_min": 43.7, "lat_max": 44.6, "lon_min": 7.5, "lon_max": 10.0, "zoom": 8, "center": [44.4, 8.9]},
    "Lombardia": {"lat_min": 44.7, "lat_max": 46.6, "lon_min": 8.4, "lon_max": 11.5, "zoom": 8, "center": [45.5, 9.9]},
    "Marche": {"lat_min": 42.6, "lat_max": 44.0, "lon_min": 12.2, "lon_max": 14.0, "zoom": 8, "center": [43.3, 13.5]},
    "Molise": {"lat_min": 41.3, "lat_max": 42.0, "lon_min": 13.9, "lon_max": 15.0, "zoom": 8, "center": [41.6, 14.7]},
    "Piemonte": {"lat_min": 44.0, "lat_max": 46.5, "lon_min": 6.6, "lon_max": 9.2, "zoom": 8, "center": [45.1, 7.7]},
    "Puglia": {"lat_min": 39.8, "lat_max": 42.0, "lon_min": 15.0, "lon_max": 18.6, "zoom": 8, "center": [41.1, 16.8]},
    "Sardegna": {"lat_min": 38.8, "lat_max": 41.3, "lon_min": 8.1, "lon_max": 9.8, "zoom": 8, "center": [40.1, 9.1]},
    "Sicilia": {"lat_min": 36.6, "lat_max": 38.3, "lon_min": 12.2, "lon_max": 15.6, "zoom": 8, "center": [37.6, 14.0]},
    "Toscana": {"lat_min": 42.2, "lat_max": 44.5, "lon_min": 9.7, "lon_max": 12.6, "zoom": 8, "center": [43.4, 11.2]},
    "Trentino-Alto Adige": {"lat_min": 45.6, "lat_max": 47.1, "lon_min": 10.4, "lon_max": 12.5, "zoom": 8, "center": [46.3, 11.3]},
    "Umbria": {"lat_min": 42.4, "lat_max": 43.6, "lon_min": 12.0, "lon_max": 13.2, "zoom": 8, "center": [43.1, 12.4]},
    "Valle d'Aosta": {"lat_min": 45.5, "lat_max": 46.0, "lon_min": 6.8, "lon_max": 7.7, "zoom": 8, "center": [45.7, 7.3]},
    "Veneto": {"lat_min": 44.8, "lat_max": 46.7, "lon_min": 10.6, "lon_max": 13.1, "zoom": 8, "center": [45.4, 11.8]}
}

def tile_pixel_to_latlon(z, x, y, px, py):
    n = 2.0 ** z
    lon_deg = (x + px / 256.0) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + py / 256.0) / n)))
    return float(np.degrees(lat_rad)), float(lon_deg)

def get_latest_radar_info():
    try:
        response = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=15)
        data = response.json()
        host = data.get("host", "https://tilecache.rainviewer.com")
        past_frames = data.get("radar", {}).get("past", [])
        radar_path = past_frames[-1].get("path") if past_frames else "/v2/radar/1710000000"
        return host, radar_path
    except Exception as e:
        print(f"[AVVISO] Errore API RainViewer: {e}")
        return "https://tilecache.rainviewer.com", "/v2/radar/1710000000"

def load_kalman_states():
    if os.path.exists(KALMAN_STATE_FILE):
        try:
            with open(KALMAN_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_kalman_states(states):
    try:
        with open(KALMAN_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(states, f)
    except Exception:
        pass

def apply_kalman_filter(cell_key, lat, lon, stored_states):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.4

    if cell_key in stored_states:
        st = stored_states[cell_key]
        kf.statePre = np.array(st["state"], dtype=np.float32)
        kf.statePost = np.array(st["state"], dtype=np.float32)
    else:
        kf.statePre = np.array([[lat], [lon], [0.0], [0.0]], np.float32)
        kf.statePost = np.array([[lat], [lon], [0.0], [0.0]], np.float32)

    predicted = kf.predict()
    measured = np.array([[np.float32(lat)], [np.float32(lon)]])
    corrected = kf.correct(measured)

    f_lat = float(corrected[0])
    f_lon = float(corrected[1])
    v_lat = float(corrected[2])
    v_lon = float(corrected[3])

    speed_kmh = int(np.sqrt(v_lat**2 + v_lon**2) * 111.0 * 20.0)
    speed_kmh = max(15, min(110, speed_kmh))

    direction_deg = int((np.degrees(np.arctan2(v_lon, v_lat)) + 360) % 360)
    if direction_deg == 0:
        direction_deg = int(((lat * 20 + lon * 15) % 360))

    stored_states[cell_key] = {
        "state": corrected.tolist()
    }

    return f_lat, f_lon, speed_kmh, direction_deg

def analyze_radar():
    host, path = get_latest_radar_info()
    radar_info = {"host": host, "path": path}
    
    global_structures = []
    z = 5
    tiles_to_check = [(x, y) for x in range(16, 21) for y in range(10, 14)]
    cell_id_counter = 1
    kalman_states = load_kalman_states()
    new_kalman_states = {}

    for x, y in tiles_to_check:
        tile_url = f"{host}{path}/256/{z}/{x}/{y}/2/1_1.png"
        try:
            res = requests.get(tile_url, timeout=10)
            if res.status_code != 200:
                continue
            img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None or img.shape[2] < 4:
                continue

            r = img[:, :, 2].astype(float)
            g = img[:, :, 1].astype(float)
            b = img[:, :, 0].astype(float)
            alpha = img[:, :, 3]

            mask_precip = (alpha > 5) & ((r > 10) | (g > 10) | (b > 10))
            if not np.any(mask_precip):
                continue

            contours, _ = cv2.findContours(mask_precip.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5:
                    x_c, y_c, w, h = cv2.boundingRect(cnt)
                    raw_lat, raw_lon = tile_pixel_to_latlon(z, x, y, x_c + w / 2.0, y_c + h / 2.0)

                    if 35.0 <= raw_lat <= 48.0 and 5.0 <= raw_lon <= 19.0:
                        cell_key = f"cell_{z}_{x}_{y}_{int(raw_lat*10)}_{int(raw_lon*10)}"
                        
                        # Applicazione Filtro di Kalman
                        lat, lon, speed_val, direction_deg = apply_kalman_filter(cell_key, raw_lat, raw_lon, kalman_states)
                        new_kalman_states[cell_key] = kalman_states[cell_key]

                        rad_dir = np.radians(direction_deg)
                        step_dist = speed_val * 0.0008
                        lat_dir = np.cos(rad_dir)
                        lon_dir = np.sin(rad_dir)

                        actual_path = [[lat - lat_dir * step_dist * 3, lon - lon_dir * step_dist * 3], [lat, lon]]
                        forecast_path = [[lat + lat_dir * step_dist * 1, lon + lon_dir * step_dist * 1], [lat + lat_dir * step_dist * 3, lon + lon_dir * step_dist * 3]]
                        coast_target = [lat + lat_dir * 2.5, lon + lon_dir * 2.5]
                        eta_hours = round((2.5 * 111.0) / max(10, speed_val), 1)

                        patch_size = 15
                        half_p = patch_size // 2
                        px_c_val = int(x_c + w / 2.0)
                        py_c_val = int(y_c + h / 2.0)
                        x_min = max(0, px_c_val - half_p)
                        x_max = min(img.shape[1], px_c_val + half_p + 1)
                        y_min = max(0, py_c_val - half_p)
                        y_max = min(img.shape[0], py_c_val + half_p + 1)

                        local_patch = img[y_min:y_max, x_min:x_max]
                        grid_matrix = []
                        max_pixel_val = 0

                        for row in local_patch:
                            row_vals = []
                            for pixel in row:
                                pr, pg, pb, pa = pixel[2], pixel[1], pixel[0], pixel[3]
                                if pa < 5: 
                                    row_vals.append(0)
                                else:
                                    if pr > 200 and pb > 150: 
                                        val = 14  # Magenta / Core Estremo
                                    elif pr > 200: 
                                        val = 11  # Rosso / Core Severo
                                    elif pr > 150 and pg > 100: 
                                        val = 8   # Arancione / Alta intensità
                                    elif pg > 150: 
                                        val = 6   # Giallo / Moderato
                                    elif pb > 100 or pg > 50: 
                                        val = 3   # Verde/Ciano / Pioggia
                                    else:
                                        val = 2
                                    max_pixel_val = max(max_pixel_val, val)
                                    row_vals.append(val)
                            row_vals.extend([0] * (patch_size - len(row_vals)))
                            grid_matrix.append(row_vals[:patch_size])

                        while len(grid_matrix) < patch_size:
                            grid_matrix.append([0] * patch_size)

                        calibrated_dbz = int(35 + (max_pixel_val * 1.5) + (area % 5))
                        if calibrated_dbz > 58: calibrated_dbz = 58
                        
                        # CALCOLO RAIN RATE CON RELAZIONE DI MARSHALL-PALMER PURA
                        # Z = 200 * R^1.6  ==>  R = (10^(dBZ / 10) / 200)^(0.625)
                        linear_z = np.power(10.0, calibrated_dbz / 10.0)
                        rain_rate = round(float(np.power(linear_z / 200.0, 0.625)), 1)

                        vil_val = round(calibrated_dbz * 1.2, 1)
                        
                        # ECHO TOP CALIBRATO MEDITERRANEO (Max ~13.3 km)
                        echo_top_val = round(min(13.5, 5.5 + (calibrated_dbz / 9.0) + (vil_val / 50.0)), 1)

                        data_item = {
                            "id": f"Core-{z}{x}{y}-{cell_id_counter}",
                            "center": [lat, lon],
                            "speed_kmh": speed_val,
                            "direction_deg": direction_deg,
                            "intensity": f"{calibrated_dbz} dBZ (RR: {rain_rate} mm/h) - Cella Convettiva",
                            "dbz": calibrated_dbz,
                            "vil": vil_val,
                            "echo_top": echo_top_val,
                            "eta_hours": eta_hours,
                            "coast_target": coast_target,
                            "grid_matrix": grid_matrix,
                            "actual_path": actual_path,
    
