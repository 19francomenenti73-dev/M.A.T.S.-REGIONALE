import os
import sys
import json
import numpy as np
import cv2
import requests
from datetime import datetime

os.makedirs("profiles", exist_ok=True)

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
        print(f"Avviso nel recupero API meteo: {e}")
    return host, radar_path, sat_path

def check_satellite_v_shape(host, sat_path, lat, lon, z=4):
    if not sat_path:
        return None, None
    try:
        lat_rad = np.radians(lat)
        n = 2.0 ** z
        tx = int((lon + 180.0) / 360.0 * n)
        ty = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)
        px = int(((lon + 180.0) / 360.0 * n - tx) * 256.0)
        py = int(((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n - ty) * 256.0)
        sat_url = f"{host}{sat_path}/256/{z}/{tx}/{ty}/0/0_0.png"
        res = requests.get(sat_url, timeout=5)
        if res.status_code == 200:
            s_img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_GRAYSCALE)
            if s_img is not None and s_img.shape[0] >= 256 and s_img.shape[1] >= 256:
                half_w = 7
                y1 = max(0, py - half_w)
                y2 = min(s_img.shape[0], py + half_w + 1)
                x1 = max(0, px - half_w)
                x2 = min(s_img.shape[1], px + half_w + 1)
                patch = s_img[y1:y2, x1:x2]
                if patch.size > 0:
                    max_val = int(np.max(patch))
                    mean_val = float(np.mean(patch))
                    if max_val > 190 and (max_val - mean_val) > 25:
                        estimated_temp_c = int(-40 - ((max_val / 255.0) * 35))
                        return "V-Shape / Cold Top", estimated_temp_c
    except Exception:
        pass
    return None, None

def analyze_radar():
    host, path, sat_path = get_latest_radar_and_satellite_info()
    radar_info = {"host": host, "path": path}
    
    global_structures = []
    z = 4
    tiles_to_check = [(x, y) for x in range(7, 10) for y in range(4, 6)]
    cell_id_counter = 1

    for x, y in tiles_to_check:
        tile_url = f"{host}{path}/256/{z}/{x}/{y}/2/1_1.png"
        try:
            res = requests.get(tile_url, timeout=8)
            if res.status_code == 200:
                img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None or img.shape[2] < 4:
                    continue

                r = img[:, :, 2].astype(float)
                g = img[:, :, 1].astype(float)
                b = img[:, :, 0].astype(float)
                alpha = img[:, :, 3]

                mask_high_dbZ = (alpha > 50) & ((r > 130) | (g > 100)) & (b < 230)
                if not np.any(mask_high_dbZ):
                    continue

                kernel = np.ones((2, 2), np.uint8)
                mask_clean = cv2.morphologyEx(mask_high_dbZ.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 10:
                        x_c, y_c, w, h = cv2.boundingRect(cnt)
                        lat, lon = tile_pixel_to_latlon(z, x, y, x_c + w / 2.0, y_c + h / 2.0)

                        if 35.0 <= lat <= 48.0 and 5.0 <= lon <= 19.0:
                            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                            if aspect_ratio > 3.5:
                                classification = "Squall Line"
                            elif aspect_ratio > 2.5 and area > 180:
                                classification = "MCS (Mesoscale Convective System)"
                            elif aspect_ratio > 1.8:
                                classification = "BOW Echo"
                            elif area > 350:
                                classification = "MCC (Mesoscale Convective Complex)"
                            elif area > 250 and aspect_ratio < 1.3:
                                classification = "V-Shape Autorigenerante"
                            elif area > 120 and abs(w - h) < 5:
                                classification = "MCV (Mesoscale Convective Vortex)"
                            elif area > 80 and aspect_ratio > 1.5:
                                classification = "Comma Head"
                            elif area > 60:
                                classification = "Echo Hook"
                            else:
                                classification = "Supercella / Cella"

                            sat_pattern, cloud_temp = check_satellite_v_shape(host, sat_path, lat, lon, z)
                            if sat_pattern:
                                classification = f"{sat_pattern} (Cloud Top: {cloud_temp}°C)" if cloud_temp is not None else sat_pattern

                            vil_val = round(min(70.0, 10.0 + (area * 0.18)), 1)
                            echo_top_val = round(min(16.0, 7.0 + (area * 0.035)), 1)
                            speed_val = int(35 + (area % 30))
                            direction_deg = int(((lat * 22 + lon * 14) % 360))

                            rad_dir = np.radians(direction_deg)
                            step_dist = speed_val * 0.0008
                            lat_dir = np.cos(rad_dir)
                            lon_dir = np.sin(rad_dir)

                            actual_path = [
                                [lat - lat_dir * step_dist * 3, lon - lon_dir * step_dist * 3],
                                [lat - lat_dir * step_dist * 2, lon - lon_dir * step_dist * 2],
                                [lat - lat_dir * step_dist * 1, lon - lon_dir * step_dist * 1],
                                [lat, lon]
                            ]
                            forecast_path = [
                                [lat + lat_dir * step_dist * 1, lon + lon_dir * step_dist * 1],
                                [lat + lat_dir * step_dist * 2, lon + lon_dir * step_dist * 2],
                                [lat + lat_dir * step_dist * 3, lon + lon_dir * step_dist * 3]
                            ]
                            coast_dist_scale = 2.5
                            coast_target = [lat + lat_dir * coast_dist_scale, lon + lon_dir * coast_dist_scale]
                            eta_hours = round((coast_dist_scale * 111.0) / max(10, speed_val), 1)

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
                            for row in local_patch:
                                row_vals = []
                                for pixel in row:
                                    pr, pg, pb, pa = pixel[2], pixel[1], pixel[0], pixel[3]
                                    if pa < 50:
                                        row_vals.append(0)
                                    else:
                                        if pr > 200 and pb > 200: row_vals.append(12)
                                        elif pr > 200 and pg < 100: row_vals.append(10)
                                        elif pr > 200 and pg > 150: row_vals.append(8)
                                        elif pg > 200: row_vals.append(6)
                                        elif pb > 200 and pg > 150: row_vals.append(4)
                                        elif pb > 150: row_vals.append(2)
                                        else: row_vals.append(1)
                                row_vals.extend([0] * (patch_size - len(row_vals)))
                                grid_matrix.append(row_vals[:patch_size])

                            while len(grid_matrix) < patch_size:
                                grid_matrix.append([0] * patch_size)

                            flat_m = [val for row in grid_matrix for val in row if val > 0]
                            max_voxel = max(flat_m) if flat_m else 4
                            real_dbz = 55 if max_voxel >= 12 else (48 if max_voxel >= 10 else (40 if max_voxel >= 8 else (34 if max_voxel >= 6 else (26 if max_voxel >= 4 else 18))))
                            rain_rate = round(((10.0 ** (real_dbz / 10.0)) / 200.0) ** (1.0 / 1.6), 1)

                            data_item = {
                                "id": f"Core-{z}{x}{y}-{cell_id_counter}",
                                "center": [lat, lon],
                                "speed_kmh": speed_val,
                                "direction_deg": direction_deg,
                                "intensity": f"{real_dbz} dBZ (RR: {rain_rate} mm/h) - {classification}",
                                "vil": vil_val,
                                "echo_top": echo_top_val,
                                "eta_hours": eta_hours,
                                "coast_target": coast_target,
                                "grid_matrix": grid_matrix,
                                "actual_path": actual_path,
                                "forecast_path": forecast_path
                            }

                            if not any(abs(m["center"][0] - lat) < 0.12 and abs(m["center"][1] - lon) < 0.12 for m in global_structures):
                                global_structures.append(data_item)
                                cell_id_counter += 1
        except Exception:
            pass

    for reg_name, bbox in REGIONS_BBOX.items():
        reg_structures = []
        if reg_name == "Nazionale":
            reg_structures = global_structures
        else:
            for st in global_structures:
                lat, lon = st["center"]
                if bbox["lat_min"] <= lat <= bbox["lat_max"] and bbox["lon_min"] <= lon <= bbox["lon_max"]:
                    reg_structures.append(st)
            if not reg_structures and global_structures:
                reg_structures = [global_structures[0]]

        reg_data = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "radar_tile": radar_info,
            "bbox": bbox,
            "macro_structures": reg_structures
        }

        filename = "centroids.json" if reg_name == "Nazionale" else f"centroids_{reg_name.lower().replace(' ', '_').replace('\'', '').replace('-', '_')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(reg_data, f, indent=4, ensure_ascii=False)

    print(f"MATS Engine: Generati file per Nazionale e tutte le Regioni. Totale strutture: {len(global_structures)}")

if __name__ == "__main__":
    analyze_radar()
    sys.exit(0)
                                        
