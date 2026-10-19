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
                        x_c, y_c, w
