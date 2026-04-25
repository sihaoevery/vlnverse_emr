# Accessing the lerobot dataset using the LMDB interface
import json
import os

import numpy as np
import pandas as pd


class LerobotAsLmdb:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_all_keys(self):
        keys = []
        for scan in os.listdir(self.dataset_path):
            scan_path = os.path.join(self.dataset_path, scan)
            if not os.path.isdir(scan_path):
                continue
            for trajectory in os.listdir(scan_path):
                trajectory_path = os.path.join(scan_path, trajectory)
                if not os.path.isdir(trajectory_path):
                    continue
                keys.append(f"{scan}_{trajectory}")
        return keys

    def get_data_by_key(self, key):
        # Special handling for vlnverse dataset (user's custom data)
        if 'vlnverse' in self.dataset_path:
            # For vlnverse: keys like 'kujiale_0003_40_0' or 'kujiale_1100_fix_40_0'
            # Need to intelligently detect where scene name ends and trajectory starts
            parts = key.split('_')

            if len(parts) >= 3 and parts[0] == 'kujiale':
                # Strategy: trajectory IDs are usually numeric (like '40_0' or '0_0')
                # Find the point where we have two consecutive numeric parts
                scene_end_idx = 2  # Default: 'kujiale_XXXX' (first 2 parts)

                # Scan from part index 2 onwards to find trajectory start
                for i in range(2, len(parts) - 1):
                    # Check if current part and next part are both numeric
                    # This indicates the start of trajectory ID like '40_0'
                    try:
                        int(parts[i])
                        int(parts[i + 1])
                        # Found two consecutive numbers, trajectory starts here
                        scene_end_idx = i
                        break
                    except ValueError:
                        # Not both numeric, continue scanning
                        # Update scene_end_idx to include this non-numeric part
                        scene_end_idx = i + 1

                scan = '_'.join(parts[:scene_end_idx])
                trajectory = '_'.join(parts[scene_end_idx:])
            else:
                # Fallback for non-kujiale scenes
                scan = '_'.join(parts[:-1])
                trajectory = parts[-1]
        else:
            # Original logic for other datasets (like interiornav)
            # Handle keys like 'kujiale_0065_861' -> scan='kujiale_0065', trajectory='861'
            parts = key.split('_')
            if len(parts) >= 3:
                # For keys like 'kujiale_0065_861'
                scan = '_'.join(parts[:-1])  # 'kujiale_0065'
                trajectory = parts[-1]  # '861'
            else:
                # Fallback for simple keys like 'scan_traj'
                scan = parts[0]
                trajectory = parts[1]
        trajectory_path = os.path.join(self.dataset_path, scan, trajectory)
        parquet_path = os.path.join(trajectory_path, "data/chunk-000/episode_000000.parquet")
        json_path = os.path.join(trajectory_path, "meta/episodes.jsonl")
        rgb_path = os.path.join(trajectory_path, "videos/chunk-000/observation.images.rgb/rgb.npy")
        depth_path = os.path.join(trajectory_path, "videos/chunk-000/observation.images.depth/depth.npy")

        df = pd.read_parquet(parquet_path)

        # Helper function to safely convert DataFrame column to numpy array
        def safe_to_numpy(column):
            """
            Safely convert DataFrame column to numpy array.
            Handles both string-encoded arrays and proper arrays.
            """
            data_list = column.tolist()

            # Check if elements are strings (e.g., "[-7.37,-1.96,0.0]")
            if len(data_list) > 0 and isinstance(data_list[0], str):
                try:
                    # Parse string-encoded arrays
                    parsed_list = [json.loads(item) for item in data_list]
                    return np.array(parsed_list, dtype=np.float64)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Failed to parse string data: {e}")
                    # Fallback: try to convert directly
                    return np.array(data_list)
            else:
                # Already proper format (list of lists or list of numbers)
                return np.array(data_list, dtype=np.float64)

        data = {}
        data['episode_data'] = {}
        data['episode_data']['camera_info'] = {}
        data['episode_data']['camera_info']['pano_camera_0'] = {}
        data['episode_data']['camera_info']['pano_camera_0']['position'] = safe_to_numpy(
            df['observation.camera_position']
        )
        data['episode_data']['camera_info']['pano_camera_0']['orientation'] = safe_to_numpy(
            df['observation.camera_orientation']
        )
        data['episode_data']['camera_info']['pano_camera_0']['yaw'] = np.array(
            df['observation.camera_yaw'].tolist(), dtype=np.float64
        )
        data['episode_data']['robot_info'] = {}
        data['episode_data']['robot_info']['position'] = safe_to_numpy(df['observation.robot_position'])
        data['episode_data']['robot_info']['orientation'] = safe_to_numpy(df['observation.robot_orientation'])
        data['episode_data']['robot_info']['yaw'] = np.array(df['observation.robot_yaw'].tolist(), dtype=np.float64)
        data['episode_data']['progress'] = np.array(df['observation.progress'].tolist())
        data['episode_data']['step'] = np.array(df['observation.step'].tolist())
        data['episode_data']['action'] = df['observation.action'].tolist()

        episodes_in_json = []
        finish_status_in_json = None
        fail_reason_in_json = None
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    json_data = json.loads(line.strip())
                    episodes_in_json.append(json_data)
                    finish_status_in_json = json_data['finish_status']
                    fail_reason_in_json = json_data['fail_reason']
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
        data["finish_status"] = finish_status_in_json
        data["fail_reason"] = fail_reason_in_json
        data["episodes_in_json"] = episodes_in_json
        data['episode_data']['camera_info']['pano_camera_0']['rgb'] = np.load(rgb_path)
        data['episode_data']['camera_info']['pano_camera_0']['depth'] = np.load(depth_path)
        return data


if __name__ == '__main__':
    ds = LerobotAsLmdb('/shared/smartbot/vln-pe/vln_pe_lerobot/mp3d')

    keys = ds.get_all_keys()
    print(f"total keys:{len(keys)}")
    for k in keys:
        o = ds.get_data_by_key(k)
        print(o)
