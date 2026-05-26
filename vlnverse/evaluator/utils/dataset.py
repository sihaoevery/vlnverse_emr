import gzip
import json
import os
import sys
from datetime import datetime

import lmdb
import msgpack_numpy

from vlnverse import PROJECT_ROOT_PATH
from vlnverse.configs.evaluator import EvalDatasetCfg
from vlnverse.evaluator.utils.common import load_data
from vlnverse.utils.common_log_util import common_logger as log
from vlnverse.evaluator.utils.config import get_lmdb_path, get_lmdb_prefix

from .config import Config


def split_data(dataset_cfg: EvalDatasetCfg):
    if isinstance(dataset_cfg.dataset_settings, dict):
        config = Config(**dataset_cfg.dataset_settings)
    run_type = config.run_type
    split_number = 1  # config.total_rank
    run_type = run_type
    name = config.task_name
    split_data_types = config.split_data_types
    base_data_dir = config.base_data_dir
    filter_stairs = config.filter_stairs

    print(f'run_type:{run_type}')
    print(f'name:{name}')
    print(f'split_data_types:{split_data_types}')
    prefix = get_lmdb_prefix(run_type)
    if run_type == 'eval':
        filter_same_trajectory = False
    elif run_type == 'sample':
        filter_same_trajectory = True
    else:
        print(f'unknown run_type:{run_type}')
        sys.exit()

    lmdb_path = get_lmdb_path(name)
    # get all data
    path_key_map = {}
    count = 0

    dataset_type = dataset_cfg.dataset_type
    instruction_type = getattr(config, 'instruction_type', 'formal')
    for split_data_type in split_data_types:
        data_map = load_data(
            base_data_dir,
            split_data_type,
            filter_same_trajectory=filter_same_trajectory,
            filter_stairs=filter_stairs,
            dataset_type=dataset_type,
            instruction_type=instruction_type,
        )
        for scan, path_list in data_map.items():
            path_key_list = []
            for path in path_list:
                trajectory_id = path['trajectory_id']
                episode_id = path['episode_id']
                path_key = f'{trajectory_id}_{episode_id}'
                path_key_list.append(path_key)
            path_key_map[scan] = path_key_list
            count += len(path_key_list)

    print(f'TOTAL:{count}')

    # split rank
    rank_map = {}
    split_length = count // split_number
    index = -1
    for scan, path_key_list in path_key_map.items():
        for path_key in path_key_list:
            index += 1
            rank = index // split_length
            if rank >= split_number:
                rank = split_number - 1
            rank_map[path_key] = rank

    ranked_data = {}
    for i in range(split_number):
        filtered_path_key_map = {}
        for scan, path_key_list in path_key_map.items():
            filtered_list = []
            for path_key in path_key_list:
                if rank_map[path_key] == i:
                    filtered_list.append(path_key)
            if len(filtered_list) > 0:
                filtered_path_key_map[scan] = filtered_list
        ranked_data[i] = filtered_path_key_map

    for rank, path_key_map in ranked_data.items():
        count = 0
        for scan, path_key_list in path_key_map.items():
            count += len(path_key_list)
            print(f'[rank:{rank}][scan:{scan}][count:{len(path_key_list)}]')
        print(f'[rank:{rank}][count:{count}]')

    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    database = lmdb.open(
        f'{lmdb_path}/sample_data.lmdb',
        map_size=1 * 1024 * 1024 * 1024 * 1024,
        max_dbs=0,
    )
    with database.begin(write=True) as txn:
        for rank, path_key_map in ranked_data.items():
            key = f'{prefix}_{rank}'.encode()
            value = msgpack_numpy.packb(path_key_map, use_bin_type=True)
            txn.put(key, value)
            print(f'finish [key:{key}]')
    database.close()


class ResultLogger:
    def __init__(self, dataset_cfg: EvalDatasetCfg):
        if isinstance(dataset_cfg.dataset_settings, dict):
            config = Config(**dataset_cfg.dataset_settings)
        self.name = config.task_name
        self.lmdb_path = get_lmdb_path(self.name)
        self.dataset_type = dataset_cfg.dataset_type
        self.split_map, self.episode_data_map = self.get_split_map(
            base_data_dir=config.base_data_dir,
            split_data_types=config.split_data_types,
            filter_stairs=config.filter_stairs,
            dataset_type=self.dataset_type,
            instruction_type=getattr(config, 'instruction_type', 'formal'),
        )
        # Fixed timestamp per ResultLogger instance so per-episode submission
        # writes overwrite the SAME file (crash-safe via atomic rename).
        self.submission_timestamp = datetime.now().strftime('%Y_%m_%d-%H%M')
        # GT presence per split: 'full' / 'none' / 'mixed'. 'mixed' triggers a
        # warning and skips on-the-fly GT aggregation (results would be unreliable).
        self.split_gt_status = {}
        for split, eps in self.episode_data_map.items():
            n_with = sum(1 for p in eps.values() if bool(p.get('reference_path')))
            n_tot = len(eps)
            if n_tot == 0 or n_with == 0:
                self.split_gt_status[split] = 'none'
            elif n_with == n_tot:
                self.split_gt_status[split] = 'full'
            else:
                self.split_gt_status[split] = 'mixed'
                log.warning(
                    f"[ResultLogger] split '{split}' has mixed GT presence "
                    f"({n_with}/{n_tot} episodes have reference_path). This does not "
                    f"match the expected JSON format. On-the-fly GT metrics (SR/SPL/"
                    f"NE/OS) will be skipped because they would be unreliable. "
                    f"Use the submission_*.json.gz + your own GT JSON to score offline."
                )

    def get_split_map(
        self,
        base_data_dir,
        split_data_types,
        filter_stairs,
        dataset_type='mp3d',
        instruction_type='formal',
    ):
        split_map = {}
        episode_data_map = {}
        for split_data_type in split_data_types:
            load_data_map = load_data(
                base_data_dir,
                split_data_type,
                filter_same_trajectory=False,
                filter_stairs=filter_stairs,
                dataset_type=dataset_type,
                instruction_type=instruction_type,
            )
            path_key_list = []
            episode_map = {}
            for scan, path_list in load_data_map.items():
                for path in path_list:
                    trajectory_id = path['trajectory_id']
                    episode_id = path['episode_id']
                    path_key = f'{trajectory_id}_{episode_id}'
                    path_key_list.append(path_key)
                    episode_map[path_key] = path
            split_map[split_data_type] = path_key_list
            episode_data_map[split_data_type] = episode_map
        return split_map, episode_data_map

    def write_now_result_json(self):
        # create log file
        log_content = []
        self.database_read = lmdb.open(
            f'{self.lmdb_path}/sample_data.lmdb',
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            readonly=True,
            lock=False,
        )
        json_data = {}
        for split, path_key_list in self.split_map.items():
            data_list = []
            for path_key in path_key_list:
                data_key = path_key
                with self.database_read.begin() as txn:
                    value = txn.get(data_key.encode())
                    if value is None:
                        continue
                    value = msgpack_numpy.unpackb(value)
                value['path_key'] = path_key
                data_list.append(value)
            count = len(data_list)
            status = self.split_gt_status.get(split, 'full')
            has_gt = (status == 'full')
            total_TL = 0
            total_NE = 0
            total_osr = 0
            total_success = 0
            total_spl = 0
            reason_map = {'reach_goal': 0}

            for data in data_list:
                # TL / fail_reason are GT-independent.
                total_TL += data['info']['TL']
                ret_type = data['fail_reason']
                if ret_type == '':
                    ret_type = 'success'
                reason_map[ret_type] = reason_map.get(ret_type, 0) + 1

                if has_gt:
                    NE = data['info']['NE']
                    if NE < 0:
                        NE = 0
                    osr = data['info']['osr']
                    if osr < 0:
                        osr = 0
                    success = data['info']['success']
                    spl = data['info']['spl']
                    total_NE += NE
                    total_osr += osr
                    total_success += success
                    total_spl += spl
                    if success > 0:
                        reason_map['reach_goal'] = reason_map['reach_goal'] + 1

            if count == 0:
                continue
            json_data[split] = {}
            json_data[split]['Count'] = count
            json_data[split]['TL'] = round((total_TL / count), 4)
            if 'fall' not in reason_map:
                reason_map['fall'] = 0
            json_data[split]['FR'] = round((reason_map['fall'] / count), 4)
            json_data[split]['StR'] = round((reason_map.get('stuck', 0) / count), 4)
            if has_gt:
                json_data[split]['NE'] = round((total_NE / count), 4)
                json_data[split]['OS'] = round((total_osr / count), 4)
                json_data[split]['SR'] = round((total_success / count), 4)
                json_data[split]['SPL'] = round((total_spl / count), 4)
            elif status == 'none':
                json_data[split]['note'] = 'no GT for this split - see submission_*.json.gz'
            else:  # mixed
                json_data[split]['note'] = (
                    'split JSON has mixed GT presence (some episodes missing '
                    'reference_path); on-the-fly GT metrics skipped - score '
                    'offline using submission_*.json.gz against your GT JSON'
                )

        # write log content to file
        log_dir = f'{PROJECT_ROOT_PATH}/logs/{self.name}'
        os.makedirs(log_dir, exist_ok=True)
        with open(f'{log_dir}/{self.dataset_type}_result.json', 'w') as f:
            json.dump(json_data, f)
        self.database_read.close()

    def write_submission_json(self):
        """Per-split GT-format submission JSON: predicted trajectory + stop pos.

        Called after every episode termination so partial results survive crashes /
        SIGINT / power loss. Each call overwrites the SAME file (stable timestamp
        from __init__) via atomic rename — readers never see a half-written file.
        GT-dependent fields (info.geodesic_distance) get -1; scorer supplies the
        real GT offline.
        """
        log_dir = f'{PROJECT_ROOT_PATH}/logs/{self.name}'
        os.makedirs(log_dir, exist_ok=True)
        ts = self.submission_timestamp

        self.database_read = lmdb.open(
            f'{self.lmdb_path}/sample_data.lmdb',
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            readonly=True,
            lock=False,
        )
        for split, path_key_list in self.split_map.items():
            episodes = []
            ep_meta = self.episode_data_map.get(split, {})
            for path_key in path_key_list:
                with self.database_read.begin() as txn:
                    value = txn.get(path_key.encode())
                if value is None:
                    continue
                value = msgpack_numpy.unpackb(value)
                pred_path = value['info'].get('pred_path', [])
                pred_path = [list(map(float, p)) for p in pred_path]
                radius = float(value['info'].get('success_distance', 3.0))
                ep = ep_meta.get(path_key)
                if ep is None:
                    continue
                # Fallback for empty trajectory (immediate fall/stuck): start pose.
                if len(pred_path) > 0:
                    stop_pos = pred_path[-1]
                else:
                    sp = ep['start_position']
                    stop_pos = [float(c) for c in sp]
                episodes.append({
                    'episode_id': ep['episode_id'],
                    'trajectory_id': ep['trajectory_id'],
                    'scan': ep.get('scan'),
                    'scene_id': ep.get('scene_id'),
                    'start_position': ep['start_position'],
                    'start_rotation': ep['start_rotation'],
                    'reference_path': pred_path,
                    'goals': {'position': stop_pos, 'radius': radius},
                    'info': {'geodesic_distance': -1},
                })
            out_path = f'{log_dir}/submission_{self.dataset_type}_{split}_{ts}.json.gz'
            tmp_path = f'{out_path}.tmp'
            with gzip.open(tmp_path, 'wt', encoding='utf-8') as f:
                json.dump({'episodes': episodes}, f)
            os.replace(tmp_path, out_path)  # atomic — no half-written files on crash
        self.database_read.close()

    def write_now_result(self):

        # create log file
        log_content = []

        def log_print(content):
            log_content.append(str(content))

        self.database_read = lmdb.open(
            f'{self.lmdb_path}/sample_data.lmdb',
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            readonly=True,
            lock=False,
        )

        for split, path_key_list in self.split_map.items():
            data_list = []
            for path_key in path_key_list:
                data_key = path_key
                with self.database_read.begin() as txn:
                    value = txn.get(data_key.encode())
                    if value is None:
                        continue
                    value = msgpack_numpy.unpackb(value)
                value['path_key'] = path_key
                data_list.append(value)
            count = len(data_list)
            status = self.split_gt_status.get(split, 'full')
            has_gt = (status == 'full')
            log_print(f'[split:{split}] total {count} data (gt_status={status})')
            total_TL = 0
            total_NE = 0
            total_osr = 0
            total_success = 0
            total_spl = 0
            reason_map = {'reach_goal': 0}

            for data in data_list:
                total_TL += data['info']['TL']
                ret_type = data['fail_reason']
                if ret_type == '':
                    ret_type = 'success'
                reason_map[ret_type] = reason_map.get(ret_type, 0) + 1

                if has_gt:
                    NE = data['info']['NE']
                    if NE < 0:
                        NE = 0
                    osr = data['info']['osr']
                    if osr < 0:
                        osr = 0
                    success = data['info']['success']
                    spl = data['info']['spl']
                    total_NE += NE
                    total_osr += osr
                    total_success += success
                    total_spl += spl
                    if success > 0:
                        reason_map['reach_goal'] = reason_map['reach_goal'] + 1

            log_print(f'############[{split}]#############')
            if count == 0:
                log_print('############[count == 0,skip]#############')
                continue
            log_print(f'TL = {total_TL} / {count} = {round((total_TL / count),4)}')
            if 'fall' not in reason_map:
                reason_map['fall'] = 0
            log_print(f"FR = {reason_map['fall']} / {count} = {round((reason_map['fall'] / count),4) * 100}%")
            log_print(f"StR = {reason_map.get('stuck', 0)} / {count} = {round((reason_map.get('stuck', 0) / count),4) * 100}%")
            if has_gt:
                log_print(f'NE = {total_NE} / {count} = {round((total_NE / count),4)}')
                log_print(f'OS = {total_osr} / {count} = {round((total_osr / count),4) * 100}%')
                log_print(f'SR = {total_success} / {count} = {round((total_success / count),4) * 100}%')
                log_print(f'SPL = {total_spl} / {count} = {round((total_spl / count),4) * 100}%')
            elif status == 'none':
                log_print('NE/OS/SR/SPL: skipped (no GT) - see submission_*.json.gz')
            else:  # mixed
                log_print('NE/OS/SR/SPL: skipped (mixed GT presence - '
                          'split JSON malformed); score offline via submission_*.json.gz')
            log_print('detail:')
            for k, v in reason_map.items():
                log_print(f'[{k}]:{v}')
            log_print('##########################')

        # write log content to file
        log_file_path = os.path.join(self.lmdb_path, 'eval.log')
        log_file_path1 = f'{PROJECT_ROOT_PATH}/logs/{self.name}/eval_result.log'
        with open(log_file_path, 'w') as f:
            f.write('\n'.join(log_content))
        with open(log_file_path1, 'w') as f:
            f.write('\n'.join(log_content))

        self.database_read.close()
