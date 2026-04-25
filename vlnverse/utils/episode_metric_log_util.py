import json
import os
from typing import Dict, Any

from vlnverse import PROJECT_ROOT_PATH

from .common_log_util import get_task_name

# 全局变量，存储所有episode的metric信息
EPISODE_METRICS: Dict[str, Any] = {}
INITED = False
JSON_FILE_PATH = None


def init(dataset_name: str):
    """
    初始化episode metric日志工具
    
    Args:
        dataset_name: 数据集名称
    """
    global INITED
    global JSON_FILE_PATH
    global EPISODE_METRICS
    
    # 创建日志目录
    log_dir = f'{PROJECT_ROOT_PATH}/logs/{get_task_name()}/episode_metrics/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 设置JSON文件路径
    JSON_FILE_PATH = f'{log_dir}/{dataset_name}_metrics.json'
    
    # 如果文件已存在，加载现有数据
    if os.path.exists(JSON_FILE_PATH):
        try:
            with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                EPISODE_METRICS = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # 如果文件损坏或无法读取，从空字典开始
            EPISODE_METRICS = {}
            print(f"Warning: Failed to load existing metrics file: {e}. Starting with empty metrics.")
    else:
        EPISODE_METRICS = {}
    
    INITED = True


def log_episode_metric(episode_id: str, metric: Dict[str, Any]):
    """
    记录单个episode的metric信息
    
    Args:
        episode_id: episode的唯一标识符
        metric: episode的metric信息字典
    """
    global INITED
    global JSON_FILE_PATH
    global EPISODE_METRICS
    
    if not INITED:
        return
    
    # 将metric添加到字典中
    EPISODE_METRICS[episode_id] = metric
    
    # 立即写入JSON文件（追加模式）
    _save_to_file()


def log_episode_metrics_batch(metrics_dict: Dict[str, Dict[str, Any]]):
    """
    批量记录多个episode的metric信息
    
    Args:
        metrics_dict: 字典，key是episode_id，value是metric信息
    """
    global INITED
    global JSON_FILE_PATH
    global EPISODE_METRICS
    
    if not INITED:
        return
    
    # 批量更新
    EPISODE_METRICS.update(metrics_dict)
    
    # 立即写入JSON文件
    _save_to_file()


def _save_to_file():
    """
    将EPISODE_METRICS保存到JSON文件
    """
    global JSON_FILE_PATH
    global EPISODE_METRICS
    
    if JSON_FILE_PATH is None:
        return
    
    try:
        # 使用临时文件确保原子性写入
        temp_file_path = JSON_FILE_PATH + '.tmp'
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(EPISODE_METRICS, f, indent=2, ensure_ascii=False)
        
        # 原子性替换原文件
        os.replace(temp_file_path, JSON_FILE_PATH)
    except IOError as e:
        print(f"Error: Failed to save metrics to file: {e}")


def get_episode_metric(episode_id: str) -> Dict[str, Any]:
    """
    获取指定episode的metric信息
    
    Args:
        episode_id: episode的唯一标识符
        
    Returns:
        episode的metric信息字典，如果不存在则返回None
    """
    global EPISODE_METRICS
    return EPISODE_METRICS.get(episode_id)


def get_all_metrics() -> Dict[str, Any]:
    """
    获取所有episode的metric信息
    
    Returns:
        所有episode的metric信息字典
    """
    global EPISODE_METRICS
    return EPISODE_METRICS.copy()


def clear_metrics():
    """
    清空所有已记录的metric信息（同时清空文件）
    """
    global EPISODE_METRICS
    global JSON_FILE_PATH
    
    EPISODE_METRICS = {}
    
    if JSON_FILE_PATH and os.path.exists(JSON_FILE_PATH):
        try:
            os.remove(JSON_FILE_PATH)
        except IOError as e:
            print(f"Error: Failed to remove metrics file: {e}")

