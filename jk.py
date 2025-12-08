#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU进程监测程序
实时监测GPU使用情况，统计每个进程的平均显存和GPU占用率
"""

import pynvml
import time
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple


class ProcessInfo:
    """进程信息类"""
    def __init__(self, pid: int, name: str, process_id: int):
        self.pid = pid
        self.name = name
        self.process_id = process_id  # 用于区分重名进程
        self.memory_samples = []  # 显存使用记录 (MB)
        self.gpu_util_samples = []  # GPU使用率记录 (%)
        self.start_time = datetime.now()
        self.end_time = None
        self.is_active = True
        
    def add_sample(self, memory_mb: float, gpu_util: float):
        """添加一次采样数据"""
        self.memory_samples.append(memory_mb)
        self.gpu_util_samples.append(gpu_util)
    
    def get_avg_memory(self) -> float:
        """获取平均显存使用量 (MB)"""
        return sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
    
    def get_avg_gpu_util(self) -> float:
        """获取平均GPU使用率 (%)"""
        return sum(self.gpu_util_samples) / len(self.gpu_util_samples) if self.gpu_util_samples else 0
    
    def mark_ended(self):
        """标记进程结束"""
        self.is_active = False
        self.end_time = datetime.now()
    
    def get_duration(self) -> float:
        """获取运行时长(秒)"""
        end = self.end_time if self.end_time else datetime.now()
        return (end - self.start_time).total_seconds()
    
    def get_display_name(self) -> str:
        """获取显示名称(包含编号)"""
        return f"{self.name}[{self.process_id}]"


class GPUMonitor:
    """GPU监测器"""
    def __init__(self, sample_interval: float = 1.0):
        """
        初始化GPU监测器
        
        Args:
            sample_interval: 采样间隔(秒)
        """
        self.sample_interval = sample_interval
        self.processes: Dict[Tuple[int, int], ProcessInfo] = {}  # key: (gpu_id, pid)
        self.process_counter = defaultdict(int)  # 记录每个进程名的计数
        self.completed_processes: List[ProcessInfo] = []  # 已结束的进程
        
        # 初始化NVML
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            print(f"✓ 检测到 {self.device_count} 个GPU设备")
        except Exception as e:
            print(f"✗ 初始化NVML失败: {e}")
            print("请确保已安装NVIDIA驱动和nvidia-ml-py库 (pip install nvidia-ml-py)")
            sys.exit(1)
    
    def __del__(self):
        """清理资源"""
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    
    def get_process_id(self, process_name: str) -> int:
        """为进程分配唯一ID"""
        self.process_counter[process_name] += 1
        return self.process_counter[process_name]
    
    def collect_gpu_info(self) -> Dict:
        """收集GPU信息"""
        gpu_info = {}
        
        for gpu_id in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # 获取GPU基本信息
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 获取所有进程信息
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except:
                processes = []
            
            gpu_info[gpu_id] = {
                'name': name,
                'memory_total': memory_info.total / (1024**2),  # MB
                'memory_used': memory_info.used / (1024**2),  # MB
                'memory_free': memory_info.free / (1024**2),  # MB
                'gpu_util': utilization.gpu,  # %
                'processes': processes
            }
        
        return gpu_info
    
    def update_processes(self, gpu_info: Dict):
        """更新进程信息"""
        current_pids = set()
        
        for gpu_id, info in gpu_info.items():
            for proc in info['processes']:
                pid = proc.pid
                memory_mb = proc.usedGpuMemory / (1024**2) if hasattr(proc, 'usedGpuMemory') else 0
                
                # 获取进程名称
                try:
                    process_name = pynvml.nvmlSystemGetProcessName(pid).decode('utf-8')
                except:
                    process_name = f"Unknown[{pid}]"
                
                key = (gpu_id, pid)
                current_pids.add(key)
                
                # 新进程
                if key not in self.processes:
                    process_id = self.get_process_id(process_name)
                    self.processes[key] = ProcessInfo(pid, process_name, process_id)
                    print(f"\n[新进程] GPU{gpu_id} - {self.processes[key].get_display_name()} (PID: {pid})")
                
                # 添加采样数据
                # 注意: nvmlDeviceGetUtilizationRates返回整个GPU的利用率,不是单个进程的
                # 这里我们用GPU总利用率作为近似值
                self.processes[key].add_sample(memory_mb, info['gpu_util'])
        
        # 检查已结束的进程
        ended_processes = []
        for key, proc_info in self.processes.items():
            if proc_info.is_active and key not in current_pids:
                proc_info.mark_ended()
                ended_processes.append(proc_info)
                self.completed_processes.append(proc_info)
        
        # 显示结束的进程统计
        for proc_info in ended_processes:
            self.print_process_summary(proc_info)
    
    def print_process_summary(self, proc_info: ProcessInfo):
        """打印进程统计摘要"""
        print(f"\n{'='*70}")
        print(f"[进程结束] {proc_info.get_display_name()} (PID: {proc_info.pid})")
        print(f"{'='*70}")
        print(f"运行时长: {proc_info.get_duration():.1f} 秒")
        print(f"平均显存: {proc_info.get_avg_memory():.2f} MB")
        print(f"平均GPU使用率: {proc_info.get_avg_gpu_util():.1f} %")
        print(f"采样次数: {len(proc_info.memory_samples)}")
        print(f"{'='*70}\n")
    
    def print_current_status(self, gpu_info: Dict):
        """打印当前状态"""
        # 清屏 (可选)
        # print("\033[2J\033[H")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*70}")
        print(f"GPU监测状态 - {timestamp}")
        print(f"{'='*70}")
        
        for gpu_id, info in gpu_info.items():
            print(f"\n【GPU {gpu_id}】 {info['name']}")
            print(f"  显存: {info['memory_used']:.0f} / {info['memory_total']:.0f} MB "
                  f"({info['memory_used']/info['memory_total']*100:.1f}%)")
            print(f"  GPU使用率: {info['gpu_util']}%")
            
            if info['processes']:
                print(f"  运行进程: {len(info['processes'])}个")
                for proc in info['processes']:
                    key = (gpu_id, proc.pid)
                    if key in self.processes:
                        proc_info = self.processes[key]
                        memory_mb = proc.usedGpuMemory / (1024**2) if hasattr(proc, 'usedGpuMemory') else 0
                        print(f"    - {proc_info.get_display_name()} (PID: {proc.pid}): "
                              f"{memory_mb:.0f} MB, "
                              f"运行时长: {proc_info.get_duration():.0f}s")
            else:
                print(f"  运行进程: 无")
        
        # 显示活跃进程统计
        active_count = sum(1 for p in self.processes.values() if p.is_active)
        print(f"\n活跃进程: {active_count} | 已完成: {len(self.completed_processes)}")
        print(f"{'='*70}")
    
    def run(self):
        """运行监测循环"""
        print("\n开始监测GPU...")
        print(f"采样间隔: {self.sample_interval}秒")
        print("按 Ctrl+C 停止监测\n")
        
        try:
            while True:
                gpu_info = self.collect_gpu_info()
                self.update_processes(gpu_info)
                self.print_current_status(gpu_info)
                time.sleep(self.sample_interval)
        
        except KeyboardInterrupt:
            print("\n\n收到停止信号,正在生成最终报告...")
            self.generate_final_report()
    
    def generate_final_report(self):
        """生成最终报告"""
        print(f"\n{'='*70}")
        print("最终统计报告")
        print(f"{'='*70}\n")
        
        # 标记所有活跃进程为结束
        for proc_info in self.processes.values():
            if proc_info.is_active:
                proc_info.mark_ended()
                self.completed_processes.append(proc_info)
        
        if not self.completed_processes:
            print("没有监测到任何进程")
            return
        
        print(f"共监测到 {len(self.completed_processes)} 个进程:\n")
        
        # 按进程名排序
        sorted_procs = sorted(self.completed_processes, 
                            key=lambda p: (p.name, p.process_id))
        
        for proc_info in sorted_procs:
            print(f"进程: {proc_info.get_display_name()} (PID: {proc_info.pid})")
            print(f"  运行时长: {proc_info.get_duration():.1f} 秒")
            print(f"  平均显存: {proc_info.get_avg_memory():.2f} MB")
            print(f"  平均GPU使用率: {proc_info.get_avg_gpu_util():.1f} %")
            print(f"  采样次数: {len(proc_info.memory_samples)}")
            print()
        
        print(f"{'='*70}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU进程监测程序')
    parser.add_argument('-i', '--interval', type=float, default=0.1,
                      help='采样间隔(秒), 默认1.0秒')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(sample_interval=args.interval)
    monitor.run()


if __name__ == "__main__":
    main()