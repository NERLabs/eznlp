#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验自动化流水线测试脚本
快速验证系统各组件是否正常工作
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/shiwenlong/NERlabs/eznlp')

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试1: 检查依赖导入")
    print("=" * 60)
    
    try:
        import psutil
        print("✓ psutil 可用")
    except ImportError:
        print("✗ psutil 未安装 - 请运行: pip install psutil")
        return False
    
    try:
        import smtplib
        from email.mime.text import MIMEText
        print("✓ 邮件模块可用")
    except ImportError:
        print("✗ 邮件模块导入失败")
        return False
    
    print("")
    return True


def test_config_templates():
    """测试配置模板"""
    print("=" * 60)
    print("测试2: 检查配置模板")
    print("=" * 60)
    
    templates_dir = Path("_7PIPELINE/config_templates")
    
    if not templates_dir.exists():
        print(f"✗ 配置模板目录不存在: {templates_dir}")
        return False
    
    templates = [
        "baseline_example.json",
        "softlexicon_example.json",
        "expert_dict_example.json"
    ]
    
    all_valid = True
    for template in templates:
        template_path = templates_dir / template
        if not template_path.exists():
            print(f"✗ 模板文件不存在: {template}")
            all_valid = False
            continue
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✓ {template} 格式正确")
        except json.JSONDecodeError as e:
            print(f"✗ {template} JSON格式错误: {e}")
            all_valid = False
    
    print("")
    return all_valid


def test_scripts():
    """测试脚本文件"""
    print("=" * 60)
    print("测试3: 检查脚本文件")
    print("=" * 60)
    
    scripts = [
        "_7PIPELINE/experiment_pipeline.py",
        "_7PIPELINE/run_pipeline.sh",
        "_7PIPELINE/run_all_pipelines.sh"
    ]
    
    all_exist = True
    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            print(f"✗ 脚本不存在: {script}")
            all_exist = False
        else:
            is_executable = os.access(script_path, os.X_OK) if script.endswith('.sh') else True
            status = "可执行" if is_executable else "不可执行"
            print(f"✓ {script} 存在 ({status})")
    
    print("")
    return all_exist


def test_pipeline_import():
    """测试流水线导入"""
    print("=" * 60)
    print("测试4: 测试流水线模块导入")
    print("=" * 60)
    
    try:
        # 不使用相对导入，直接读取文件
        pipeline_file = Path("_7PIPELINE/experiment_pipeline.py")
        if not pipeline_file.exists():
            print("✗ 流水线脚本不存在")
            return False
        
        # 检查文件是否可读
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键类是否存在
        required_classes = [
            'PipelineStage',
            'ParameterModificationStage',
            'TestRunStage',
            'BackgroundTrainingStage',
            'ResourceMonitoringStage',
            'ResultComparisonStage',
            'ReportGenerationStage',
            'EmailNotificationStage',
            'ExperimentPipeline'
        ]
        
        for cls in required_classes:
            if f"class {cls}" in content:
                print(f"✓ {cls} 类定义存在")
            else:
                print(f"✗ {cls} 类定义缺失")
                return False
        
        print("")
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        print("")
        return False


def test_directories():
    """测试目录结构"""
    print("=" * 60)
    print("测试5: 检查目录结构")
    print("=" * 60)
    
    dirs = [
        "_7PIPELINE",
        "_7PIPELINE/config_templates",
        "_1CONFIG/redjujube",
        "_2DATA",
        "_4MONITORING",
        "_6EVALUATE",
        "_8TOOL/monitoring"
    ]
    
    all_exist = True
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} 不存在")
            all_exist = False
    
    print("")
    return all_exist


def test_taskfile():
    """测试Taskfile集成"""
    print("=" * 60)
    print("测试6: 检查Taskfile集成")
    print("=" * 60)
    
    taskfile = Path("Taskfile.yml")
    if not taskfile.exists():
        print("✗ Taskfile.yml 不存在")
        print("")
        return False
    
    with open(taskfile, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_tasks = [
        'pipeline:baseline',
        'pipeline:softlexicon',
        'pipeline:expert-dict',
        'pipeline:all'
    ]
    
    all_found = True
    for task in required_tasks:
        if task in content:
            print(f"✓ Task '{task}' 已定义")
        else:
            print(f"✗ Task '{task}' 未定义")
            all_found = False
    
    print("")
    return all_found


def test_gpu_availability():
    """测试GPU可用性"""
    print("=" * 60)
    print("测试7: 检查GPU可用性（可选）")
    print("=" * 60)
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # 解析输出获取GPU信息
            lines = result.stdout.split('\n')
            gpu_found = False
            for line in lines:
                if 'NVIDIA' in line or 'GeForce' in line or 'Tesla' in line:
                    print(f"✓ GPU可用: {line.strip()}")
                    gpu_found = True
                    break
            
            if not gpu_found:
                print("? nvidia-smi可用但未检测到GPU信息")
        else:
            print("✗ nvidia-smi执行失败")
            return False
            
    except FileNotFoundError:
        print("? nvidia-smi 未找到（可能未安装NVIDIA驱动）")
        return False
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi 超时")
        return False
    except Exception as e:
        print(f"? GPU检测异常: {e}")
        return False
    
    print("")
    return True


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "实验自动化流水线系统测试" + " " * 22 + "║")
    print("╚" + "=" * 58 + "╝")
    print("")
    
    # 切换到项目目录
    os.chdir('/home/shiwenlong/NERlabs/eznlp')
    
    tests = [
        ("依赖检查", test_imports),
        ("配置模板", test_config_templates),
        ("脚本文件", test_scripts),
        ("流水线模块", test_pipeline_import),
        ("目录结构", test_directories),
        ("Taskfile集成", test_taskfile),
        ("GPU可用性", test_gpu_availability)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ 测试 '{name}' 执行异常: {e}")
            print("")
            results.append((name, False))
    
    # 打印总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status:8s} - {name}")
    
    print("")
    print(f"总计: {passed}/{total} 通过")
    print("")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用。")
        print("")
        print("快速开始:")
        print("  task pipeline:baseline")
        print("  或")
        print("  bash _7PIPELINE/run_pipeline.sh --baseline --name \"测试实验\"")
        print("")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息。")
        print("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
