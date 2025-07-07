#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用增强型奖励函数训练强化学习代理
"""

from train_rl_agent import main

if __name__ == "__main__":
    # 调用main函数，设置use_enhanced_reward=True以使用增强型奖励函数
    main(use_enhanced_reward=True)