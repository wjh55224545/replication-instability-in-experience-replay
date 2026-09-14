"""
实验配置（精简版）
====================
本复现包只涉及三个语言模型端点，故只保留这些配置项；
平台相关的其余配置（数据库、对象存储、调度器等）与本论文的实验无关，已略去。

密钥从环境变量或仓库根目录的 `.env` 读取（见 `.env.example`）。

⚠️ 论文中 M1 = Lingshu-32B，M2 = deepseek-chat。
   Lingshu-32B 经由 MoArk 平台（OpenAI 兼容接口）访问。
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- M1: Lingshu-32B（MoArk 平台，OpenAI 兼容）----
    lingshu_api_key: str = ""
    lingshu_base_url: str = "https://api.moark.com/v1"
    lingshu_model: str = "Lingshu-32B"

    # ---- M2: DeepSeek chat ----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ---- 第三个端点：仅用于评估者分析 ----
    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    zhipu_model: str = "glm-4-flash"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
