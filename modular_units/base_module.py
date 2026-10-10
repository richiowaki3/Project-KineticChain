# -*- coding: utf-8 -*-
"""
base_module.py: モジュラーシンセサイザー共通基底ユニットクラス.

すべてのユニットは本インターフェースを実装し、
独立してパラメータを受け取り、単体で信号（Audio / CV）を処理・出力します。
"""

from typing import Dict, Any, Optional
import numpy as np


class BaseModule:
    """
    全モジュールユニット共通の基底クラス.
    """

    def __init__(self, name: str, category: str = "generic", sample_rate: int = 44100):
        self.name = name
        self.category = category
        self.sr = sample_rate
        self.inputs: Dict[str, str] = {}    # 入力端子名: 型/説明
        self.outputs: Dict[str, str] = {}   # 出力端子名: 型/説明
        self.params: Dict[str, Any] = {}    # 制御パラメータ (ノブ / CVターゲット)

    def set_param(self, key: str, value: Any) -> None:
        """パラメータ値を更新"""
        self.params[key] = value

    def get_param(self, key: str, default: Any = None) -> Any:
        """パラメータ値を取得"""
        return self.params.get(key, default)

    def process(
        self,
        audio_in: Optional[np.ndarray] = None,
        cv_in: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        単体での信号処理ロジック (サブクラスでオーバーライド).
        
        Args:
            audio_in: 音声信号 (np.ndarray) [オプション]
            cv_in: 制御信号 (Dict[str, Any]) [オプション]
            
        Returns:
            np.ndarray または Dict[str, Any] (処理後のAudioまたはCV)
        """
        raise NotImplementedError(f"Module [{self.name}] must implement process()")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' category='{self.category}'>"
