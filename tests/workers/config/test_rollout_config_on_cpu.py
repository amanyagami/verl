# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for https://github.com/verl-project/verl/issues/4604.

``actor_rollout_ref.rollout.mode`` used to let users pick between "sync" and
"async" rollout. The "sync" path has been removed for a while (RolloutConfig
already rejected it), so only "async" ever worked. The field itself has now
been removed entirely: rollout always runs in async mode.
"""

import dataclasses
import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig


def _config_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "verl", "trainer", "config"))


class TestRolloutConfigModeRemoved:
    def test_dataclass_has_no_mode_field(self):
        """RolloutConfig should no longer declare a `mode` field."""
        field_names = {f.name for f in dataclasses.fields(RolloutConfig)}
        assert "mode" not in field_names

    def test_default_config_instance_has_no_mode_attr(self):
        config = RolloutConfig(name="vllm")
        assert not hasattr(config, "mode")

    def test_rollout_yaml_has_no_mode_key(self):
        """The rollout.yaml default config should no longer define `mode`."""
        with initialize_config_dir(config_dir=_config_dir(), version_base=None):
            cfg = compose(config_name="ppo_trainer", overrides=["actor_rollout_ref.rollout.name=vllm"])

        assert "mode" not in cfg.actor_rollout_ref.rollout

    def test_rollout_config_resolves_cleanly_without_mode(self):
        """Resolving the actor_rollout_ref.rollout config should still work end to end."""
        with initialize_config_dir(config_dir=_config_dir(), version_base=None):
            cfg = compose(config_name="ppo_trainer", overrides=["actor_rollout_ref.rollout.name=vllm"])

        rollout_config = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
        assert isinstance(rollout_config, RolloutConfig)
        assert not hasattr(rollout_config, "mode")
        assert rollout_config.name == "vllm"

    def test_setting_mode_via_cli_override_raises(self):
        """Explicitly passing `actor_rollout_ref.rollout.mode=...` should now fail clearly,
        since the key no longer exists in the (struct) composed config."""
        with initialize_config_dir(config_dir=_config_dir(), version_base=None):
            with pytest.raises(ConfigCompositionException):
                compose(config_name="ppo_trainer", overrides=["actor_rollout_ref.rollout.mode=async"])
