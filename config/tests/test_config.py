"""Tests for V3.6 Configuration Module.

Hardware: Single machine (192.168.71.166) with dual GPU:
    - GPU 0: RTX 3060 Ti 8G — display + NVENC/NVDEC + ffmpeg IO (host)
    - GPU 1: RTX 3090 24G — CUDA inference primary

Note: 3060Ti Combo overflow is DEPRECATED. All inference runs on 3090.
"""

import pytest

from config.stage_config import STAGE_CONFIG, STAGE_CONFIG_INV
from config.combo_config import COMBO_3060TI
from config.models_registry import MODELS, LIGHT_MODELS, HEAVY_MODELS
from config.routing_table import ROUTING_TABLE, build_routing_table


class TestStageConfig:
    """Test STAGE_CONFIG completeness and validity."""

    def test_seven_stages_defined(self):
        """V3.6 defines exactly 7 stages."""
        assert len(STAGE_CONFIG) == 7

    def test_required_stage_keys(self):
        """Each stage must have heavy_model, heavy_vram, light_pool_max, resident_light, desc."""
        required = {"heavy_model", "heavy_vram", "light_pool_max", "resident_light", "desc"}
        for stage_name, cfg in STAGE_CONFIG.items():
            missing = required - set(cfg.keys())
            assert not missing, f"Stage {stage_name} missing: {missing}"

    def test_vram_within_hard_cap(self):
        """heavy_vram + light_pool_max must be ≤ 21000 MB."""
        for stage_name, cfg in STAGE_CONFIG.items():
            total = cfg["heavy_vram"] + cfg["light_pool_max"]
            assert total <= 21000, (
                f"Stage {stage_name}: {total}MB > 21000MB hard cap"
            )

    def test_heavy_models_in_registry(self):
        """All heavy_model references must exist in MODELS."""
        for stage_name, cfg in STAGE_CONFIG.items():
            model_id = cfg["heavy_model"]
            assert model_id in MODELS, f"Stage {stage_name}: unknown model {model_id}"

    def test_resident_light_are_light_models(self):
        """All resident_light models must be Light category."""
        for stage_name, cfg in STAGE_CONFIG.items():
            for model_id in cfg["resident_light"]:
                assert model_id in LIGHT_MODELS, (
                    f"Stage {stage_name}: {model_id} not a Light model"
                )

    def test_stage_config_inv_complete(self):
        """STAGE_CONFIG_INV maps every heavy_model to a stage."""
        assert len(STAGE_CONFIG_INV) == len(STAGE_CONFIG)
        for stage_name, cfg in STAGE_CONFIG.items():
            assert cfg["heavy_model"] in STAGE_CONFIG_INV

    @pytest.mark.parametrize("stage_name", list(STAGE_CONFIG.keys()))
    def test_heavy_vram_positive(self, stage_name):
        """Heavy VRAM must be positive."""
        assert STAGE_CONFIG[stage_name]["heavy_vram"] > 0

    @pytest.mark.parametrize("stage_name", list(STAGE_CONFIG.keys()))
    def test_light_pool_max_non_negative(self, stage_name):
        """Light pool max must be non-negative."""
        assert STAGE_CONFIG[stage_name]["light_pool_max"] >= 0


class TestComboConfig:
    """Test COMBO_3060TI completeness and validity.
    
    ⚠️ DEPRECATED — 3060Ti no longer handles CUDA inference.
    These tests are kept for backwards compatibility.
    """

    def test_seven_combos_defined(self):
        """V3.6 defines exactly 7 Combos."""
        assert len(COMBO_3060TI) == 7

    def test_required_combo_keys(self):
        """Each Combo must have models, strategy, desc."""
        required = {"models", "strategy", "desc"}
        for combo_id, cfg in COMBO_3060TI.items():
            missing = required - set(cfg.keys())
            assert not missing, f"Combo {combo_id} missing: {missing}"

    def test_combo_vram_within_8g(self):
        """Resident Combos must fit within 8192 MB (serial_swap can exceed)."""
        for combo_id, cfg in COMBO_3060TI.items():
            total = sum(cfg["models"].values())
            if cfg["strategy"] == "resident":
                assert total <= 8192, (
                    f"Combo {combo_id} (resident): {total}MB > 8192MB (8G hard cap)"
                )
            else:
                # serial_swap: only one model loaded at a time
                for model_id, vram in cfg["models"].items():
                    assert vram <= 8192, (
                        f"Combo {combo_id} serial model {model_id}: "
                        f"{vram}MB > 8192MB single-model cap"
                    )

    def test_valid_strategies(self):
        """Strategy must be 'resident' or 'serial_swap'."""
        valid = {"resident", "serial_swap"}
        for combo_id, cfg in COMBO_3060TI.items():
            assert cfg["strategy"] in valid, (
                f"Combo {combo_id}: invalid strategy '{cfg['strategy']}'"
            )

    def test_combo_models_in_registry(self):
        """All Combo model_ids must exist in MODELS."""
        for combo_id, cfg in COMBO_3060TI.items():
            for model_id in cfg["models"]:
                assert model_id in MODELS, (
                    f"Combo {combo_id}: unknown model {model_id}"
                )

    def test_combo_model_vram_matches_registry(self):
        """Combo model VRAM values should match MODELS registry."""
        for combo_id, cfg in COMBO_3060TI.items():
            for model_id, vram_mb in cfg["models"].items():
                model = MODELS.get(model_id)
                if model:
                    assert vram_mb == model["vram"], (
                        f"Combo {combo_id} model {model_id}: "
                        f"combo says {vram_mb}MB, registry says {model['vram']}MB"
                    )


class TestModelsRegistry:
    """Test MODELS registry completeness."""

    def test_all_models_have_required_fields(self):
        """Each model must have vram, precision, category, runtime."""
        required = {"vram", "precision", "category", "runtime", "combo_id",
                    "weight_path", "triton_hash"}
        for model_id, meta in MODELS.items():
            missing = required - set(meta.keys())
            assert not missing, f"Model {model_id} missing: {missing}"

    def test_categories_are_valid(self):
        """Category must be Heavy or Light."""
        valid = {"Heavy", "Light"}
        for model_id, meta in MODELS.items():
            assert meta["category"] in valid, (
                f"Model {model_id}: invalid category '{meta['category']}'"
            )

    def test_heavy_models_categorized(self):
        """HEAVY_MODELS should only contain Heavy models."""
        for model_id, meta in HEAVY_MODELS.items():
            assert meta["category"] == "Heavy"

    def test_light_models_categorized(self):
        """LIGHT_MODELS should only contain Light models."""
        for model_id, meta in LIGHT_MODELS.items():
            assert meta["category"] == "Light"

    def test_no_model_exceeds_3090_cap(self):
        """No model VRAM should exceed 21G (3090 hard cap)."""
        
        # Hardware: RTX 3090 24G (CUDA inference primary)
        # 3060Ti is no longer used for inference (display + NVENC/NVDEC + ffmpeg IO only)
        for model_id, meta in MODELS.items():
            assert meta["vram"] <= 21000, (
                f"Model {model_id}: {meta['vram']}MB > 21000MB cap"
            )

    def test_combo_models_fit_3060ti(self):
        """Light models with combo_id must fit in 7680MB (3060Ti cap).
        
        ⚠️ DEPRECATED — 3060Ti no longer used for inference.
        """
        for model_id, meta in LIGHT_MODELS.items():
            if meta["combo_id"]:
                assert meta["vram"] <= 7680, (
                    f"Model {model_id}: {meta['vram']}MB > 7680MB 3060Ti cap"
                )

    def test_precision_values(self):
        """Precision must be bf16 or fp16."""
        valid = {"bf16", "fp16"}
        for model_id, meta in MODELS.items():
            assert meta["precision"] in valid


class TestRoutingTable:
    """Test 35-node routing table completeness."""

    def test_routing_table_validates(self):
        """build_routing_table should succeed without errors."""
        table = build_routing_table()
        assert len(table) > 0

    def test_all_nodes_have_model_id(self):
        """Every node must have a model_id."""
        for node_id, node in ROUTING_TABLE.items():
            assert "model_id" in node, f"Node {node_id} missing model_id"

    def test_all_node_models_in_registry(self):
        """All node model_ids should exist in MODELS or be CPU tools."""
        cpu_models = {"ffmpeg_x264", "blender_cycles"}
        for node_id, node in ROUTING_TABLE.items():
            model_id = node["model_id"]
            if model_id in cpu_models:
                continue
            assert model_id in MODELS, (
                f"Node {node_id}: unknown model {model_id}"
            )

    def test_preview_and_heavy_slots_valid(self):
        """Preview and heavy slots must specify valid GPU.
        
        Valid GPUs: 3090 (inference primary), 3060ti (display/IO only, no inference)
        """
        valid_gpus = {"3090", "3060ti"}
        for node_id, node in ROUTING_TABLE.items():
            for slot in ("preview", "heavy"):
                entry = node.get(slot)
                if entry is not None:
                    assert entry.get("gpu") in valid_gpus, (
                        f"Node {node_id} {slot}: invalid GPU '{entry.get('gpu')}'"
                    )

    def test_overflow_combo_references_valid(self):
        """Overflow Combo references must be valid Combo IDs."""
        for node_id, node in ROUTING_TABLE.items():
            overflow = node.get("overflow")
            if overflow and overflow.get("combo"):
                combo_id = overflow["combo"]
                assert combo_id in COMBO_3060TI, (
                    f"Node {node_id}: unknown overflow combo '{combo_id}'"
                )

    def test_at_least_35_nodes(self):
        """V3.6 requires at least 35 task nodes."""
        assert len(ROUTING_TABLE) >= 35

    def test_cpu_nodes_have_cpu_entry(self):
        """CPU-only nodes must have a cpu entry."""
        cpu_model_ids = {"ffmpeg_x264", "blender_cycles"}
        for node_id, node in ROUTING_TABLE.items():
            if node["model_id"] in cpu_model_ids:
                assert node.get("cpu") is not None, (
                    f"Node {node_id}: CPU model without cpu entry"
                )
