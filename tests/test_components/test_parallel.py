import contextlib
from unittest import mock

import pytest

from fandb.components.parallel import ParallelConfig
from fandb.components.precision import TorchDtypeEnum
from fandb.distributed.device_mesh import DeviceTypeEnum, DPBackend


class TestParallelConfigDefaults:
    def test_default_initialization(self) -> None:
        config = ParallelConfig()
        assert config.device_type == DeviceTypeEnum.CUDA
        assert config.dp_backend == DPBackend.FSDP
        assert config.dp_shard_size == 1
        assert config.tp_size == 1
        assert config.cp_size == 1
        assert config.pp_size == 1
        assert config.ep_size == 1
        assert config.etp_size == 1

    def test_default_non_distributed(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.is_distributed is False

    def test_default_fsdp_distributed(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.FSDP)
        assert config.is_distributed is True


class TestParallelConfigValidation:
    def test_valid_dp_shard_size(self) -> None:
        config = ParallelConfig(dp_shard_size=2)
        assert config.dp_shard_size == 2

    def test_invalid_dp_shard_size_zero(self) -> None:
        with pytest.raises(ValueError, match="dp_shard_size must be a positive integer"):
            ParallelConfig(dp_shard_size=0)

    def test_invalid_dp_shard_size_negative(self) -> None:
        with pytest.raises(ValueError, match="dp_shard_size must be a positive integer"):
            ParallelConfig(dp_shard_size=-1)

    def test_valid_tp_size(self) -> None:
        config = ParallelConfig(tp_size=4)
        assert config.tp_size == 4

    def test_invalid_tp_size_zero(self) -> None:
        with pytest.raises(ValueError, match="tp_size must be a positive integer"):
            ParallelConfig(tp_size=0)

    def test_invalid_tp_size_negative(self) -> None:
        with pytest.raises(ValueError, match="tp_size must be a positive integer"):
            ParallelConfig(tp_size=-1)


class TestParallelConfigProperties:
    def test_has_nd_parallel_fsdp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.FSDP)
        assert config.has_nd_parallel is True

    def test_has_nd_parallel_no(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.has_nd_parallel is False

    def test_has_nd_parallel_ddp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.DDP)
        assert config.has_nd_parallel is False

    def test_non_data_parallel_size_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.non_data_parallel_size == 1

    def test_non_data_parallel_size_with_tp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, tp_size=4)
        assert config.non_data_parallel_size == 4

    def test_non_data_parallel_size_with_cp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, cp_size=2)
        assert config.non_data_parallel_size == 2

    def test_non_data_parallel_size_with_pp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, pp_size=2)
        assert config.non_data_parallel_size == 2

    def test_non_data_parallel_size_combined(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, tp_size=2, cp_size=2, pp_size=2)
        assert config.non_data_parallel_size == 8

    def test_seq_len_divisor_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.seq_len_divisor == 2

    def test_seq_len_divisor_with_tp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, tp_size=4)
        assert config.seq_len_divisor == 8

    def test_seq_len_divisor_with_cp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, cp_size=2)
        assert config.seq_len_divisor == 4

    def test_seq_len_divisor_with_tp_and_cp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, tp_size=4, cp_size=2)
        assert config.seq_len_divisor == 16


class TestParallelConfigDpProperties:
    def test_dp_enabled_false_with_no_backend(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.dp_enabled is False

    def test_dp_enabled_true_with_shard(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, dp_shard_size=2)
        assert config.dp_enabled is True

    def test_dp_replicate_enabled_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.dp_replicate_enabled is False

    def test_dp_shard_enabled_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.dp_shard_enabled is False

    def test_dp_shard_enabled_true(self) -> None:
        config = ParallelConfig(dp_shard_size=2)
        assert config.dp_shard_enabled is True

    def test_dp_shard_enabled_false_with_no_backend(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.dp_shard_enabled is False

    def test_cp_enabled_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.cp_enabled is False

    def test_cp_enabled_true(self) -> None:
        config = ParallelConfig(cp_size=2)
        assert config.cp_enabled is True

    def test_cp_enabled_false_with_no_backend(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, cp_size=2)
        assert config.cp_enabled is False

    def test_dp_cp_enabled_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.dp_cp_enabled is False

    def test_dp_cp_enabled_with_cp_and_fsdp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.dp_cp_enabled is False

    def test_fsdp_enabled_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.fsdp_enabled is False

    def test_fsdp_enabled_with_cp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, cp_size=2)
        assert config.fsdp_enabled is False

    def test_fsdp_enabled_with_dp_shard(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, dp_shard_size=2)
        assert config.fsdp_enabled is False

    def test_tp_enabled_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.tp_enabled is False

    def test_tp_enabled_true(self) -> None:
        config = ParallelConfig(tp_size=4, dp_backend=DPBackend.FSDP)
        assert config.tp_enabled is True

    def test_tp_enabled_false_with_no_backend(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, tp_size=4)
        assert config.tp_enabled is False

    def test_pp_enabled_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.pp_enabled is False

    def test_pp_enabled_true_with_fsdp(self) -> None:
        config = ParallelConfig(pp_size=2, dp_backend=DPBackend.FSDP)
        assert config.pp_enabled is True

    def test_ep_enabled_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.ep_enabled is False

    def test_ep_enabled_true_with_fsdp(self) -> None:
        config = ParallelConfig(ep_size=2, dp_backend=DPBackend.FSDP)
        assert config.ep_enabled is True

    def test_etp_enabled_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.etp_enabled is False

    def test_etp_enabled_true_with_fsdp(self) -> None:
        config = ParallelConfig(etp_size=2, dp_backend=DPBackend.FSDP)
        assert config.etp_enabled is True


class TestParallelConfigDevice:
    def test_device_non_distributed_cpu(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, device_type=DeviceTypeEnum.CPU)
        config.set_device()
        assert config.device.type == "cpu"

    def test_device_non_distributed_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        config.set_device()
        assert config.device.type == "cuda"


class TestParallelConfigAutocast:
    def test_autocast_fsdp_returns_nullcontext(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.FSDP)
        result = config.maybe_enable_amp_autocast(TorchDtypeEnum.FP16)
        assert isinstance(result, type(contextlib.nullcontext()))

    def test_autocast_ddp_returns_autocast(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.DDP)
        result = config.maybe_enable_amp_autocast(TorchDtypeEnum.FP16)
        assert result is not None

    def test_autocast_no_returns_autocast(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        result = config.maybe_enable_amp_autocast(TorchDtypeEnum.FP16)
        assert result is not None


class TestParallelConfigDpReplicateSizeInference:
    def test_infer_dp_replicate_size_no_backend(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config._infer_dp_replicate_size() == 1

    def test_dp_world_size_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.dp_world_size == 1

    def test_fsdp_gradient_divide_factor_default(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO)
        assert config.fsdp_gradient_divide_factor == 1

    def test_fsdp_gradient_divide_factor_with_cp(self) -> None:
        config = ParallelConfig(dp_backend=DPBackend.NO, cp_size=2)
        assert config.fsdp_gradient_divide_factor == 2


class TestParallelConfigInferDpReplicateSize:
    def test_infer_dp_replicate_size_with_mock_world_size(self) -> None:
        import torch.distributed as dist

        config = ParallelConfig(dp_backend=DPBackend.FSDP, dp_shard_size=2, tp_size=2)
        with mock.patch.object(dist, "get_world_size", return_value=8):
            result = config._infer_dp_replicate_size()
            assert result == 2

    def test_infer_dp_replicate_size_invalid_world_size(self) -> None:
        import torch.distributed as dist

        config = ParallelConfig(dp_backend=DPBackend.FSDP, dp_shard_size=2, tp_size=3)
        with (
            mock.patch.object(dist, "get_world_size", return_value=8),
            pytest.raises(ValueError, match="world_size"),
        ):
            config._infer_dp_replicate_size()

    def test_infer_dp_replicate_size_ddp_returns_world_size(self) -> None:
        import torch.distributed as dist

        config = ParallelConfig(dp_backend=DPBackend.DDP)
        with mock.patch.object(dist, "get_world_size", return_value=4):
            result = config._infer_dp_replicate_size()
            assert result == 4
