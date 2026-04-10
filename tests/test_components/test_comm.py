import os
from unittest.mock import MagicMock, patch

import pytest


class TestCommConfigDefaults:
    """Test suite for CommConfig default values."""

    def test_default_init_timeout(self) -> None:
        """Test default init timeout is 300 seconds."""
        from fandb.components.comm import DEFAULT_INIT_TIMEOUT_SECONDS, CommConfig

        config = CommConfig()
        assert config.init_timeout_seconds == DEFAULT_INIT_TIMEOUT_SECONDS

    def test_default_train_timeout(self) -> None:
        """Test default train timeout is 100 seconds."""
        from fandb.components.comm import DEFAULT_TRAIN_TIMEOUT_SECONDS, CommConfig

        config = CommConfig()
        assert config.train_timeout_seconds == DEFAULT_TRAIN_TIMEOUT_SECONDS

    def test_default_trace_buf_size(self) -> None:
        """Test default trace buffer size is 20,000."""
        from fandb.components.comm import DEFAULT_TRACE_BUF_SIZE, CommConfig

        config = CommConfig()
        assert config.trace_buf_size == DEFAULT_TRACE_BUF_SIZE

    def test_default_save_traces_folder(self) -> None:
        """Test default save traces folder is 'comm_traces'."""
        from fandb.components.comm import CommConfig

        config = CommConfig()
        assert config.save_traces_folder == "comm_traces"

    def test_default_save_traces_file_prefix(self) -> None:
        """Test default save traces file prefix is 'rank_'."""
        from fandb.components.comm import CommConfig

        config = CommConfig()
        assert config.save_traces_file_prefix == "rank_"

    def test_default_mode(self) -> None:
        """Test default mode is 'default'."""
        from fandb.components.comm import CommConfig

        config = CommConfig()
        assert config.mode == "default"

    def test_default_enable_cpu_backend(self) -> None:
        """Test default enable_cpu_backend is True."""
        from fandb.components.comm import CommConfig

        config = CommConfig()
        assert config.enable_cpu_backend is True


class TestCommConfigMethods:
    """Test suite for CommConfig methods."""

    @patch("fandb.components.comm._get_distributed_backend")
    def test_get_distributed_backend_with_cpu_enabled(self, mock_backend: MagicMock) -> None:
        """Test get_distributed_backend returns correct backend when CPU is enabled."""
        from fandb.components.comm import CommConfig

        mock_backend.return_value = "gloo"
        config = CommConfig(enable_cpu_backend=True)
        result = config.get_distributed_backend()

        mock_backend.assert_called_once_with(enable_cpu_backend=True)
        assert result == "gloo"

    @patch("fandb.components.comm._get_distributed_backend")
    def test_get_distributed_backend_with_cpu_disabled(self, mock_backend: MagicMock) -> None:
        """Test get_distributed_backend returns correct backend when CPU is disabled."""
        from fandb.components.comm import CommConfig

        mock_backend.return_value = "nccl"
        config = CommConfig(enable_cpu_backend=False)
        result = config.get_distributed_backend()

        mock_backend.assert_called_once_with(enable_cpu_backend=False)
        assert result == "nccl"


class TestCommConfigInitDistributed:
    """Test suite for init_distributed method."""

    @patch("torch.distributed.is_initialized")
    def test_init_distributed_already_initialized(self, mock_is_initialized: MagicMock) -> None:
        """Test init_distributed skips when already initialized."""
        from fandb.components.comm import CommConfig

        mock_is_initialized.return_value = True

        config = CommConfig()
        with patch("fandb.components.comm.logger") as mock_logger:
            config.init_distributed()

        mock_logger.warning.assert_called_once()
        mock_is_initialized.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch.dict(os.environ, {"NGPU": "2"}, clear=False)
    def test_init_distributed_fake_backend_missing_ngpu(self, mock_is_initialized: MagicMock) -> None:
        """Test init_distributed raises when using fake_backend without NGPU."""
        from fandb.components.comm import CommConfig

        mock_is_initialized.return_value = False
        original_ngpu = os.environ.pop("NGPU", None)

        try:
            config = CommConfig(mode="fake_backend")
            with pytest.raises(ValueError, match="NGPU environment variable must be set"):
                config.init_distributed()
        finally:
            if original_ngpu:
                os.environ["NGPU"] = original_ngpu

    @patch("torch.distributed.is_initialized")
    def test_init_distributed_fake_backend_invalid_ngpu(self, mock_is_initialized: MagicMock) -> None:
        """Test init_distributed raises when NGPU is not a valid integer."""
        from fandb.components.comm import CommConfig

        mock_is_initialized.return_value = False
        with patch.dict(os.environ, {"NGPU": "invalid"}):
            config = CommConfig(mode="fake_backend")
            with pytest.raises(ValueError, match="NGPU environment variable must be a valid integer"):
                config.init_distributed()

    @patch("torch.distributed.is_initialized")
    @patch("fandb.components.comm.init_fake_mode")
    def test_init_distributed_fake_backend_success(
        self, mock_init_fake: MagicMock, mock_is_initialized: MagicMock
    ) -> None:
        """Test init_distributed works with fake_backend mode."""
        from fandb.components.comm import CommConfig

        mock_is_initialized.return_value = False
        with patch.dict(os.environ, {"NGPU": "4"}):
            config = CommConfig(mode="fake_backend")
            config.init_distributed()

        mock_init_fake.assert_called_once_with(4, "fake_backend")

    @patch("torch.distributed.is_initialized")
    @patch("fandb.components.comm.init_fake_mode")
    def test_init_distributed_local_tensor_mode(
        self, mock_init_fake: MagicMock, mock_is_initialized: MagicMock
    ) -> None:
        """Test init_distributed works with local_tensor mode."""
        from fandb.components.comm import CommConfig

        mock_is_initialized.return_value = False
        with patch.dict(os.environ, {"NGPU": "2"}):
            config = CommConfig(mode="local_tensor")
            config.init_distributed()

        mock_init_fake.assert_called_once_with(2, "local_tensor")

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.init_process_group")
    @patch("fandb.components.comm._warn_overwrite_env")
    def test_init_distributed_default_mode(
        self,
        mock_warn: MagicMock,
        mock_init_pg: MagicMock,
        mock_is_initialized: MagicMock,
    ) -> None:
        """Test init_distributed works with default mode."""
        from fandb.components.comm import DEFAULT_INIT_TIMEOUT_SECONDS, CommConfig

        mock_is_initialized.return_value = False

        with patch("fandb.components.comm._get_distributed_backend", return_value="gloo"):
            config = CommConfig(mode="default", trace_buf_size=0)
            config.init_distributed(base_folder="test_output")

        mock_init_pg.assert_called_once()
        call_kwargs = mock_init_pg.call_args.kwargs
        assert call_kwargs["backend"] == "gloo"
        assert call_kwargs["timeout"].seconds == DEFAULT_INIT_TIMEOUT_SECONDS

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.init_process_group")
    @patch("fandb.components.comm._warn_overwrite_env")
    def test_init_distributed_with_ranks(
        self,
        mock_warn: MagicMock,
        mock_init_pg: MagicMock,
        mock_is_initialized: MagicMock,
    ) -> None:
        """Test init_distributed with specific ranks."""
        from fandb.components.comm import CommConfig

        mock_is_initialized.return_value = False

        with patch("fandb.components.comm._get_distributed_backend", return_value="gloo"):
            config = CommConfig(mode="default", trace_buf_size=0)
            config.init_distributed(ranks=[0, 1, 2])

        call_kwargs = mock_init_pg.call_args.kwargs
        assert call_kwargs["_ranks"] == [0, 1, 2]


class TestCommConfigSetTrainTimeout:
    @patch("fandb.components.comm.set_pg_timeout")
    def test_set_train_timeout(self, mock_set_timeout: MagicMock) -> None:
        """Test set_train_timeout sets correct timeout."""
        from fandb.components.comm import CommConfig

        config = CommConfig(train_timeout_seconds=60)
        config.set_train_timeout()

        mock_set_timeout.assert_called_once()
        call_kwargs = mock_set_timeout.call_args.kwargs
        assert call_kwargs["timeout"].seconds == 60


class TestCommConfigFlightRecorder:
    """Test suite for flight recorder configuration."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.init_process_group")
    @patch("fandb.components.comm._warn_overwrite_env")
    def test_flight_recorder_env_vars_set(
        self,
        mock_warn: MagicMock,
        mock_init_pg: MagicMock,
        mock_is_initialized: MagicMock,
    ) -> None:
        """Test flight recorder environment variables are set correctly."""
        from fandb.components.comm import (
            ASYNC_ERROR_HANDLING,
            DUMP_ON_TIMEOUT,
            TRACE_BUFFER_SIZE,
            CommConfig,
        )

        mock_is_initialized.return_value = False

        with patch("fandb.components.comm._get_distributed_backend", return_value="gloo"):
            config = CommConfig(
                mode="default",
                trace_buf_size=1000,
                save_traces_folder="traces",
                save_traces_file_prefix="worker_",
            )
            config.init_distributed(base_folder="test_output")

        env_calls = mock_warn.call_args_list
        env_vars = [call[0][0] for call in env_calls]

        assert ASYNC_ERROR_HANDLING in env_vars
        assert TRACE_BUFFER_SIZE in env_vars
        assert DUMP_ON_TIMEOUT in env_vars

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.init_process_group")
    @patch("fandb.components.comm._warn_overwrite_env")
    def test_flight_recorder_disabled(
        self,
        mock_warn: MagicMock,
        mock_init_pg: MagicMock,
        mock_is_initialized: MagicMock,
    ) -> None:
        """Test flight recorder is disabled when trace_buf_size is 0."""
        from fandb.components.comm import (
            ASYNC_ERROR_HANDLING,
            TRACE_BUFFER_SIZE,
            TRACE_FILE,
            CommConfig,
        )

        mock_is_initialized.return_value = False

        with patch("fandb.components.comm._get_distributed_backend", return_value="gloo"):
            config = CommConfig(mode="default", trace_buf_size=0)
            config.init_distributed(base_folder="test_output")

        env_calls = mock_warn.call_args_list
        env_vars = [call[0][0] for call in env_calls]

        assert ASYNC_ERROR_HANDLING in env_vars
        assert TRACE_BUFFER_SIZE in env_vars
        trace_file_calls = [call for call in env_calls if call[0][0] == TRACE_FILE]
        assert len(trace_file_calls) == 0
