# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Before Writing Code

Read `pyproject.toml` first and use it as ground truth for line length, quote style, lint rules, mypy flags, and test options — never assume defaults.

## Commands

```bash
hatch fmt          # Format and lint (ruff)
hatch test         # Run all tests with coverage
hatch run typing   # Run mypy type checking
hatch run release  # Full check: fmt + typing + test
hatch run update   # Update dependency lockfiles
```

Run a single test file:
```bash
hatch test tests/test_components/test_configuration.py
```

Run a single test function:
```bash
hatch test -k test_function_name
```

**Do not** install pytest directly into the current Python environment — always use `hatch test`.

## Architecture

**fandb** (`src/fandb/`) is a PyTorch-native distributed training library for education, built on PyTorch's modern distributed APIs (device mesh, DTensor, functional collectives).

### `components/`
- `configuration.py` — `ConfigMixin` for nested dataclass serialization/deserialization. Supports dot-path dict format (e.g., `"model.hidden_size": 512`) and nested dict format. Built on `dacite` for type-safe deserialization.

### `distributed/`
The core of the library. Each file targets a specific distributed training concern:

| File | Purpose |
|------|---------|
| `ddp.py` | `DDPConfig` + hook registration for DDP (supports FP16/BF16 compression, PowerSGD, batched PowerSGD) |
| `device_mesh.py` | Device mesh initialization for DP, TP, PP, CP, EP, FSDP — enums + factory functions |
| `data_loader.py` | Distributed data loader wrapping accelerate's `DataLoaderShard`/`DataLoaderDispatcher`; handles batch splitting, RNG sync, stateful loading |
| `grad_norm.py` | Gradient norm clipping across PP/DP/TP process group dimensions |
| `reduce.py` | Generic distributed reductions (`all_reduce`, `max`) with DTensor support |
| `context_parallel.py` | Context Parallel (CP) for sequence dimension sharding |
| `loss_parallel.py` | Loss parallel utilities |
| `state.py` | Thin wrapper around accelerate's `PartialState` |
| `utils.py` | Device init, backend selection (NCCL/GLOO/fake), process group timeout management |

The fake backend (`utils.py`) enables local testing without actual distributed setup.

## Code Conventions

- **All function signatures must be fully type-annotated** (parameters and return types).
- **Flat over nested** — use early returns and guard clauses.
- **One thing per function** — function does exactly what its name says.
- **No magic numbers** — define default constants at module top in `UPPER_SNAKE_CASE` with explicit types. Use them in function signatures or dataclass defaults. In tests, import these constants to assert default values instead of hard-coding them.
- Avoid `lambda` beyond trivial cases, bare `except:`, mutable default arguments, and `*args/**kwargs` without justification.
- Any clever one-liner or magic must have a docstring or inline comment explaining what and why.
- Do not silence `mypy` errors with `# type: ignore` without an explanatory comment.

## Testing Conventions

- Follow **Arrange / Act / Assert** with explicit comments separating the three phases.
- Test function naming: `test_<what>_<condition>_<expected_outcome>`
- One assertion *concept* per test (multiple `assert` lines are fine if they verify the same outcome).
- No loops or conditionals inside test bodies — use `@pytest.mark.parametrize` for variations.
- Use pytest fixtures over setUp; test behavior, not internals (never assert on private attributes).

## Pre-Commit Checklist

- `hatch fmt` passes with no changes
- `hatch test` passes
- `hatch run typing` passes
- All functions have type annotations
- No unexplained magic or silenced errors
