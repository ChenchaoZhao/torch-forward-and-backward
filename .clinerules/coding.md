# Coding Rules

## Setup: Read `pyproject.toml` First

Before writing any code, read `pyproject.toml` and extract the active configuration. Use those values as ground truth — do not assume defaults if a value is explicitly set.

| Setting | Location in `pyproject.toml` |
|---------|-------------------------------|
| Max line length | `[tool.ruff]` → `line-length` |
| Quote style | `[tool.ruff.format]` → `quote-style` |
| Selected / ignored lint rules | `[tool.ruff.lint]` → `select`, `ignore` |
| Mypy strictness flags | `[tool.mypy]` |
| Test paths and options | `[tool.hatch.envs.default.scripts]` or `[tool.pytest.ini_options]` |

## Toolchain

| Tool | Purpose |
|------|---------|
| **hatch** | Project and environment management |
| **uv** | Fast dependency resolution |
| **ruff** | Linting and formatting |
| **mypy** | Static type checking |

```bash
hatch fmt        # Format and lint (runs ruff)
hatch test       # Run unit tests
```

## Project Conventions

- `pyproject.toml` is the single source of truth. Dependencies go there, not in `requirements.txt`.
- Never leave the project in a state where `hatch fmt` or `hatch test` fails.

## Philosophy

Follow the **Zen of Python** (`import this`):

- Explicit is better than implicit.
- Simple is better than complex.
- Readability counts.
- If the implementation is hard to explain, it's a bad idea.

## General Rules

- **Flat over nested.** Prefer early returns and guard clauses over deeply indented logic.
- **One thing per function.** A function should do exactly what its name says.
- **Name things well.** A good name is worth more than a comment.

## Types

- Annotate **all** function signatures (parameters and return types).
- Fix `mypy` errors; do not silence them with `# type: ignore` without a comment explaining why.
- Prefer `TypeAlias` and `NewType` over bare primitives when the semantic meaning matters.

```python
# Good
def get_user(user_id: int) -> User: ...

# Bad
def get_user(id): ...
```

## What to Avoid

| Avoid | Prefer |
|-------|--------|
| `lambda` beyond trivial cases | Named `def` |
| `*args / **kwargs` without good reason | Explicit parameters |
| Mutable default arguments | `None` with guard |
| Bare `except:` | `except SomeError as e:` |

## No Magic Numbers

Define any default constants at the top of the module in UPPER_SNAKE_CASE with explicit types. Use these constants in function signatures or dataclass defaults instead of hard-coding values.

```python
# Good
DEFAULT_INIT_TIMEOUT_SECONDS: int = 300

def init(timeout: int = DEFAULT_INIT_TIMEOUT_SECONDS) -> None: ...

# Good — dataclass defaults
DEFAULT_BATCH_SIZE: int = 32

@dataclass
class Config:
    batch_size: int = DEFAULT_BATCH_SIZE
```

**In tests**, import these constants to assert default values instead of hard-coding them:

```python
# Good
from fandb.some_module import DEFAULT_BATCH_SIZE

def test_default_batch_size():
    assert config.batch_size == DEFAULT_BATCH_SIZE

# Bad
assert config.batch_size == 32
```

## Magic and One-Liners

Occasional clever code is acceptable when it meaningfully improves performance or reduces boilerplate — but it **must** have a docstring or inline comment explaining what it does and why.

```python
# Good — one-liner with explanation
def active_ids(users: list[User]) -> list[int]:
    """Return ids of active users. List comp used for performance over large inputs."""
    return [u.id for u in users if u.is_active]

# Good — magic with explanation
class Registry:
    """
    Uses __init_subclass__ to auto-register subclasses by name.
    This avoids manual registration boilerplate in every subclass.
    """
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        Registry._registry[cls.__name__] = cls

# Bad — clever with no explanation
return {k: v for d in maps for k, v in d.items() if v is not None}
```

**Rule: if you have to think twice to read it, you must explain it.**

## Pre-Commit Checklist

- [ ] `hatch fmt` passes with no changes
- [ ] `hatch test` passes
- [ ] All functions have type annotations
- [ ] No silenced `mypy` or `ruff` errors without explanation
- [ ] No unexplained magic or one-liners
