# AGENTS.md

## Initial Exploration

Before writing any code, explore the project structure and understand the conventions:

### Read First (Priority Order)

1. **`pyproject.toml`** — Single source of truth for all configuration
2. **`README.md`** — Project overview and setup instructions
3. **`AGENTS.md`** — Agent-specific instructions if it exists

### Extract Key Configuration

From `pyproject.toml`, identify:

| Setting          | Location                              |
| ---------------- | ------------------------------------- |
| Package name     | `[project].name`                      |
| Python version   | `[project].requires-python`           |
| Source directory | `[tool.hatch.build.targets.wheel]`    |
| Test directory   | `[tool.pytest.ini_options].testpaths` |
| Max line length  | `[tool.ruff].line-length`             |
| Mypy config      | `[tool.mypy]`                         |

### Common Patterns to Match

- Import style and organization
- Type annotation conventions
- Naming conventions (modules, classes, functions)
- Error handling patterns
- Documentation approach

## Toolchain Standard

Standard development commands (verify in `pyproject.toml`):

- `hatch fmt` — Format + lint (ruff)
- `hatch test` — Run tests (pytest)
- `hatch run typing` — Type check (mypy)
- `hatch run release` — fmt + typing + test

### Tool Purposes

Ask human to install those tools if they are not installed yet.

| Tool | Purpose |
|------|---------|
| **hatch** | Project and environment management |
| **uv** | Fast dependency resolution |
| **ruff** | Linting and formatting |
| **mypy** | Static type checking |
| **pytest** | Test execution |

## Behavioral Guidelines

### Think Before Coding

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so.
- If something is unclear, stop and ask.

### Simplicity First

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" that wasn't requested.
- If you write 200 lines and it could be 50, rewrite it.

### Surgical Changes

- Touch only what you must.
- Don't "improve" adjacent code.
- Match existing style, even if you'd do it differently.
- Clean up only YOUR changes' orphans.

### Goal-Driven Execution

- Define verifiable success criteria:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

### Additional Pythonic Behaviors

- Follow the **Zen of Python** (`import this`):
- Explicit is better than implicit.
- Simple is better than complex.
- Readability counts.
- If the implementation is hard to explain, it's a bad idea.
- **Flat over nested.** Prefer early returns and guard clauses over deeply indented logic.
- **One thing per function.** A function should do exactly what its name says.
- **Name things well.** A good name is worth more than a comment.

## Code Conventions

### No Magic Numbers

Define default constants at module top in `UPPER_SNAKE_CASE` with explicit types. Use these constants in function signatures, `dataclass` defaults or doc strings instead of hard-coding values.

```python
DEFAULT_INIT_TIMEOUT_SECONDS: int = 300

def init(timeout: int = DEFAULT_INIT_TIMEOUT_SECONDS) -> None: ...
```

In tests, import constants to assert default values:

```python
from mypackage.some_module import DEFAULT_BATCH_SIZE

def test_default_batch_size():
    assert config.batch_size == DEFAULT_BATCH_SIZE
```

### Docstring Defaults

Use string templating to keep docstrings in sync with defaults:

```python
DEFAULT_LR: float = 8e-4

_OPTIMIZER_CONFIG_DOC = f"Learning rate (Defaults to {DEFAULT_LR})"

@dataclass
class OptimizerConfig:
    __doc__ = _OPTIMIZER_CONFIG_DOC
```

### Magic and One-Liners

Occasional clever code is acceptable when it meaningfully improves performance or reduces boilerplate — but it **must** have a docstring or inline comment explaining what it does and why.

```python
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

### Enum is Preferred

Enums are better than arbitrary string inputs

```python
class SchedulerNameEnum(StrEnum):
    LINEAR = auto()
    COSINE = auto()
    CONSTANT = auto()

# automatic validation
string_input = "linear"
scheduler_name = SchedulerNameEnum(string_input)
```

### Type Annotations

- Annotate ALL function signatures (parameters and return types)
- Use `TypeAlias` and `NewType` over bare primitives when semantic meaning matters
- Fix `mypy` errors; do not silence with `# type: ignore` without explanation
- Prefer `TypeAlias` and `NewType` over bare primitives when the semantic meaning matters.

## Testing Rules

### Structure: Arrange / Act / Assert

```python
def test_invoice_total_includes_tax() -> None:
    # Arrange
    invoice = Invoice(subtotal=100.0, tax_rate=0.1)

    # Act
    total = invoice.total()

    # Assert
    assert total == 110.0
```

### Naming Conventions

- Test file: `test_<module>.py`
- Test function: `test_<what>_<condition>_<expected_outcome>`

```python
def test_parse_date_with_invalid_string_raises_value_error() -> None: ...
def test_user_is_active_after_email_verification() -> None: ...
```

### Rules

- **One assertion concept per test**
- **No logic in tests** — No loops, no conditionals. Use `@pytest.mark.parametrize`
- **Fixtures over setUp** — Use pytest fixtures for shared setup
- **Test behavior, not implementation** — Tests must survive refactoring

### Parametrize for Variations

```python
@pytest.mark.parametrize(("raw", "expected"), [
    ("2024-01-01", date(2024, 1, 1)),
    ("2024-12-31", date(2024, 12, 31)),
])
def test_parse_date_returns_correct_date(raw: str, expected: date) -> None:
    assert parse_date(raw) == expected
```

## CI Pipeline Setup

Typical CI pipeline stages:

1. **Typing** — `hatch run typing` (mypy)
2. **Test** — `hatch test` (pytest)

Note: Lint is typically NOT run in CI (ran locally).

## Pre-Commit Checklist

Before submitting changes:

- [ ] `hatch fmt` passes with no changes
- [ ] `hatch test` passes
- [ ] `hatch run typing` passes
- [ ] All functions have type annotations
- [ ] No silenced errors without explanation
- [ ] No unexplained magic or one-liners

## Common Gotchas

- Do NOT install pytest directly; use `hatch test`
- `requirements.txt` is auto-generated — do not edit manually
- Dependencies go in `pyproject.toml`, not in `requirements.txt`
- Never leave project in state where `hatch fmt` or `hatch test` fails