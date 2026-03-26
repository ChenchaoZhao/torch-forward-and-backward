# Testing Rules

* Use `hatch fmt` to format and lint.
* Use `hatch test` to run unit test.
* Do NOT try to install pytest into current python environment
* Use `hatch run typing` to run type checks
* Finally run `hatch run release` which run format, typing, and testing

## Philosophy

Tests are **documentation**. A failing test must tell the reader exactly what broke and why — without requiring them to read the implementation.

## Structure: Arrange / Act / Assert

Always separate the three phases explicitly with comments:

```python
def test_invoice_total_includes_tax() -> None:
    # Arrange
    invoice = Invoice(subtotal=100.0, tax_rate=0.1)

    # Act
    total = invoice.total()

    # Assert
    assert total == 110.0
```

## Naming

- Test file: `test_<module>.py`
- Test function: `test_<what>_<condition>_<expected_outcome>`

```python
def test_parse_date_with_invalid_string_raises_value_error() -> None: ...
def test_user_is_active_after_email_verification() -> None: ...
```

## Rules

- **One assertion concept per test.** Multiple `assert` lines are fine if they all verify the same logical outcome.
- **No logic in tests.** No loops, no conditionals. Use `@pytest.mark.parametrize` for variations.
- **Fixtures over setUp.** Use `pytest` fixtures for shared setup; keep them small and focused.
- **Test behavior, not implementation.** Tests must survive a refactor of internals.

```python
# Good — tests observable behavior
def test_cart_is_empty_after_clear() -> None:
    # Arrange
    cart = Cart(items=[Item("apple"), Item("banana")])

    # Act
    cart.clear()

    # Assert
    assert len(cart) == 0


# Bad — reaches into internals
def test_cart_clear_resets_internal_list() -> None:
    cart = Cart(items=[Item("apple")])
    cart.clear()
    assert cart._items == []  # never assert on private attributes
```

## Parametrize for Variations

```python
@pytest.mark.parametrize(("raw", "expected"), [
    ("2024-01-01", date(2024, 1, 1)),
    ("2024-12-31", date(2024, 12, 31)),
])
def test_parse_date_returns_correct_date(raw: str, expected: date) -> None:
    assert parse_date(raw) == expected
```

## Pre-Commit Checklist

- [ ] Test names describe behavior, not implementation
- [ ] Every test follows Arrange / Act / Assert
- [ ] No loops or conditionals inside test bodies
- [ ] No assertions on private attributes or internal state
- [ ] `hatch test` passes
