# Exceptions

Here are the custom exceptions pyrmute raises. You can import them
directly from `pyrmute`:

```python
from pyrmute import ModelNotFoundError, MigrationError, InvalidVersionError
```

::: pyrmute.ModelNotFoundError

::: pyrmute.MigrationError

::: pyrmute.InvalidVersionError

They are all a base class of:

::: pyrmute.VersionedModelError
