## Semantic Versioning in Practice

### Patch Versions (1.0.0 → 1.0.1)

**When to use:** Bug fixes, documentation, no schema changes

**Migration required?** No

### Minor Versions (1.0.0 → 1.1.0)

**When to use:**

- Adding optional fields with defaults
- Marking `backward_compatible=True`

**Migration required?** No (if backward_compatible)

### Major Versions (1.0.0 → 2.0.0)

**When to use:**

- Removing fields
- Making optional fields required
- Changing field types
- Renaming fields

**Migration required?** Yes

## The Backward Compatible Flag

### When to Use It

- ✅ Adding fields with defaults
- ✅ Making required fields optional
- ✅ Adding validation that accepts all previous values

### When NOT to Use It

- ❌ Removing fields (they'll just be ignored)
- ❌ Changing field types
- ❌ Complex transformations needed
