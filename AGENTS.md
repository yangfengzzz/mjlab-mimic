# Development Workflow

**Always use Chinese**.
**Always use the `mjlab_sideflip` conda environment, not `uv run` or bare
`python`**.

```sh

# 1. Make changes.

# 2. Type check.
conda run -n mjlab_sideflip ty check  # Fast
conda run -n mjlab_sideflip pyright  # More thorough, but slower

# 3. Run tests.
conda run -n mjlab_sideflip python -m pytest tests/  # Single suite
conda run -n mjlab_sideflip python -m pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
conda run -n mjlab_sideflip python -m ruff format
conda run -n mjlab_sideflip python -m ruff check --fix
```

We've bundled common commands into a Makefile for convenience. Run these from
the `mjlab_sideflip` conda environment.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories.

# Skill Maintenance

After modifying any file under `src/mjlab/tasks/hopping/`, update
`.claude/commands/go2-hopping.md` to reflect the current state of the task.
The skill should stay accurate and concise — update or remove outdated entries
(gotchas, design decisions, reward weights) rather than accumulating stale notes.
Do not add entries for things that are already obvious from the code.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve
    iteration speed.
