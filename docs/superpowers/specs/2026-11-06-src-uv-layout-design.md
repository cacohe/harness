# src Package and uv Project Design

## Intent

Treat this repository as an internal Python application: it is not built or distributed as a
wheel, `src` is the import package, and uv owns dependency resolution and the local environment.

## Constraints

- Preserve all business behavior.
- Change only package markers, import paths, dependency metadata, lock data, CI, and tests that
  enforce the chosen layout.
- Import project modules through `src`, for example `from src.core.task import Task`.
- Run application modules from the repository root with `python -m src.<module>`.
- Configure uv with `package = false`; do not declare a build backend.

## Approach

The selected approach makes `src` an explicit regular Python package and keeps each existing
subdirectory in place. The rejected alternatives were treating `src` as a search root with
top-level imports such as `core`, or adding a second `agent_harness` package layer. Both conflict
with the requested import convention.

Runtime dependencies remain standard project dependencies. Test and lint tools move to the uv
`dev` dependency group. `uv.lock` is committed and CI uses locked synchronization.

