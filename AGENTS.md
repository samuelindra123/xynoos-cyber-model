# Repository instructions

All agents must use the canonical repository working directory. Do not create additional permanent Git worktrees or repository clones. Session A and Session B operate sequentially using separate Git branches. Preserve all existing changes before switching branches.

- Maintain one permanent working tree. Use branches in the same checkout to isolate planning, implementation, remediation, and independent verification.
- Check `git status` before switching branches. Switch only after the working tree is clean or its changes have been safely preserved.
- Coordinate through branch commits, pull requests, and documented handoffs. Finish one session's modifications before the next session begins modifying the workspace.
- Keep local builds and tests small. Remove regenerable build output when it is no longer needed, and use bounded GitHub Actions workflows for heavier validation.
- Use repository-relative paths in tracked material. Keep personal paths, machine details, and credentials out of the repository.
