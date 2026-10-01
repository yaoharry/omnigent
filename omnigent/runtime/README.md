# runtime

Shared libraries for prompt composition, policies, history, compaction, streams,
and the harness process contract. Session and turn orchestration live in
`omnigent/runner`; provider and launch configuration live in
`omnigent/harnesses/config`.

`RuntimeServices` holds process-scoped dependencies for existing getter-based
callers. Library operations can take dependencies directly, as
`CompactionServices` does. `workflow.py` only preserves old imports until 0.18.

See [Architecture](../../docs/ARCHITECTURE.md) for ownership, module boundaries,
and verification commands.
