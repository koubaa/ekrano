# Xtask dev utilities

Kompari-based diffing between reference snapshots and latest outputs.

```bash
cargo xtask snapshots report   # HTML report: snapshots/ vs current/
cargo xtask snapshots review   # Interactive review
cargo xtask snapshots dead-snapshots  # Find orphan reference images
cargo xtask snapshots size-check     # Size check
```

Paths are rooted at `ekrano_tests/snapshots` (left) and `ekrano_tests/current` (right).
