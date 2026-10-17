# Token-graph P0 receipt

- Date:
- Commit:
- GPU / driver:
- oneAPI:
- Build command:
- Run command:
- Host free GiB before run:
- `limited_graph` / `full_graph`:
- `replays` per arm:
- `spin_max`:
- `host_timeout_ms`:

## Configuration line

```text
[TOKEN GRAPH CONFIG] ...
[TOKEN GRAPH RING] ...
```

## CPU-work sweep

| CPU work us | measured | fresh bit-exact | stale detected | route->flag median us | route->flag p99 us | handoff median us | handoff p99 us | replay wall median us |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 |  |  |  |  |  |  |  |  |
| 50 |  |  |  |  |  |  |  |  |
| 200 |  |  |  |  |  |  |  |  |
| 500 |  |  |  |  |  |  |  |  |

## Correctness

- Fresh replay timeout count:
- Fresh replay stale count:
- Fresh replay mismatch count:
- Host route stale count:
- Negative-control stale detected:
- Negative-control mismatch detected:
- Output/ring bit-exact:

## Verdict

- P0 GO / NO-GO:
- Reason:
- If GO, next action:
- If NO-GO, observed failure mode:

## Raw output

Paste the complete `[TOKEN GRAPH ...]` output and the `run.ps1` header/footer.
