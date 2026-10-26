# Benchmark health checks

`watch.py` checks the suite service, served model identity, GPU use, available
disk space, prediction progress, empty responses, generation limits, and answer
formatting every 60 seconds. It follows model transitions automatically and exits
after the suite completes. A ten-minute interval without new predictions raises
a stall alert. Quality warnings remain separate from request failures.

The monitor reads prediction journals incrementally and tolerates a record that
is still being written. It writes `monitoring/health.json` and
`monitoring/alert-history.jsonl` inside the selected run directory. It does not
restart evaluations or change their protocol. Alerts are recorded locally;
this script does not send chat notifications.

```bash
evaluation/.venv/bin/python evaluation/monitoring/watch.py \
  --run-dir artifacts/evaluation/standard-v1-run \
  --suite-unit ming-benchmarks-standard-v1-run --interval 60

cat artifacts/evaluation/standard-v1-run/monitoring/health.json
```

Use `--once` for one check. For a persistent monitor, launch the command under a
separate user systemd service, using the same working directory as the suite.
The `--suite-unit` argument must match the service running your evaluation.
