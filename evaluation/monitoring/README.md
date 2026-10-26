# Benchmark health checks

`watch.py` checks the suite's shared-volume lock, served model identity, GPU use,
free disk space, prediction progress, empty responses, generation limits, and
answer formatting every 60 seconds. It follows model transitions and exits after
the suite completes. Ten minutes without predictions raises a stall alert.
Quality warnings remain separate from request failures.

After starting the `ming-eval` container, run the monitor inside that container
so its health requests reach the same inference server:

```bash
docker exec -d ming-eval /opt/ming/evaluation/.venv/bin/python \
  evaluation/monitoring/watch.py --run-dir /work/full-suite --interval 60
cat "$MING_WORK_DIR/full-suite/monitoring/health.json"
```

For one check, omit `-d` and append `--once`. The monitor reads journals
incrementally, tolerates incomplete final records, and writes `health.json`
and `alert-history.jsonl` under the run's `monitoring/` directory. It neither
restarts jobs nor changes their protocol. Alerts remain local. The monitor exits
when its container stops; rerun the command after resuming a suite.
