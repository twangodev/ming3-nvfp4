import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest
import requests

import runtime
import suite as runner


def test_shared_run_lock_tracks_owner(tmp_path):
    path = tmp_path / '.suite.lock'
    assert not runtime.lock_held(path)
    with path.open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert runtime.lock_held(path)
    assert not runtime.lock_held(path)


def test_stop_reaps_owned_process_and_preserves_unrelated_process():
    own = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        runtime.stop(own, timeout=1)
        assert own.returncode == -signal.SIGTERM
        assert other.poll() is None
    finally:
        own.kill() if own.poll() is None else None
        own.wait()
        other.terminate()
        other.wait()


def test_stop_kills_worker_even_after_parent_exits(tmp_path):
    # The worker ignores TERM; a pipe drains its PID before the leader exits.
    script = '''import os, signal, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
    time.sleep(30)
else:
    time.sleep(.1)
'''
    process = subprocess.Popen([sys.executable, '-c', script], start_new_session=True,
                               stdout=subprocess.PIPE, text=True)
    child = int(process.stdout.readline())
    process.wait(timeout=5)
    try:
        runtime.stop(process, timeout=.2)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = Path(f'/proc/{child}/stat')
            if not status.exists() or status.read_text().split()[2] == 'Z':
                break
            time.sleep(.05)
        else:
            pytest.fail('Owned inference worker survived shutdown')
    finally:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.stdout.close()


def test_readiness_rejects_wrong_model_and_early_exit(monkeypatch):
    process = Mock(returncode=7)
    process.poll.return_value = 7
    with pytest.raises(RuntimeError, match='exited with 7'):
        runtime.wait_ready(process, 'expected', 'http://example.invalid')
    process.poll.return_value = None
    response = Mock()
    response.json.return_value = {'data': [{'id': 'different'}]}
    monkeypatch.setattr(runtime.requests, 'get', lambda *a, **kw: response)
    with pytest.raises(RuntimeError, match='Unexpected model'):
        runtime.wait_ready(process, 'expected', 'http://example.invalid')
    response.json.return_value = {'data': [{'id': 'expected'}]}
    runtime.wait_ready(process, 'expected', 'http://example.invalid')


@pytest.fixture
def fake_suite(tmp_path, monkeypatch):
    config = tmp_path / 'models.json'
    config.write_text(json.dumps([{'name': 'tiny', 'checkpoint': '/models/tiny', 'concurrency': 4}]))
    def disconnected(*a, **kw):
        raise requests.ConnectionError()
    monkeypatch.setattr(runner.requests, 'get', disconnected)
    process = Mock()
    monkeypatch.setattr(runner, 'launch', Mock(return_value=process))
    monkeypatch.setattr(runner, 'wait_ready', Mock())
    monkeypatch.setattr(runner, 'run', Mock())
    monkeypatch.setattr(runner, 'score', Mock())
    monkeypatch.setattr(runner, 'stop', Mock())
    def invoke():
        runner.suite(config, tmp_path / 'run', tmp_path / 'manifest', tmp_path / 'serving',
                     8091, limit=2, min_free_gib=0)
    return invoke, process, tmp_path / 'run'


@pytest.mark.parametrize('failure', [RuntimeError('startup failed'), SystemExit(143)])
def test_suite_cleans_up_on_failure_or_signal(fake_suite, failure):
    invoke, process, output = fake_suite
    previous = signal.getsignal(signal.SIGTERM)
    runner.wait_ready.side_effect = failure
    with pytest.raises(type(failure)):
        invoke()
    runner.stop.assert_called_once_with(process)
    assert json.loads((output / 'suite-progress.json').read_text())['state'] == 'failed'
    assert not runtime.lock_held(output / '.suite.lock')
    assert signal.getsignal(signal.SIGTERM) == previous


def test_suite_marks_bounded_run_and_releases_server(fake_suite):
    invoke, process, output = fake_suite
    invoke()
    runner.stop.assert_called_once_with(process)
    assert runner.run.call_args.kwargs['limit'] == 2
    assert json.loads((output / 'suite-progress.json').read_text()) == {
        'state': 'complete', 'scope': 'smoke', 'models': ['tiny'], 'min_free_gib': 0}


def test_suite_refuses_existing_listener_without_stopping_it(fake_suite, monkeypatch):
    invoke, _, _ = fake_suite
    monkeypatch.setattr(runner.requests, 'get', Mock())
    with pytest.raises(RuntimeError, match='already occupied'):
        invoke()
    runner.launch.assert_not_called()
    runner.stop.assert_not_called()


def test_image_evaluators_match_submodule_revisions():
    root = Path(__file__).resolve().parents[2]
    # In-image clones are validated separately during the build.
    if not (root / '.git').exists():
        pytest.skip('Host checkout-only gitlink consistency check')
    manifest = json.loads((root / 'docker/upstream.json').read_text())
    for name, (_, revision) in manifest.items():
        tracked = subprocess.check_output(['git', 'ls-tree', 'HEAD', f'evaluation/upstream/{name}'],
                                          cwd=root, text=True).split()[2]
        assert tracked == revision


def test_resume_keeps_successes_and_rejects_changed_container(tmp_path, monkeypatch):
    import client

    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps({'id': name, 'benchmark': 'test', 'audio': None}) + '\n'
                                for name in ('first', 'second')))
    monkeypatch.setattr(client, 'REGISTRY', {'test': (None, None, None, 2)})
    monkeypatch.setattr(client, 'provenance', lambda: {'implementation': {'test': 'stable'}})
    monkeypatch.setattr(client, 'checkpoint_identity', lambda model: {'path': model})
    response = Mock()
    response.json.return_value = {'data': [{'id': '/models/tiny'}]}
    monkeypatch.setattr(client.requests, 'get', lambda *a, **kw: response)
    def prediction(name):
        return {'id': name, 'benchmark': 'test', 'text': 'answer', 'finish_reason': 'stop'}
    inference = Mock(side_effect=[prediction('first'), RuntimeError('interrupted request')])
    monkeypatch.setattr(client, 'request', inference)
    output = tmp_path / 'run'
    output.mkdir()
    runtime_file = output / 'server.json'
    runtime_file.write_text(json.dumps({'dockerfile_sha256': 'original-image-recipe'}))
    with pytest.raises(RuntimeError, match='successful predictions saved'):
        client.run(manifest, output, '/models/tiny', 'http://example.invalid/v1', concurrency=1)
    inference.reset_mock(side_effect=True)
    inference.return_value = prediction('second')
    client.run(manifest, output, '/models/tiny', 'http://example.invalid/v1', concurrency=1)
    assert inference.call_count == 1
    assert inference.call_args.args[0]['id'] == 'second'
    before = (output / 'predictions.jsonl').read_bytes()
    assert len(before.splitlines()) == 2
    runtime_file.write_text(json.dumps({'dockerfile_sha256': 'different-image-recipe'}))
    with pytest.raises(ValueError, match='Run identity changed'):
        client.run(manifest, output, '/models/tiny', 'http://example.invalid/v1', concurrency=1)
    assert (output / 'predictions.jsonl').read_bytes() == before
    assert inference.call_count == 1


def test_interruption_returns_control_before_blocked_requests_finish():
    from threading import Event
    from client import request_pool

    entered, release = Event(), Event()
    def blocked_request():
        entered.set()
        release.wait(5)
    future = None
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            with request_pool(1) as pool:
                future = pool.submit(blocked_request)
                assert entered.wait(2)
                raise KeyboardInterrupt()
        # The suite can now stop the inference server, unblocking its requests.
        assert time.monotonic() - started < 3
        assert not future.done()
    finally:
        release.set()
        if future:
            future.result(timeout=2)
