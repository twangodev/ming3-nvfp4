import json

import pytest

from client import payload, read_journal, request, run, select_rows
from score import chinese_normalize, score, score_asr
from upstream import english_normalizer, mmlu, mmlu_prompt, module


def test_raw_answers_reach_upstream_scoring(capsys):
    for source in ("MMAU/evaluation.py", "MMAR/code/evaluation.py"):
        scorer = module("test_mc", source)
        assert scorer.string_match("Rain", "Rain", ["Birds", "Rain"])
        assert not scorer.string_match("Rain", "B", ["Birds", "Rain"])
        assert not scorer.string_match("Rain", "Rain and birds", ["Birds", "Rain"])
    scorer = module("test_mmsu", "MMSU/mmsu_evaluation.py")
    examples = [{"response": text, "answer_gt": "Rain", "choice_a": "Rain"}
                for text in ("A", "not an answer", "")]
    _, _, accuracy, denominator = scorer.calculate_accuracy_per_task_and_category(examples)
    assert (accuracy, denominator) == (0.5, 2)  # Upstream excludes malformed text.


def test_upstream_mmlu_prompt_and_parser():
    official = mmlu()
    example = {"question_id": 1, "category": "math", "question": "2+2?", "options": ["4", "5"]}
    demos = {"math": [{"question": f"example {i}", "options": ["x", "y"],
                       "cot_content": "A: Work. The answer is (A)."} for i in range(5)]}
    prompt = mmlu_prompt(official, example, demos)
    assert prompt.count("Question:") == 6
    assert prompt.endswith("Question: 2+2?\nOptions: A. 4\nB. 5\nAnswer: Let's think step by step.\n\n")
    assert official.extract_answer("Reasoning. The answer is (B)") == "B"
    assert official.extract_answer("no answer") is None


def test_asr_normalization_and_corpus_weighting():
    normalizer = english_normalizer()
    assert normalizer("fifty apples") == normalizer("50 apples")
    assert chinese_normalize("五 十。") == "五十"
    assert chinese_normalize("50") != chinese_normalize("五十")
    assert chinese_normalize("中文 (English) 50") == "中文english50"
    rows = [{"id": "1", "reference": "the red cat"}, {"id": "2", "reference": "dog"}]
    pred = {"1": {"text": "English\tthe red cat"}, "2": {"text": "English\tcat"}}
    result, _ = score_asr(rows, pred, normalizer)
    assert result["score_percent"] == 25  # Corpus counts, not mean per-utterance WER.


def test_unicode_jsonl_and_journal_recovery(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"id": str(i), "benchmark": "test", "prompt": "a\u2028b"} for i in range(3)]
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert select_rows(manifest, 2) == rows[:2]
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b'{"id":"1","text":"ok"}\n{"id":"2","text":')
    with pytest.raises(ValueError, match="Incomplete"):
        read_journal(path)
    assert set(read_journal(path, repair=True)) == {"1"}
    assert path.read_bytes().endswith(b"\n")
    with path.open("a") as stream:
        stream.write('{"id":"1","text":"duplicate"}\n')
    with pytest.raises(ValueError, match="Duplicate"):
        read_journal(path)


def test_stage_sampling_and_incomplete_stream(monkeypatch):
    row = {"id": "1", "benchmark": "mmlu_pro", "prompt": "hello", "audio": None,
           "max_tokens": 4000, "ngram": 0}
    body = payload(row, "test-model")
    assert body["max_tokens"] == body["sampling_params_list"][0]["max_tokens"] == 4000
    assert body["sampling_params_list"][0]["extra_args"]["no_repeat_ngram_size"] == 0

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_lines(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}'

    monkeypatch.setattr("client.requests.post", lambda *a, **kw: Response())
    with pytest.raises(RuntimeError, match="finish reason"):
        request(row, "test-model", "http://example.invalid/v1")


def test_incomplete_run_cannot_be_scored(tmp_path):
    from prepare import digest

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"missing","benchmark":"mmau"}\n')
    (tmp_path / "run.json").write_text(json.dumps({"manifest": str(manifest),
        "manifest_sha256": digest(manifest), "limit_per_benchmark": 0}))
    (tmp_path / "predictions.jsonl").write_text("")
    with pytest.raises(ValueError, match="every selected example"):
        score(tmp_path)


def test_subset_cannot_be_mislabeled_as_full_suite(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"one","benchmark":"mmau"}\n')
    with pytest.raises(ValueError, match="full pinned suite"):
        run(manifest, tmp_path / "run", "/missing-model", "http://example.invalid/v1")
