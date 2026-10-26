import pytest

from ming3_nvfp4.calibration_data import bounded_text, choose_audio_questions, verify_official


def test_bounded_text_counts_teacher_forced_reference_and_utf8():
    assert bounded_text("你好", "答案")
    assert not bounded_text("a", "中" * 500)
    assert not bounded_text(" ", "answer")


def test_question_selection_requires_agreement_and_unique_audio():
    rows = []
    for name in ["a.wav", "b.wav"]:
        for question in ["What?", "Why?"]:
            rows.extend(
                dict(file_name=name, QuestionText=question, answer="bird", confidence="yes")
                for _ in range(2)
            )
    rows.append(dict(file_name="c.wav", QuestionText="What?", answer="cat", confidence="maybe"))
    selected = choose_audio_questions(rows, 2, 42)
    assert len({r["file_name"] for r in selected}) == 2
    assert all(r["agreeing_annotations"] == 2 for r in selected)
    assert choose_audio_questions(rows, 2, 42) == selected
    with pytest.raises(ValueError, match="Only 2"):
        choose_audio_questions(rows, 3, 42)


def test_official_source_rejects_wrong_size(tmp_path):
    path = tmp_path / "clotho_aqa_train.csv"
    path.write_text("not training data")
    with pytest.raises(ValueError, match="size"):
        verify_official(path, path.name)


def test_range_extraction_verifies_member_and_never_uses_archive_path(tmp_path, monkeypatch):
    import hashlib
    import io
    import random
    import zipfile
    import requests
    from ming3_nvfp4 import calibration_data

    data = random.Random(7).randbytes(600000)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../unsafe/recording.wav", data)
    raw = stream.getvalue()
    monkeypatch.setitem(calibration_data.OFFICIAL, "audio_files.zip", ("publisher-md5", len(raw)))
    monkeypatch.setattr(calibration_data, "disk_budget", lambda *args: None)

    class Response:
        status_code = 206

        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    def get(url, headers, timeout):
        value = headers["Range"].removeprefix("bytes=")
        if value.startswith("-"):
            return Response(raw[int(value) :])
        start, end = map(int, value.split("-"))
        return Response(raw[start : end + 1])

    monkeypatch.setattr(requests, "get", get)
    report = calibration_data.fetch_clotho_audio(tmp_path, [{"file_name": "recording.wav"}])
    assert not report["whole_archive_checksum_verified"]
    expected = tmp_path / "clotho-train" / (hashlib.sha256(data).hexdigest() + ".wav")
    assert expected.read_bytes() == data
    assert not (tmp_path / "unsafe").exists()
