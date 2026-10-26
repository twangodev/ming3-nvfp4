from choice_scoring import mmsu_official_label, parse_choice


def test_unambiguous_formats():
    choices = ["Rain", "Wind", "Birds", "Traffic"]
    for text, index in [("B", 1), ("(C)", 2), ("**D**.", 3),
                        ("Sounds like wind.\nB", 1), ("The answer is (A).", 0),
                        ("D. Traffic", 3), ("Birds", 2)]:
        assert parse_choice(text, choices)[0] == index


def test_prose_is_not_a_letter():
    choices = ["Rain", "Wind", "Birds", "Traffic"]
    for text in ["Based on the audio", "A bird is singing", "Both are plausible", "Dunno"]:
        assert parse_choice(text, choices)[0] is None
    assert mmsu_official_label("Based on the audio, the answer is C.") == "B"
    assert parse_choice("Based on the audio, the answer is C.", choices)[0] == 2


def test_empty_and_duplicate_options():
    assert parse_choice("C", ["Yes", "No", ""])[0] is None
    assert parse_choice("eight", ["six", "eight", "eight"])[0] == 1


def test_final_answer_beats_reasoning_mentions():
    assert parse_choice("A seems plausible, but B is better.\nB", ["Rain", "Wind"])[0] == 1


def test_explicit_answer_in_prose_and_conflicts():
    choices = ["Rain", "Wind"]
    assert parse_choice("The correct answer is B. Wind can be heard.", choices)[0] == 1
    assert parse_choice("The answer is A. The answer is B. It is ambiguous.", choices)[0] is None
