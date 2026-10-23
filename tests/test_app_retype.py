"""App-level tests for the Re-typeset engine, run with Streamlit's AppTest (no API calls)."""

import os

from streamlit.testing.v1 import AppTest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script, **kwargs):
    at = AppTest.from_function(script, kwargs={"repo": REPO, **kwargs}, default_timeout=60)
    return at.run()


def _flagged_page(repo):
    import os
    import sys
    from unittest import mock

    sys.path.insert(0, repo)
    import app
    import retypeset as rt

    photo = open(os.path.join(repo, "table_crop.png"), "rb").read()
    result = rt.RetypeResult(
        html="<p>4- cool- süß- Junge- Interessant.</p>",
        model=rt.RETYPE_CHAIN[0],
        verify_model=rt.RETYPE_CHAIN[1],
        flagged=[rt.LineDiff("junge `x` ![img](http://e.example/a.png)", "Junge")],
        figures=[
            rt.Figure((1, 2, 3, 4), has_handwriting=True, embedded=True),   # 1: marked
            rt.Figure(None),                                                 # 2: not located
            rt.Figure((5, 5, 50, 50), has_handwriting=None, embedded=True),  # 3: unsure
            rt.Figure((5, 5, 50, 50), has_handwriting=True, embedded=False), # 4: boxed, not copied
            rt.Figure((5, 5, 50, 50), has_handwriting=False, embedded=True), # 5: clean
        ],
        notes=["a note"],
        exhausted={rt.RETYPE_CHAIN[0]},
    )
    with mock.patch.object(rt, "transcribe_page", return_value=result):
        pdf = app.retype_one_page(1, "p.png", photo, "AIzaKeyOne", True, True)
    import streamlit as st

    st.text("RETURNED PDF " + str(pdf.startswith(b"%PDF")))
    st.text("SKIP " + ",".join(sorted(app.exhausted_models("AIzaKeyOne"))))
    st.text("OTHER KEY " + ",".join(sorted(app.exhausted_models("AIzaKeyTwo"))))


def test_flagged_lines_are_shown_verbatim_with_warnings():
    at = _run(_flagged_page)
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("disagree on 1 line" in w for w in warnings)
    assert any("picture(s) 1 have the student's marks" in w for w in warnings)
    assert any("check picture(s) 3 for the student's marks" in w for w in warnings)
    assert any("2 picture(s) could not be located" in w for w in warnings)
    assert any("3 picture(s) copied" in i.value for i in at.info)
    table = at.dataframe[0].value
    assert list(table["In the PDF"]) == ["junge `x` ![img](http://e.example/a.png)"]
    texts = [t.value for t in at.text]
    assert "RETURNED PDF True" in texts
    # quota bookkeeping belongs to the key that used it up
    assert any(t.startswith("SKIP gemini-robotics-er-2-preview") for t in texts)
    assert "OTHER KEY" in texts  # a different key starts with every model available


def _unrenderable_page(repo):
    import os
    import sys
    from unittest import mock

    sys.path.insert(0, repo)
    import app
    import retypeset as rt

    photo = open(os.path.join(repo, "table_crop.png"), "rb").read()
    result = rt.RetypeResult(html="<p>x</p>", model=rt.RETYPE_CHAIN[0])
    with mock.patch.object(rt, "transcribe_page", return_value=result), \
            mock.patch.object(rt, "render_pdf", side_effect=RuntimeError("layout failed")):
        pdf = app.retype_one_page(2, "q.png", photo, "AIzaKey", True, True)
    import streamlit as st

    st.text("RETURNED [" + pdf.decode() + "]")


def test_page_that_cannot_be_laid_out_is_left_out_with_an_error():
    at = _run(_unrenderable_page)
    assert not at.exception
    assert any("left out" in e.value for e in at.error)
    assert "RETURNED []" in [t.value for t in at.text]


def _unchecked_page(repo):
    import os
    import sys
    from unittest import mock

    sys.path.insert(0, repo)
    import app
    import retypeset as rt

    photo = open(os.path.join(repo, "table_crop.png"), "rb").read()
    result = rt.RetypeResult(html="<p>x</p>", model=rt.RETYPE_CHAIN[2])  # nobody checked it
    with mock.patch.object(rt, "transcribe_page", return_value=result):
        app.retype_one_page(3, "r.png", photo, "AIzaKey", True, False)
        app.retype_one_page(4, "s.png", photo, "AIzaKey", False, False)  # check switched off


def test_page_read_by_one_model_only_gets_a_warning_when_checking_was_asked_for():
    at = _run(_unchecked_page)
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any(w.startswith("Page 3 (r.png): the text was read by one model only") for w in warnings)
    assert not any(w.startswith("Page 4") for w in warnings)


def test_retypeset_is_the_default_engine_and_all_engines_load():
    at = AppTest.from_file(os.path.join(REPO, "app.py"), default_timeout=60).run()
    assert not at.exception
    engine = at.sidebar.radio[0]
    assert engine.value.startswith("📄 Re-typeset")
    for option in engine.options:
        at.sidebar.radio[0].set_value(option).run()
        assert not at.exception, option
