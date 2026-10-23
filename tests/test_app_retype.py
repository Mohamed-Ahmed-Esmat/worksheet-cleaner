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
        figures=["1,2,3,4"],
        notes=["a note"],
        exhausted={rt.RETYPE_CHAIN[0]},
    )
    with mock.patch.object(rt, "transcribe_page", return_value=result):
        html = app.retype_one_page(1, "p.png", photo, "AIzaKeyOne", True, True)
    import streamlit as st

    st.text("RETURNED " + html)
    st.text("SKIP " + ",".join(sorted(app.exhausted_models("AIzaKeyOne"))))
    st.text("OTHER KEY " + ",".join(sorted(app.exhausted_models("AIzaKeyTwo"))))


def test_flagged_lines_are_shown_verbatim_with_warnings():
    at = _run(_flagged_page)
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("disagree on 1 line" in w for w in warnings)
    assert any("picture" in w for w in warnings)
    table = at.dataframe[0].value
    assert list(table["In the PDF"]) == ["junge `x` ![img](http://e.example/a.png)"]
    texts = [t.value for t in at.text]
    assert "RETURNED <p>4- cool- süß- Junge- Interessant.</p>" in texts
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
        html = app.retype_one_page(2, "q.png", photo, "AIzaKey", True, True)
    import streamlit as st

    st.text("RETURNED [" + html + "]")


def test_page_that_cannot_be_laid_out_is_left_out_with_an_error():
    at = _run(_unrenderable_page)
    assert not at.exception
    assert any("left out" in e.value for e in at.error)
    assert "RETURNED []" in [t.value for t in at.text]


def test_retypeset_is_the_default_engine_and_all_engines_load():
    at = AppTest.from_file(os.path.join(REPO, "app.py"), default_timeout=60).run()
    assert not at.exception
    engine = at.sidebar.radio[0]
    assert engine.value.startswith("📄 Re-typeset")
    for option in engine.options:
        at.sidebar.radio[0].set_value(option).run()
        assert not at.exception, option
