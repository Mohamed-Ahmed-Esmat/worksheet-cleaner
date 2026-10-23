"""Tests for the re-typeset engine. Gemini calls are always mocked."""

import json
import os
from unittest import mock

import pypdfium2 as pdfium
import pytest
import requests

import gemini_client
import retypeset as rt

ROBOTICS, FLASH, GEMMA = rt.RETYPE_CHAIN

PAGE = (
    "<h1>Grammatik</h1>"
    "<p>1-Das ist ........ Gitarre .</p>"
    "<p>1-Das ist ( der - das - die ) Junge .</p>"
    "<p>4- cool- süß- Junge- Interessant.</p>"
)


def reply(html: str = PAGE, handwriting=None) -> str:
    return json.dumps({"handwriting": handwriting or [], "html": html}, ensure_ascii=False)


def ok(text: str) -> mock.Mock:
    resp = mock.Mock(status_code=200, text=text)
    resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return resp


def error(status: int, message: str) -> mock.Mock:
    resp = mock.Mock(status_code=status, text=message)
    resp.json.return_value = {"error": {"message": message}}
    return resp


class FakeGemini:
    """Routes generateContent calls by model id and records them."""

    def __init__(self, **responses):
        # keyword = model id with '-' and '.' replaced by '_'
        self.responses = {k: v if isinstance(v, list) else [v] for k, v in responses.items()}
        self.calls = []

    def __call__(self, url, headers, json, timeout):
        model = url.split("/models/")[1].split(":")[0]
        self.calls.append((model, json))
        queue = self.responses[model.replace("-", "_").replace(".", "_")]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    @property
    def models_called(self):
        return [m for m, _ in self.calls]


def key(model: str) -> str:
    return model.replace("-", "_").replace(".", "_")


@pytest.fixture
def fake():
    def install(**responses):
        f = FakeGemini(**{key(m): r for m, r in responses.items()})
        patches = [
            mock.patch.object(gemini_client.requests, "post", side_effect=f),
            mock.patch.object(gemini_client.time, "sleep"),
        ]
        for p in patches:
            p.start()
        install.patches.extend(patches)
        return f

    install.patches = []
    yield install
    for p in install.patches:
        p.stop()


# ---------------------------------------------------------------------------
# parse_reply / generation_config
# ---------------------------------------------------------------------------
def test_parse_reply_reads_json():
    html, hw = rt.parse_reply(reply(handwriting=[{"handwritten": "die"}]))
    assert html == PAGE and hw == [{"handwritten": "die"}]


def test_parse_reply_tolerates_code_fence_and_trailing_text():
    text = "```json\n" + reply() + "\n```"
    assert rt.parse_reply(text)[0] == PAGE
    assert rt.parse_reply("Here you go: " + reply() + " hope this helps")[0] == PAGE


def test_parse_reply_accepts_plain_html():
    assert rt.parse_reply(PAGE) == (PAGE, [])


def test_parse_reply_rejects_reply_without_html():
    with pytest.raises(ValueError):
        rt.parse_reply('{"handwriting": [], "html": ""}')
    with pytest.raises(ValueError):
        rt.parse_reply("Sorry, I cannot help with that.")


def test_generation_config_has_no_thinking_settings_for_gemma():
    assert "thinkingConfig" not in rt.generation_config(GEMMA)
    assert rt.generation_config(ROBOTICS)["thinkingConfig"] == {"thinkingLevel": "minimal"}
    for model in rt.RETYPE_CHAIN:
        cfg = rt.generation_config(model)
        assert cfg["responseMimeType"] == "application/json" and cfg["temperature"] == 0


# ---------------------------------------------------------------------------
# transcribe_page: model chain and cross-check
# ---------------------------------------------------------------------------
def test_reads_with_first_model_and_checks_with_second(fake):
    misread = PAGE.replace("süß- Junge-", "süß- unge-")
    f = fake(**{ROBOTICS: ok(reply()), FLASH: ok(reply(misread))})

    result = rt.transcribe_page(b"jpeg", " key ")

    assert f.models_called == [ROBOTICS, FLASH]
    assert (result.model, result.verify_model) == (ROBOTICS, FLASH)
    assert result.flagged == [
        rt.LineDiff("4- cool- süß- Junge- Interessant.", "4- cool- süß- unge- Interessant.")
    ]
    headers_key = gemini_client.requests.post.call_args.kwargs["headers"]["x-goog-api-key"]
    assert headers_key == "key"


def test_identical_readings_are_not_flagged_despite_spacing_and_blank_length(fake):
    other = (PAGE.replace("........", "." * 40)
             .replace("1-Das ist (", "1-Das ist(")
             .replace("Interessant.", "Interessant ."))
    fake(**{ROBOTICS: ok(reply()), FLASH: ok(reply(other))})
    assert rt.transcribe_page(b"jpeg", "k").flagged == []


def test_verify_off_makes_a_single_call(fake):
    f = fake(**{ROBOTICS: ok(reply())})
    result = rt.transcribe_page(b"jpeg", "k", verify=False)
    assert f.models_called == [ROBOTICS] and result.verify_model is None


def quota_error(quota_id: str) -> mock.Mock:
    resp = error(429, "Quota exceeded for metric generate_content_free_tier_requests, limit: 20")
    resp.json.return_value["error"]["details"] = [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": quota_id, "quotaValue": "20"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"},
    ]
    return resp


def test_daily_quota_error_falls_back_and_marks_model_exhausted(fake):
    f = fake(**{
        ROBOTICS: quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
        FLASH: ok(reply()),
        GEMMA: ok(reply()),
    })
    result = rt.transcribe_page(b"jpeg", "k")
    assert f.models_called == [ROBOTICS, FLASH, GEMMA]
    assert (result.model, result.verify_model) == (FLASH, GEMMA)
    assert result.exhausted == {ROBOTICS}
    assert any("today's free quota" in n for n in result.notes)


def test_per_minute_quota_error_skips_model_only_for_this_page(fake):
    fake(**{
        ROBOTICS: quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
        FLASH: ok(reply()),
        GEMMA: ok(reply()),
    })
    result = rt.transcribe_page(b"jpeg", "k")
    assert result.model == FLASH and result.exhausted == set()
    assert any("per-minute" in n for n in result.notes)


def test_skipped_models_are_not_called(fake):
    f = fake(**{FLASH: ok(reply()), GEMMA: ok(reply())})
    result = rt.transcribe_page(b"jpeg", "k", skip={ROBOTICS})
    assert ROBOTICS not in f.models_called and result.model == FLASH


def test_overloaded_model_is_retried_then_skipped(fake):
    busy = error(503, "high demand")
    f = fake(**{ROBOTICS: busy, FLASH: ok(reply()), GEMMA: ok(reply())})
    result = rt.transcribe_page(b"jpeg", "k")
    assert f.models_called.count(ROBOTICS) == 3
    assert [c.args[0] for c in gemini_client.time.sleep.call_args_list] == [2, 5]
    assert result.model == FLASH and ROBOTICS not in result.exhausted


def test_unusable_reply_moves_to_next_model(fake):
    f = fake(**{ROBOTICS: ok("I can't read this image."), FLASH: ok(reply()), GEMMA: ok(reply())})
    result = rt.transcribe_page(b"jpeg", "k")
    assert result.model == FLASH and f.models_called[:2] == [ROBOTICS, FLASH]


def test_last_model_reads_without_cross_check(fake):
    fake(**{GEMMA: ok(reply())})
    result = rt.transcribe_page(b"jpeg", "k", skip={ROBOTICS, FLASH})
    assert result.model == GEMMA and result.verify_model is None
    assert any("not double-checked" in n for n in result.notes)


def test_failed_cross_check_still_returns_the_reading(fake):
    fake(**{ROBOTICS: ok(reply()), FLASH: error(500, "boom"), GEMMA: error(500, "boom")})
    result = rt.transcribe_page(b"jpeg", "k")
    assert result.model == ROBOTICS and result.verify_model is None
    assert any("not double-checked" in n for n in result.notes)


def test_rejected_api_key_stops_immediately(fake):
    f = fake(**{ROBOTICS: error(400, "API key not valid. Please pass a valid API key.")})
    with pytest.raises(rt.RetypeError, match="key was rejected"):
        rt.transcribe_page(b"jpeg", "bad")
    assert f.models_called == [ROBOTICS]


def test_all_models_failing_raises(fake):
    fake(**{m: error(500, "boom") for m in rt.RETYPE_CHAIN})
    with pytest.raises(rt.RetypeError, match="No model could read the page"):
        rt.transcribe_page(b"jpeg", "k")


def test_failure_still_reports_models_whose_daily_quota_ran_out(fake):
    daily = quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    fake(**{ROBOTICS: daily, FLASH: daily, GEMMA: error(503, "busy")})
    with pytest.raises(rt.RetypeError) as info:
        rt.transcribe_page(b"jpeg", "k")
    assert info.value.exhausted == {ROBOTICS, FLASH}


def test_reply_with_no_printed_text_after_sanitizing_moves_to_next_model(fake):
    empty = ok(reply("<script>alert(1)</script><p>  </p>"))
    fake(**{ROBOTICS: empty, FLASH: ok(reply()), GEMMA: ok(reply())})
    result = rt.transcribe_page(b"jpeg", "k")
    assert result.model == FLASH and any("no printed text" in n for n in result.notes)


def test_everything_skipped_raises_without_calls(fake):
    f = fake()
    with pytest.raises(rt.RetypeError, match="quota"):
        rt.transcribe_page(b"jpeg", "k", skip=set(rt.RETYPE_CHAIN))
    assert f.calls == []


def test_gemma_request_omits_thinking_config(fake):
    f = fake(**{GEMMA: ok(reply())})
    rt.transcribe_page(b"jpeg", "k", skip={ROBOTICS, FLASH})
    payload = f.calls[0][1]
    assert "thinkingConfig" not in payload["generationConfig"]
    assert payload["contents"][0]["parts"][1]["text"] == rt.RETYPE_PROMPT


# ---------------------------------------------------------------------------
# compare_readings
# ---------------------------------------------------------------------------
def test_compare_flags_case_changes_and_missing_lines():
    a = "<p>Spiele mit !</p><p>2-Ich ........ Emil .</p><table><tr><td>ich</td><td></td></tr></table>"
    b = "<p>spiele mit !</p><table><tr><td>ich</td><td></td></tr></table>"
    assert rt.compare_readings(a, b) == [
        rt.LineDiff("Spiele mit !", "spiele mit !"),
        rt.LineDiff("2-Ich ........ Emil .", ""),
    ]


def test_text_lines_splits_blocks_and_table_rows():
    html = "<h2>Titel</h2><p>a<br>b</p><table><tr><th>x</th><th>y</th></tr></table>"
    assert rt.text_lines(html) == ["Titel", "a", "b", "x | y"]


# ---------------------------------------------------------------------------
# sanitize_html
# ---------------------------------------------------------------------------
def test_sanitize_removes_scripts_styles_links_and_attributes():
    raw = (
        '<p style="color:red" onclick="x()">Hallo<script>alert(1)</script></p>'
        '<style>p{}</style><a href="http://evil.example">Link</a>'
        '<table border="1"><tr><td colspan="2" style="x">A</td><td colspan="x">B</td></tr></table>'
        '<p class="num">5</p><p class="evil">6</p>'
    )
    html, figures = rt.sanitize_html(raw)
    assert html == (
        "<p>Hallo</p>Link<table><tr><td colspan=\"2\">A</td><td>B</td></tr></table>"
        '<p class="num">5</p><p>6</p>'
    )
    assert figures == []


def test_sanitize_replaces_pictures_and_records_their_boxes():
    html, figures = rt.sanitize_html(
        '<img src="https://x.example/a.png" data-box="10, 20,300,400"><img data-box="bad">'
    )
    assert html.count("[picture]") == 2 and "src" not in html
    assert figures == ["10,20,300,400", ""]


def test_sanitize_escapes_text_and_widens_blanks():
    html, _ = rt.sanitize_html("<p>1 &lt; 2 und ich ... Saxofon … gut. Ende.</p>")
    assert "1 &lt; 2" in html
    assert "ich " + "." * rt.MIN_BLANK_DOTS + " Saxofon " + "." * rt.MIN_BLANK_DOTS + " gut." in html
    assert "Ende." in html  # a single full stop is not a blank


# ---------------------------------------------------------------------------
# render_pdf / render_preview
# ---------------------------------------------------------------------------
def test_render_pdf_puts_each_worksheet_on_its_own_page_with_umlauts():
    pdf = rt.render_pdf([
        rt.sanitize_html("<h1>Grammatik</h1><p>5-Das ist ( der - das - die ) Fußball .</p>")[0],
        rt.sanitize_html("<p>2- Monopoly- Tennis- Fußball- Mädchen</p>")[0],
    ])
    assert pdf.startswith(b"%PDF")
    doc = pdfium.PdfDocument(pdf)
    try:
        assert len(doc) == 2
        first = doc[0].get_textpage().get_text_bounded()
        second = doc[1].get_textpage().get_text_bounded()
    finally:
        doc.close()
    assert "Fußball" in first and "Mädchen" in second


def test_render_preview_returns_png_per_page():
    pngs = rt.render_preview(rt.render_pdf(["<p>Hallo</p>"]), scale=0.5)
    assert len(pngs) == 1 and pngs[0].startswith(b"\x89PNG")


def test_renderer_cannot_read_local_files_or_the_network():
    # Bypass the sanitizer on purpose: the resource policy is the last line of defence.
    photo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "table_crop.png"))
    plain = rt.render_pdf(["<p>x</p>"])
    with mock.patch("socket.create_connection", side_effect=AssertionError("network used")):
        with_image = rt.render_pdf([f'<p>x</p><img src="{photo}"><img src="http://example.com/a.png">'])
    assert abs(len(with_image) - len(plain)) < 500  # the 47 KB photo was not embedded


def test_render_pdf_wraps_renderer_failures_in_runtime_error():
    from xhtml2pdf import pisa

    with mock.patch.object(pisa, "CreatePDF", side_effect=ValueError("bad table")):
        with pytest.raises(RuntimeError, match="bad table"):
            rt.render_pdf(["<p>x</p>"])
    with mock.patch.object(pisa, "CreatePDF", return_value=mock.Mock(err=2)):
        with pytest.raises(RuntimeError, match="2 error"):
            rt.render_pdf(["<p>x</p>"])


def test_unclosed_table_on_one_page_does_not_swallow_the_next_pages():
    pages = [
        rt.sanitize_html("<h1>Blatt 1</h1><table><tr><td>Name</td><td>Alter")[0],
        rt.sanitize_html("<h1>Blatt 2</h1><p>P2</p>")[0],
        rt.sanitize_html("<h1>Blatt 3</h1><p>P3")[0],
    ]
    doc = pdfium.PdfDocument(rt.render_pdf(pages))
    try:
        texts = [doc[i].get_textpage().get_text_bounded() for i in range(len(doc))]
    finally:
        doc.close()
    assert len(texts) == 3
    assert "Alter" in texts[0] and "P2" in texts[1] and "P3" in texts[2]


# ---------------------------------------------------------------------------
# Review follow-ups: sanitizer robustness
# ---------------------------------------------------------------------------
def test_sanitize_closes_open_tags_and_ignores_stray_end_tags():
    html, _ = rt.sanitize_html("<p>a<b>b</p></td><table><tr><td>x<td>y<tr><td>z")
    assert html == (
        "<p>a<b>b</b></p><table><tr><td>x</td><td>y</td></tr><tr><td>z</td></tr></table>"
    )


def test_self_closing_dropped_tags_do_not_hide_the_rest_of_the_page():
    html, figures = rt.sanitize_html("<p>vor</p><script/><svg/><iframe/><p>nach</p>")
    assert "nach" in html and html.count("[picture]") == 1 and figures == [""]
    html, _ = rt.sanitize_html("<p>a</p><svg><text>Logo</text></svg><p>b</p>")
    assert "Logo" not in html and "<p>b</p>" in html


def test_malformed_markup_is_reported_as_value_error():
    with pytest.raises(ValueError, match="malformed HTML"):
        rt.sanitize_html("<p>x</p><![foo[ bar")


def test_malformed_markup_moves_to_next_model(fake):
    fake(**{ROBOTICS: ok(reply("<p>x</p><![foo[ bar")), FLASH: ok(reply()), GEMMA: ok(reply())})
    assert rt.transcribe_page(b"jpeg", "k").model == FLASH


def test_symbols_missing_from_pdf_fonts_get_printable_fallbacks():
    html, _ = rt.sanitize_html("<p>☐ richtig □ falsch ○ ja ◯ nein ☑ ☒ GROẞ</p>")
    assert html == "<p>[   ] richtig [   ] falsch ( ) ja ( ) nein [✓] [✗] GROSS</p>"


def test_oversized_spans_are_dropped():
    html, _ = rt.sanitize_html('<table><tr><td colspan="999">a</td><td rowspan="²">b</td></tr></table>')
    assert "colspan" not in html and "rowspan" not in html


# ---------------------------------------------------------------------------
# Review follow-ups: parsing and comparison
# ---------------------------------------------------------------------------
def test_parse_reply_tolerates_raw_newlines_lists_and_braces_in_prose():
    raw = '{"handwriting": [], "html": "<p>a</p>\n<p>b\tc</p>"}'  # raw newline/tab inside a JSON string
    assert rt.parse_reply(raw)[0] == "<p>a</p>\n<p>b\tc</p>"
    assert rt.parse_reply("[" + reply() + "]")[0] == PAGE
    assert rt.parse_reply("Note {see below}: " + reply())[0] == PAGE


def test_comparable_ignores_spacing_and_blank_length_but_not_letters():
    same = rt._comparable("1-Das ist ...................... Gitarre .")
    assert same == rt._comparable("1-Das  ist … Gitarre.")
    assert rt._comparable("Junge") != rt._comparable("junge")


def test_compare_pairs_misread_line_with_its_counterpart_despite_an_extra_line():
    a = "<p>Titel</p><p>4- cool- süß- Junge- Interessant.</p><p>Ende</p>"
    b = "<p>Titel</p><p>Neue Zeile</p><p>4- cool- süß- unge- Interessant.</p><p>Ende</p>"
    assert rt.compare_readings(a, b) == [
        rt.LineDiff("", "Neue Zeile"),
        rt.LineDiff("4- cool- süß- Junge- Interessant.", "4- cool- süß- unge- Interessant."),
    ]


def test_compare_reports_line_only_in_second_reading():
    assert rt.compare_readings("<p>A</p>", "<p>A</p><p>B</p>") == [rt.LineDiff("", "B")]


# ---------------------------------------------------------------------------
# Review follow-ups: model chain
# ---------------------------------------------------------------------------
def test_cross_check_falls_through_to_gemma_when_flash_is_down(fake):
    f = fake(**{ROBOTICS: ok(reply()), FLASH: error(500, "boom"), GEMMA: ok(reply())})
    result = rt.transcribe_page(b"jpeg", "k")
    assert f.models_called == [ROBOTICS, FLASH, FLASH, FLASH, GEMMA]
    assert result.verify_model == GEMMA


def test_daily_quota_hit_during_cross_check_is_remembered(fake):
    fake(**{
        ROBOTICS: ok(reply()),
        FLASH: quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
        GEMMA: ok(reply()),
    })
    result = rt.transcribe_page(b"jpeg", "k")
    assert result.exhausted == {FLASH} and result.verify_model == GEMMA


def test_retired_model_404_falls_back(fake):
    fake(**{ROBOTICS: error(404, "models/x is not found"), FLASH: ok(reply()), GEMMA: ok(reply())})
    result = rt.transcribe_page(b"jpeg", "k")
    assert result.model == FLASH and any("404" in n for n in result.notes)


def test_permission_denied_403_stops_immediately(fake):
    f = fake(**{ROBOTICS: error(403, "Permission denied")})
    with pytest.raises(rt.RetypeError, match="key was rejected"):
        rt.transcribe_page(b"jpeg", "k")
    assert f.models_called == [ROBOTICS]


def test_last_model_note_does_not_promise_another_model(fake):
    daily = quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    fake(**{GEMMA: daily})
    with pytest.raises(rt.RetypeError) as info:
        rt.transcribe_page(b"jpeg", "k", skip={ROBOTICS, FLASH})
    assert "today's free quota is used up." in str(info.value)
    assert "next model" not in str(info.value)


def test_daily_quota_detected_from_message_without_details(fake):
    fake(**{ROBOTICS: error(429, "Quota exceeded: requests per day"), FLASH: ok(reply()), GEMMA: ok(reply())})
    assert rt.transcribe_page(b"jpeg", "k").exhausted == {ROBOTICS}


# ---------------------------------------------------------------------------
# Review follow-ups: gemini_client
# ---------------------------------------------------------------------------
def test_connection_errors_are_retried_then_reported_without_status():
    with mock.patch.object(gemini_client.requests, "post", side_effect=requests.ConnectionError("down")) as post, \
            mock.patch.object(gemini_client.time, "sleep") as sleep:
        with pytest.raises(gemini_client.GeminiError) as info:
            gemini_client.call_gemini([], "AIzaKey", "m")
    assert post.call_count == 3 and [c.args[0] for c in sleep.call_args_list] == [2, 5]
    assert info.value.status is None and info.value.is_unavailable


def test_read_timeout_is_not_retried():
    with mock.patch.object(gemini_client.requests, "post", side_effect=requests.ReadTimeout("slow")) as post:
        with pytest.raises(gemini_client.GeminiError):
            gemini_client.call_gemini([], "AIzaKey", "m")
    assert post.call_count == 1


def test_network_failure_moves_to_next_model(fake):
    calls = []

    def post(url, headers, json, timeout):
        calls.append(url)
        if ROBOTICS in url:
            raise requests.ReadTimeout("stalled")
        return ok(reply())

    with mock.patch.object(gemini_client.requests, "post", side_effect=post):
        result = rt.transcribe_page(b"jpeg", "k")
    assert result.model == FLASH and len(calls) == 3


def test_api_key_is_cleaned_and_invalid_keys_never_sent():
    assert gemini_client.clean_api_key(" AIza_x-1\u200b\n") == "AIza_x-1"
    with mock.patch.object(gemini_client.requests, "post") as post:
        with pytest.raises(gemini_client.GeminiError) as info:
            gemini_client.call_gemini([], "AIza\u00e9x", "m")
    assert info.value.status == 401 and not post.called
    assert "\u00e9" not in str(info.value)


def test_api_key_is_redacted_from_network_errors():
    err = requests.ConnectionError("failed for header AIzaSecret123")
    with mock.patch.object(gemini_client.requests, "post", side_effect=err), \
            mock.patch.object(gemini_client.time, "sleep"):
        with pytest.raises(gemini_client.GeminiError) as info:
            gemini_client.call_gemini([], "AIzaSecret123", "m")
    assert "AIzaSecret123" not in str(info.value) and "***" in str(info.value)


def test_nested_table_rows_stay_inside_their_own_table():
    html, _ = rt.sanitize_html(
        "<table><tr><td>a<table><tr><td>x</td></tr><tr><td>y</td></tr></table></td>"
        "<td>b</td></tr><tr><td>c</td></tr></table>"
    )
    assert html == (
        "<table><tr><td>a<table><tr><td>x</td></tr><tr><td>y</td></tr></table></td>"
        "<td>b</td></tr><tr><td>c</td></tr></table>"
    )
    # a new row still ends the open row of the same table
    assert rt.sanitize_html("<table><tr><td>a<tr><td>b</table>")[0] == (
        "<table><tr><td>a</td></tr><tr><td>b</td></tr></table>"
    )
