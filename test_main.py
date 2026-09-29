import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

from google.genai import errors as genai_errors
from google.genai import types

from main import (
    TELEGRAM_MESSAGE_LIMIT,
    ask_gemini_with_retry,
    format_answer_part,
    format_telegram_html,
    gemini_retry_delay,
    is_back_request,
    is_next_request,
    parse_model_names,
    split_message,
)


class SplitMessageTests(unittest.TestCase):
    def test_short_text_is_unchanged(self) -> None:
        self.assertEqual(split_message("Короткий ответ"), ["Короткий ответ"])

    def test_empty_text_has_fallback(self) -> None:
        self.assertEqual(split_message(""), ["Модель не вернула текстовый ответ."])

    def test_long_text_respects_limit_and_preserves_words(self) -> None:
        source = "\n\n".join(f"Абзац {index}: " + "слово " * 20 for index in range(30))
        chunks = split_message(source, limit=180)
        self.assertTrue(all(len(chunk) <= 180 for chunk in chunks))
        self.assertEqual(" ".join(" ".join(chunks).split()), " ".join(source.split()))

    def test_new_task_is_not_left_at_end_of_previous_part(self) -> None:
        task_1 = "ЗАДАЧА 1\n\n" + "Первое решение. " * 5
        task_2 = "ЗАДАЧА 2\n\n" + "Второе решение. " * 5

        chunks = split_message(f"{task_1}\n\n{task_2}", limit=120)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("ЗАДАЧА 1"))
        self.assertNotIn("ЗАДАЧА 2", chunks[0])
        self.assertTrue(chunks[1].startswith("ЗАДАЧА 2"))

    def test_long_task_continuations_repeat_task_heading(self) -> None:
        source = (
            "ЗАДАЧА 3\n\n"
            + "Требуется найти математическое ожидание.\n\n"
            + "РЕШЕНИЕ\n"
            + "Формула и вычисления. " * 12
            + "\n\nОТВЕТ: значение."
        )

        chunks = split_message(source, limit=150)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.startswith("ЗАДАЧА 3") for chunk in chunks))
        self.assertTrue(
            all("ПРОДОЛЖЕНИЕ" in chunk for chunk in chunks[1:])
        )
        self.assertTrue(all(len(chunk) <= 150 for chunk in chunks))

    def test_default_limit_fits_android_notification(self) -> None:
        chunks = split_message("слово " * 1000)
        self.assertEqual(TELEGRAM_MESSAGE_LIMIT, 900)
        self.assertTrue(all(len(chunk) <= 900 for chunk in chunks))

    def test_numbered_part_stays_below_android_limit(self) -> None:
        part = format_answer_part("а" * TELEGRAM_MESSAGE_LIMIT, 0, 12)
        self.assertLessEqual(len(part), 1024)
        self.assertIn("Часть 1 из 12", part)
        self.assertTrue(part.endswith("/next"))

    def test_middle_part_has_both_navigation_commands(self) -> None:
        part = format_answer_part("Текст", 2, 5)
        self.assertTrue(part.endswith("/back · /next"))

    def test_last_part_keeps_back_command(self) -> None:
        part = format_answer_part("Ответ", 4, 5)
        self.assertTrue(part.endswith("/back"))
        self.assertNotIn("/next", part)

    def test_single_part_has_no_navigation_text(self) -> None:
        self.assertEqual(format_answer_part("Готово", 0, 1), "Готово")

    def test_telegram_html_escapes_formulas_and_bolds_headings(self) -> None:
        rendered = format_telegram_html(
            "ЗАДАЧА 3\n\nРЕШЕНИЕ\n0 < x < L\n\nОТВЕТ: ⟨x⟩ = L/2"
        )
        self.assertIn("<b>ЗАДАЧА 3</b>", rendered)
        self.assertIn("<b>РЕШЕНИЕ</b>", rendered)
        self.assertIn("0 &lt; x &lt; L", rendered)
        self.assertIn("<b>ОТВЕТ: ⟨x⟩ = L/2</b>", rendered)
        self.assertEqual(
            format_telegram_html("ЗАДАЧА 3 — ПРОДОЛЖЕНИЕ"),
            "<b>ЗАДАЧА 3 — ПРОДОЛЖЕНИЕ</b>",
        )

    def test_next_request_from_watch_without_command_entity(self) -> None:
        self.assertTrue(is_next_request("/next"))
        self.assertTrue(is_next_request(" ДАЛЬШЕ "))
        self.assertTrue(is_next_request("/next@gemini_kvanti_bot"))
        self.assertFalse(is_next_request("/start"))

    def test_back_request_from_watch_without_command_entity(self) -> None:
        self.assertTrue(is_back_request("/back"))
        self.assertTrue(is_back_request(" НАЗАД "))
        self.assertTrue(is_back_request("/back@gemini_kvanti_bot"))
        self.assertFalse(is_back_request("/start"))

    def test_gemini_retry_delay_grows_and_is_capped(self) -> None:
        self.assertEqual(gemini_retry_delay(1), 3)
        self.assertEqual(gemini_retry_delay(2), 6)
        self.assertEqual(gemini_retry_delay(3), 12)
        self.assertEqual(gemini_retry_delay(20), 30)

    def test_default_models_include_fallback(self) -> None:
        self.assertEqual(
            parse_model_names(""),
            ("gemini-3.5-flash", "gemini-3.1-flash-lite"),
        )

    def test_configured_models_are_deduplicated(self) -> None:
        self.assertEqual(
            parse_model_names("first, second, first"),
            ("first", "second"),
        )


class GeminiRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_same_request_until_success(self) -> None:
        generate_content = AsyncMock(
            side_effect=[
                genai_errors.ServerError(503, {"message": "overloaded"}),
                genai_errors.ServerError(503, {"message": "overloaded"}),
                SimpleNamespace(text="  Готовый ответ  "),
            ]
        )
        ai_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content)
            )
        )
        request_gate = asyncio.Semaphore(1)
        retry_calls: list[tuple[int, int, str, str]] = []

        async def on_retry(
            attempt: int,
            delay: int,
            failed_model: str,
            next_model: str,
        ) -> None:
            self.assertFalse(request_gate.locked())
            retry_calls.append((attempt, delay, failed_model, next_model))

        with patch("main.asyncio.sleep", new=AsyncMock()) as sleep:
            answer = await ask_gemini_with_retry(
                ai_client=ai_client,
                model_names=("primary-model", "fallback-model"),
                request_gate=request_gate,
                contents=[types.Part.from_text(text="Реши задачу")],
                on_retry=on_retry,
            )

        self.assertEqual(answer, "Готовый ответ")
        self.assertEqual(generate_content.await_count, 3)
        self.assertEqual(
            retry_calls,
            [
                (1, 3, "primary-model", "fallback-model"),
                (2, 6, "fallback-model", "primary-model"),
            ],
        )
        self.assertEqual(sleep.await_args_list, [call(3), call(6)])
        used_models = [
            await_call.kwargs["model"]
            for await_call in generate_content.await_args_list
        ]
        self.assertEqual(
            used_models,
            ["primary-model", "fallback-model", "primary-model"],
        )

    async def test_does_not_retry_non_transient_error(self) -> None:
        generate_content = AsyncMock(
            side_effect=genai_errors.ClientError(400, {"message": "bad request"})
        )
        ai_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content)
            )
        )

        with self.assertRaises(genai_errors.ClientError):
            await ask_gemini_with_retry(
                ai_client=ai_client,
                model_names=("test-model",),
                request_gate=asyncio.Semaphore(1),
                contents=[types.Part.from_text(text="test")],
                on_retry=AsyncMock(),
            )

        self.assertEqual(generate_content.await_count, 1)


if __name__ == "__main__":
    unittest.main()
