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
    gemini_retry_delay,
    is_back_request,
    is_next_request,
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
        self.assertEqual(gemini_retry_delay(1), 5)
        self.assertEqual(gemini_retry_delay(2), 10)
        self.assertEqual(gemini_retry_delay(3), 20)
        self.assertEqual(gemini_retry_delay(20), 60)


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
        retry_calls: list[tuple[int, int]] = []

        async def on_retry(attempt: int, delay: int) -> None:
            self.assertFalse(request_gate.locked())
            retry_calls.append((attempt, delay))

        with patch("main.asyncio.sleep", new=AsyncMock()) as sleep:
            answer = await ask_gemini_with_retry(
                ai_client=ai_client,
                model_name="test-model",
                request_gate=request_gate,
                contents=[types.Part.from_text(text="Реши задачу")],
                on_retry=on_retry,
            )

        self.assertEqual(answer, "Готовый ответ")
        self.assertEqual(generate_content.await_count, 3)
        self.assertEqual(retry_calls, [(1, 5), (2, 10)])
        self.assertEqual(sleep.await_args_list, [call(5), call(10)])

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
                model_name="test-model",
                request_gate=asyncio.Semaphore(1),
                contents=[types.Part.from_text(text="test")],
                on_retry=AsyncMock(),
            )

        self.assertEqual(generate_content.await_count, 1)


if __name__ == "__main__":
    unittest.main()
