import unittest

from main import (
    TELEGRAM_MESSAGE_LIMIT,
    format_answer_part,
    is_next_request,
    parse_allowed_user_ids,
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
        self.assertIn("напишите: дальше", part)

    def test_single_part_has_no_navigation_text(self) -> None:
        self.assertEqual(format_answer_part("Готово", 0, 1), "Готово")

    def test_next_request_from_watch_without_command_entity(self) -> None:
        self.assertTrue(is_next_request("/next"))
        self.assertTrue(is_next_request(" ДАЛЬШЕ "))
        self.assertTrue(is_next_request("/next@gemini_kvanti_bot"))
        self.assertFalse(is_next_request("/start"))

    def test_allowed_user_ids(self) -> None:
        self.assertEqual(parse_allowed_user_ids("10, 20,30"), {10, 20, 30})
        self.assertEqual(parse_allowed_user_ids(""), set())

    def test_invalid_user_id_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_allowed_user_ids("10,not-a-number")


if __name__ == "__main__":
    unittest.main()
