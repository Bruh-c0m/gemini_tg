import unittest

from main import parse_allowed_user_ids, split_message


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

    def test_allowed_user_ids(self) -> None:
        self.assertEqual(parse_allowed_user_ids("10, 20,30"), {10, 20, 30})
        self.assertEqual(parse_allowed_user_ids(""), set())

    def test_invalid_user_id_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_allowed_user_ids("10,not-a-number")


if __name__ == "__main__":
    unittest.main()
