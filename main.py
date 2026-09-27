import asyncio
import io
import logging
import os
from collections.abc import Iterable

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from google import genai
from google.genai import errors as genai_errors
from google.genai import types


# Запас относительно системного лимита Android (1024 UTF-16 единицы), чтобы
# отдельная часть ответа лучше помещалась в уведомление Telegram.
TELEGRAM_MESSAGE_LIMIT = 900
DEFAULT_MODEL = "gemini-3.8-flash"
NEXT_WORDS = {"дальше", "далее", "продолжить", "next"}
GEMINI_RETRY_DELAYS = (2, 4)

SYSTEM_PROMPT = """
Ты — эксперт по теоретической физике и квантовой механике. Дай краткое, но
математически полное решение. Если условие на фотографии неразборчиво, не
угадывай: кратко перечисли, что нужно уточнить.

Требования к решению:
1. Не переписывай условие и список «дано», если без этого понятны обозначения.
2. Сразу определи физическую модель и запиши исходное уравнение.
3. Покажи только преобразования, необходимые для получения ответа. Не выводи
   заново общеизвестные формулы, но укажи граничные условия и нормировку, если
   от них зависит результат.
4. Не добавляй исторические справки, длинный физический комментарий, формулу
   Родрига, отдельный анализ размерностей или предельных случаев, если в них
   нет необходимости для проверки именно этой задачи.
5. Заверши блоком «Ответ» с искомыми формулами и диапазоном квантовых чисел.
6. Не повторяй в конце формулы, уже явно выделенные как окончательный ответ.
7. Целевой объём всего решения — 900–1500 символов. Превышай его только тогда,
   когда более короткое изложение потеряет существенный шаг или станет неверным.

Формат для Telegram и часов:
- Пиши по-русски, обычным текстом, короткими абзацами.
- Не используй Markdown-таблицы, кодовые блоки и сырой LaTeX вроде \\frac,
  \\sqrt, $$ или \\psi.
- Используй читаемые Unicode-символы: ψ, Ψ, φ, ℏ, ∂, ∫, Σ, ∇, √, ∞, ±, ×,
  ⟨ψ|, |ψ⟩, â, â†.
- Формулы записывай линейно и каждую важную формулу помещай на отдельную строку.
- Дроби и степени пиши понятно, например:
  (-ℏ² / 2m) · d²ψ/dx² + V(x)ψ = Eψ.
- Используй короткие заголовки: «Решение», «Ответ». Не создавай многочисленные
  нумерованные разделы.
""".strip()


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Не задана обязательная переменная окружения {name}")
    return value


def parse_allowed_user_ids(raw: str) -> set[int]:
    if not raw.strip():
        return set()

    result: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError as exc:
            raise RuntimeError(
                "ALLOWED_USER_IDS должен содержать Telegram ID через запятую"
            ) from exc
    return result


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split text without losing characters, preferring paragraph boundaries."""
    remainder = text.strip()
    if not remainder:
        return ["Модель не вернула текстовый ответ."]

    chunks: list[str] = []
    while len(remainder) > limit:
        candidates = (
            remainder.rfind("\n\n", 0, limit + 1),
            remainder.rfind("\n", 0, limit + 1),
            remainder.rfind(". ", 0, limit + 1),
            remainder.rfind(" ", 0, limit + 1),
        )
        cut = max(candidates)
        if cut < limit // 2:
            cut = limit
        elif remainder[cut : cut + 2] == ". ":
            cut += 1

        chunk = remainder[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remainder = remainder[cut:].lstrip()

    if remainder:
        chunks.append(remainder)
    return chunks


def is_next_request(text: str) -> bool:
    normalized = text.strip().casefold()
    return (
        normalized in NEXT_WORDS
        or normalized == "/next"
        or normalized.startswith("/next@")
    )


def format_answer_part(chunk: str, index: int, total: int) -> str:
    if total == 1:
        return chunk

    text = f"Часть {index + 1} из {total}\n\n{chunk}"
    if index + 1 < total:
        text += "\n\nЧтобы получить продолжение, напишите: дальше"
    return text


def make_router(
    ai_client: genai.Client,
    model_name: str,
    allowed_user_ids: set[int],
    request_gate: asyncio.Semaphore,
) -> Router:
    router = Router()
    pending_answers: dict[int, tuple[list[str], int]] = {}

    async def authorize(message: Message) -> bool:
        user_id = message.from_user.id if message.from_user else None
        if allowed_user_ids and user_id not in allowed_user_ids:
            await message.answer("У этого аккаунта нет доступа к боту.")
            return False
        return True

    async def ask_gemini(contents: Iterable[types.Part]) -> str:
        parts = list(contents)
        async with request_gate:
            for attempt in range(len(GEMINI_RETRY_DELAYS) + 1):
                try:
                    response = await ai_client.aio.models.generate_content(
                        model=model_name,
                        contents=parts,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            automatic_function_calling=(
                                types.AutomaticFunctionCallingConfig(disable=True)
                            ),
                        ),
                    )
                    break
                except genai_errors.ServerError as exc:
                    if exc.code != 503 or attempt >= len(GEMINI_RETRY_DELAYS):
                        raise
                    delay = GEMINI_RETRY_DELAYS[attempt]
                    logging.warning(
                        "Gemini временно недоступен (503), повтор через %s с",
                        delay,
                    )
                    await asyncio.sleep(delay)
        return (response.text or "").strip()

    async def send_first_part(message: Message, answer: str) -> None:
        chunks = split_message(answer)
        user_id = message.from_user.id if message.from_user else None

        if user_id is not None and len(chunks) > 1:
            pending_answers[user_id] = (chunks, 1)
        elif user_id is not None:
            pending_answers.pop(user_id, None)

        await message.answer(format_answer_part(chunks[0], 0, len(chunks)))

    async def send_next_part(message: Message) -> None:
        if not await authorize(message):
            return

        user_id = message.from_user.id if message.from_user else None
        if user_id is None or user_id not in pending_answers:
            await message.answer(
                "Продолжения пока нет. Сначала отправьте новую задачу."
            )
            return

        chunks, index = pending_answers[user_id]
        await message.answer(format_answer_part(chunks[index], index, len(chunks)))

        if index + 1 < len(chunks):
            pending_answers[user_id] = (chunks, index + 1)
        else:
            pending_answers.pop(user_id, None)

    async def process_request(message: Message, contents: Iterable[types.Part]) -> None:
        if not await authorize(message):
            return

        status = await message.answer("Решаю задачу...")
        try:
            answer = await ask_gemini(contents)
            try:
                await status.delete()
            except Exception:
                logging.warning("Не удалось удалить служебное сообщение", exc_info=True)
            await send_first_part(message, answer)
        except genai_errors.ServerError as exc:
            logging.exception(
                "Gemini не обработал запрос пользователя %s",
                message.from_user.id if message.from_user else "unknown",
            )
            error_text = (
                "Gemini сейчас перегружен. Бот уже повторил запрос несколько раз. "
                "Попробуйте отправить задачу ещё раз через минуту."
                if exc.code == 503
                else "Сервис Gemini временно недоступен. Повторите запрос позже."
            )
            try:
                await status.edit_text(error_text)
            except Exception:
                logging.exception("Не удалось обновить служебное сообщение")
        except Exception:
            logging.exception(
                "Не удалось обработать запрос пользователя %s",
                message.from_user.id if message.from_user else "unknown",
            )
            try:
                await status.edit_text(
                    "Не удалось получить решение. Проверьте ключ Gemini, доступ к "
                    "модели и журнал Railway, затем повторите запрос."
                )
            except Exception:
                logging.exception("Не удалось обновить служебное сообщение")

    @router.message(CommandStart())
    async def start_handler(message: Message) -> None:
        if not await authorize(message):
            return
        await message.answer(
            "Пришлите фотографию задачи или её текст. Для снимка можно добавить "
            "подпись с уточнением, что именно требуется найти.\n\n"
            "Длинный ответ приходит частями. Для следующей части напишите "
            "«дальше» или отправьте /next.\n\n"
            "Команда /id покажет ваш Telegram ID для ограничения доступа."
        )

    @router.message(Command("id"))
    async def id_handler(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else "неизвестен"
        await message.answer(f"Ваш Telegram ID: {user_id}")

    @router.message(Command("next"))
    async def next_handler(message: Message) -> None:
        await send_next_part(message)

    @router.message(F.photo)
    async def photo_handler(message: Message, bot: Bot) -> None:
        photo = message.photo[-1]
        telegram_file = await bot.get_file(photo.file_id)
        if not telegram_file.file_path:
            await message.answer("Telegram не вернул путь к фотографии.")
            return

        buffer = io.BytesIO()
        await bot.download_file(telegram_file.file_path, destination=buffer)
        prompt = message.caption or (
            "Распознай условие на фотографии и дай краткое, но полное решение."
        )
        await process_request(
            message,
            [
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"),
            ],
        )

    @router.message(F.document)
    async def image_document_handler(message: Message, bot: Bot) -> None:
        document = message.document
        mime_type = (document.mime_type or "").lower()
        if not mime_type.startswith("image/"):
            await message.answer(
                "Поддерживаются текст, обычная фотография или изображение, "
                "отправленное как файл."
            )
            return

        telegram_file = await bot.get_file(document.file_id)
        if not telegram_file.file_path:
            await message.answer("Telegram не вернул путь к изображению.")
            return

        buffer = io.BytesIO()
        await bot.download_file(telegram_file.file_path, destination=buffer)
        prompt = message.caption or (
            "Распознай условие на изображении и дай краткое, но полное решение."
        )
        await process_request(
            message,
            [
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=buffer.getvalue(), mime_type=mime_type),
            ],
        )

    @router.message(F.text)
    async def text_handler(message: Message) -> None:
        text = message.text or ""
        if is_next_request(text):
            await send_next_part(message)
            return
        if text.startswith("/"):
            return
        await process_request(
            message,
            [types.Part.from_text(text=text)],
        )

    @router.message()
    async def unsupported_handler(message: Message) -> None:
        if await authorize(message):
            await message.answer(
                "Пришлите текст, фотографию или изображение, отправленное как файл."
            )

    return router


async def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    telegram_token = required_env("TELEGRAM_TOKEN")
    gemini_api_key = required_env("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    allowed_user_ids = parse_allowed_user_ids(os.getenv("ALLOWED_USER_IDS", ""))
    max_parallel = max(1, int(os.getenv("MAX_PARALLEL_REQUESTS", "2")))

    if not allowed_user_ids:
        logging.warning(
            "ALLOWED_USER_IDS не задан: бот будет отвечать всем пользователям"
        )

    bot = Bot(token=telegram_token)
    dispatcher = Dispatcher()
    ai_client = genai.Client(api_key=gemini_api_key)
    dispatcher.include_router(
        make_router(
            ai_client=ai_client,
            model_name=model_name,
            allowed_user_ids=allowed_user_ids,
            request_gate=asyncio.Semaphore(max_parallel),
        )
    )

    logging.info("Запуск бота с моделью %s", model_name)
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await dispatcher.start_polling(bot)
    finally:
        await ai_client.aio.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
