import asyncio
import io
import logging
import os
from collections.abc import Iterable

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from google import genai
from google.genai import types


# Запас относительно системного лимита Android (1024 UTF-16 единицы), чтобы
# отдельная часть ответа лучше помещалась в уведомление Telegram.
TELEGRAM_MESSAGE_LIMIT = 900
DEFAULT_MODEL = "gemini-3.8-flash"

SYSTEM_PROMPT = """
Ты — эксперт по теоретической физике и квантовой механике. Решай задачу строго,
последовательно и проверяемо. Если условие на фотографии неразборчиво, не
угадывай: перечисли, какие обозначения или числа нужно уточнить.

Требования к решению:
1. Кратко перепиши дано и укажи искомую величину.
2. Назови физическую модель, допущения, базис, операторы и гамильтониан, когда
   они нужны.
3. Приведи все ключевые уравнения и выведи результат по шагам, не пропуская
   критические преобразования, граничные условия, нормировку и проверку
   размерностей.
4. Отдельно отметь физический смысл результата и выполни проверку предельных
   случаев, если она уместна.
5. Заверши однозначным итоговым ответом.

Формат для Telegram и часов:
- Пиши по-русски, обычным текстом, короткими абзацами.
- Не используй Markdown-таблицы, кодовые блоки и сырой LaTeX вроде \\frac,
  \\sqrt, $$ или \\psi.
- Используй читаемые Unicode-символы: ψ, Ψ, φ, ℏ, ∂, ∫, Σ, ∇, √, ∞, ±, ×,
  ⟨ψ|, |ψ⟩, â, â†.
- Формулы записывай линейно и каждую важную формулу помещай на отдельную строку.
- Дроби и степени пиши понятно, например:
  (-ℏ² / 2m) · d²ψ/dx² + V(x)ψ = Eψ.
- Не сокращай решение только ради длины: длинный ответ будет разбит ботом на
  несколько сообщений.
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


async def send_long_answer(message: Message, answer: str) -> None:
    for chunk in split_message(answer):
        await message.answer(chunk)
        await asyncio.sleep(0.15)


def make_router(
    ai_client: genai.Client,
    model_name: str,
    allowed_user_ids: set[int],
    request_gate: asyncio.Semaphore,
) -> Router:
    router = Router()

    async def authorize(message: Message) -> bool:
        user_id = message.from_user.id if message.from_user else None
        if allowed_user_ids and user_id not in allowed_user_ids:
            await message.answer("У этого аккаунта нет доступа к боту.")
            return False
        return True

    async def ask_gemini(contents: Iterable[types.Part]) -> str:
        async with request_gate:
            response = await ai_client.aio.models.generate_content(
                model=model_name,
                contents=list(contents),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
        return (response.text or "").strip()

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
            await send_long_answer(message, answer)
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
            "Команда /id покажет ваш Telegram ID для ограничения доступа."
        )

    @router.message(Command("id"))
    async def id_handler(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else "неизвестен"
        await message.answer(f"Ваш Telegram ID: {user_id}")

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
            "Распознай условие на фотографии и реши задачу максимально строго, "
            "понятно и последовательно."
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
            "Распознай условие на изображении и реши задачу максимально строго, "
            "понятно и последовательно."
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
        if message.text and message.text.startswith("/"):
            return
        await process_request(
            message,
            [types.Part.from_text(text=message.text or "")],
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
