import asyncio
import io
import logging
import os
from collections.abc import Awaitable, Callable, Iterable, Sequence

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from google import genai
from google.genai import errors as genai_errors
from google.genai import types


# Запас относительно системного лимита Android (1024 UTF-16 единицы), чтобы
# отдельная часть ответа лучше помещалась в уведомление Telegram.
TELEGRAM_MESSAGE_LIMIT = 900
DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3.1-flash-lite")
NEXT_WORDS = {"дальше", "далее", "продолжить", "next"}
BACK_WORDS = {"назад", "обратно", "back"}
GEMINI_RETRY_INITIAL_DELAY = 3
GEMINI_RETRY_MAX_DELAY = 30
RETRYABLE_GEMINI_CODES = {429, 500, 502, 503, 504}

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
7. Для одной стандартной задачи ориентируйся примерно на 900–1500 символов,
   но это не ограничение. Если задач несколько, реши каждую из них отдельно и
   полностью. Никогда не пропускай задачу, формулу, существенный шаг или ответ
   ради сокращения объёма. Длинное решение бот безопасно отправит частями.

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


def parse_model_names(raw: str) -> tuple[str, ...]:
    configured = [item.strip() for item in raw.split(",") if item.strip()]
    candidates = configured or list(DEFAULT_MODELS)
    return tuple(dict.fromkeys(candidates))


def gemini_retry_delay(failed_attempt: int) -> int:
    """Return exponential retry delay capped at 30 seconds."""
    exponent = max(0, min(failed_attempt - 1, 10))
    return min(GEMINI_RETRY_INITIAL_DELAY * (2**exponent), GEMINI_RETRY_MAX_DELAY)


def is_retryable_gemini_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) in RETRYABLE_GEMINI_CODES


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


def is_back_request(text: str) -> bool:
    normalized = text.strip().casefold()
    return (
        normalized in BACK_WORDS
        or normalized == "/back"
        or normalized.startswith("/back@")
    )


def format_answer_part(chunk: str, index: int, total: int) -> str:
    if total == 1:
        return chunk

    text = f"Часть {index + 1} из {total}\n\n{chunk}"
    if index == 0:
        text += "\n\n/next"
    elif index + 1 == total:
        text += "\n\n/back"
    else:
        text += "\n\n/back · /next"
    return text


async def ask_gemini_with_retry(
    ai_client: genai.Client,
    model_names: Sequence[str],
    request_gate: asyncio.Semaphore,
    contents: Iterable[types.Part],
    on_retry: Callable[[int, int, str, str], Awaitable[None]],
) -> str:
    if not model_names:
        raise ValueError("Нужна хотя бы одна модель Gemini")

    parts = list(contents)
    failed_attempt = 0
    model_index = 0

    while True:
        model_name = model_names[model_index]
        try:
            # The semaphore covers only the API call. A request waiting for a
            # retry must not occupy a slot needed by other users.
            async with request_gate:
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
            return (response.text or "").strip()
        except (genai_errors.ClientError, genai_errors.ServerError) as exc:
            if not is_retryable_gemini_error(exc):
                raise

            failed_attempt += 1
            delay = gemini_retry_delay(failed_attempt)
            model_index = (model_index + 1) % len(model_names)
            next_model_name = model_names[model_index]
            logging.warning(
                "Gemini %s временно недоступен (HTTP %s), "
                "попытка №%s через %s с; следующая модель — %s",
                model_name,
                exc.code,
                failed_attempt,
                delay,
                next_model_name,
            )
            await on_retry(
                failed_attempt, delay, model_name, next_model_name
            )
            await asyncio.sleep(delay)


def make_router(
    ai_client: genai.Client,
    model_names: Sequence[str],
    request_gate: asyncio.Semaphore,
) -> Router:
    router = Router()
    pending_answers: dict[int, tuple[list[str], int]] = {}

    async def send_first_part(message: Message, answer: str) -> None:
        chunks = split_message(answer)
        user_id = message.from_user.id if message.from_user else None

        if user_id is not None and len(chunks) > 1:
            pending_answers[user_id] = (chunks, 0)
        elif user_id is not None:
            pending_answers.pop(user_id, None)

        await message.answer(format_answer_part(chunks[0], 0, len(chunks)))

    async def send_next_part(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if user_id is None or user_id not in pending_answers:
            await message.answer(
                "Продолжения пока нет. Сначала отправьте новую задачу."
            )
            return

        chunks, current_index = pending_answers[user_id]
        next_index = current_index + 1
        if next_index >= len(chunks):
            await message.answer("Последняя часть. /back")
            return

        pending_answers[user_id] = (chunks, next_index)
        await message.answer(
            format_answer_part(chunks[next_index], next_index, len(chunks))
        )

    async def send_previous_part(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if user_id is None or user_id not in pending_answers:
            await message.answer(
                "Сохранённого ответа пока нет. Сначала отправьте новую задачу."
            )
            return

        chunks, current_index = pending_answers[user_id]
        previous_index = current_index - 1
        if previous_index < 0:
            await message.answer("Первая часть. /next")
            return

        pending_answers[user_id] = (chunks, previous_index)
        await message.answer(
            format_answer_part(
                chunks[previous_index], previous_index, len(chunks)
            )
        )

    async def process_request(message: Message, contents: Iterable[types.Part]) -> None:
        status = await message.answer("Решаю задачу...")

        async def update_retry_status(
            failed_attempt: int,
            delay: int,
            _failed_model: str,
            _next_model: str,
        ) -> None:
            try:
                await status.edit_text(
                    "Gemini сейчас перегружен. Запрос не потерян — "
                    f"через {delay} с попробую другую модель. "
                    f"Неудачных попыток: {failed_attempt}."
                )
            except Exception:
                logging.warning(
                    "Не удалось обновить статус повтора", exc_info=True
                )

        try:
            answer = await ask_gemini_with_retry(
                ai_client=ai_client,
                model_names=model_names,
                request_gate=request_gate,
                contents=contents,
                on_retry=update_retry_status,
            )
            try:
                await status.delete()
            except Exception:
                logging.warning("Не удалось удалить служебное сообщение", exc_info=True)
            await send_first_part(message, answer)
        except (genai_errors.ClientError, genai_errors.ServerError):
            logging.exception(
                "Gemini не обработал запрос пользователя %s",
                message.from_user.id if message.from_user else "unknown",
            )
            try:
                await status.edit_text(
                    "Gemini отклонил запрос. Проверьте API-ключ, "
                    "доступ к модели и журнал Railway."
                )
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
        await message.answer(
            "Пришлите фотографию задачи или её текст. Для снимка можно добавить "
            "подпись с уточнением, что именно требуется найти.\n\n"
            "Навигация по частям: /next — вперёд, /back — назад.\n\n"
            "Если Gemini перегружен, бот сам сохранит и повторит запрос."
        )

    @router.message(Command("next"))
    async def next_handler(message: Message) -> None:
        await send_next_part(message)

    @router.message(Command("back"))
    async def back_handler(message: Message) -> None:
        await send_previous_part(message)

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
        if is_back_request(text):
            await send_previous_part(message)
            return
        if text.startswith("/"):
            return
        await process_request(
            message,
            [types.Part.from_text(text=text)],
        )

    @router.message()
    async def unsupported_handler(message: Message) -> None:
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
    model_names = parse_model_names(os.getenv("GEMINI_MODELS", ""))
    max_parallel = max(1, int(os.getenv("MAX_PARALLEL_REQUESTS", "4")))

    bot = Bot(token=telegram_token)
    dispatcher = Dispatcher()
    ai_client = genai.Client(api_key=gemini_api_key)
    dispatcher.include_router(
        make_router(
            ai_client=ai_client,
            model_names=model_names,
            request_gate=asyncio.Semaphore(max_parallel),
        )
    )

    logging.info("Запуск бота с моделями %s", ", ".join(model_names))
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await dispatcher.start_polling(bot)
    finally:
        await ai_client.aio.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
