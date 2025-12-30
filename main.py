# main.py
from io import BytesIO
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart
from aiogram.enums import ContentType
from config import TELEGRAM_BOT_TOKEN, STATE_TTL_SECONDS
from keyboards import market_keyboard, tickers_keyboard, timeframe_keyboard
from state import TTLState
from predictor import analyze  # теперь async
import logging

state = TTLState(STATE_TTL_SECONDS)

async def start(m: Message):
    await m.answer(
        "🤖 Боттрейд — анализ свечных графиков\n\n"
        "Выберите рынок для анализа:",
        reply_markup=market_keyboard()
    )

async def image_handler(m: Message):
    bio = BytesIO()
    file_id = m.photo[-1].file_id if m.photo else m.document.file_id
    file = await m.bot.get_file(file_id)
    await m.bot.download_file(file.file_path, bio)
    await state.set(m.from_user.id, "data", bio.getvalue())
    await state.set(m.from_user.id, "mode", "image")
    await m.answer("Выберите таймфрейм:", reply_markup=timeframe_keyboard())

async def callback_handler(cb: CallbackQuery):
    if not cb.data:
        await cb.answer()
        return

    data = cb.data
    user_id = cb.from_user.id

    logging.info(f"Получен callback: '{data}' от пользователя {user_id}")

    # Выбор рынка
    if data.startswith("market:"):
        market = data.split(":")[1]
        kb, info = tickers_keyboard(market)
        await cb.message.edit_text(info, reply_markup=kb)
        await state.set(user_id, "market", market)
        await cb.answer()
        return

    # Выбор тикера
    if data.startswith("ticker:"):
        ticker = data.split(":")[1]
        logging.info(f"Пользователь {user_id} выбрал тикер: {ticker}")
        await state.set(user_id, "ticker", ticker)
        await state.set(user_id, "mode", "api")  # режим API
        await cb.message.edit_text(
            f"Выбран инструмент: {ticker}\n\nВыберите таймфрейм:",
            reply_markup=timeframe_keyboard()
        )
        await cb.answer()
        return

    # Выбор таймфрейма — запуск анализа
    if data.startswith("tf:"):
        tf = data.split(":")[1]
        logging.info(f"Пользователь {user_id} выбрал таймфрейм: {tf}")

        mode = await state.get(user_id, "mode")

        if mode == "image":
            img_data = await state.get(user_id, "data")
            symbol = None  # неизвестен при скриншоте
            res, err = await analyze(image_bytes=img_data, tf=tf, symbol=symbol)
        else:
            symbol = await state.get(user_id, "ticker")
            logging.info(f"Анализ: режим=API, символ={symbol}, tf={tf}")
            res, err = await analyze(tf=tf, symbol=symbol)

        if err:
            await cb.message.answer(f"Ошибка: {err}")
        else:
            await send_result(cb.message, res)
            await cb.message.answer("Готов анализировать другой график?", reply_markup=market_keyboard())

        await state.clear(user_id)
        await cb.answer("Анализ завершён!")
        return

    # Возврат к рынкам
    if data.startswith("back:"):
        await cb.message.edit_text("Выберите рынок для анализа:", reply_markup=market_keyboard())
        await state.clear(user_id)
        await cb.answer()
        return

    await cb.answer("Неизвестная команда")

async def send_result(message: Message, res: dict):
    growth_pct = int(res["prob"] * 100)
    down_pct = int(res["down_prob"] * 100)
    txt = (
        f"📊 {res['symbol']} | {res['tf']} мин\n"
        f"Вероятность роста на 2–3 свечи: {growth_pct}%\n"
        f"Вероятность падения: {down_pct}%\n"
        f"Уверенность: {res['confidence']} ({res['confidence_score']})\n"
        f"Источник данных: {res['source']}\n"
    )
    if res["quality"] < 0.9:
        txt += f"Качество распознавания скрина: {res['quality']:.2f}\n"
    if res["patterns"]:
        txt += "Обнаруженные паттерны: " + ", ".join(res["patterns"]) + "\n"
    txt += "\n⚠ Это не финансовая рекомендация!"
    await message.answer(txt)

def main():
    bot = Bot(TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    dp.message.register(start, CommandStart())
    dp.message.register(image_handler, F.content_type.in_({ContentType.PHOTO, ContentType.DOCUMENT}))
    dp.callback_query.register(callback_handler)

    print("Бот запущен — финальная версия с асинхронным Grok!")
    dp.run_polling(bot)

if __name__ == "__main__":
    main()
