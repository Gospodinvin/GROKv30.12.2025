from io import BytesIO
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart
from aiogram.enums import ContentType
from config import TELEGRAM_BOT_TOKEN, STATE_TTL_SECONDS
from keyboards import market_keyboard, tickers_keyboard, timeframe_keyboard
from state import TTLState
from predictor import analyze
import logging

state = TTLState(STATE_TTL_SECONDS)

async def start(m: Message):
    await m.answer(
        "🤖 Боттрейд — анализ графиков с индикаторами и скальпинг-стратегией\n\n"
        "Выберите рынок:",
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
    logging.info(f"Callback: '{data}' от {user_id}")

    if data.startswith("market:"):
        market = data.split(":")[1]
        kb, info = tickers_keyboard(market)
        await cb.message.edit_text(info, reply_markup=kb)
        await state.set(user_id, "market", market)
        await cb.answer()
        return

    if data.startswith("ticker:"):
        ticker = data.split(":")[1]
        logging.info(f"Выбран тикер: {ticker}")
        await state.set(user_id, "ticker", ticker)
        await state.set(user_id, "mode", "api")
        await cb.message.edit_text(f"Инструмент: {ticker}\n\nВыберите таймфрейм:", reply_markup=timeframe_keyboard())
        await cb.answer()
        return

    if data.startswith("tf:"):
        tf = data.split(":")[1]
        logging.info(f"Выбран TF: {tf}")

        mode = await state.get(user_id, "mode")
        if mode == "image":
            img_data = await state.get(user_id, "data")
            res, err = await analyze(image_bytes=img_data, tf=tf)
        else:
            symbol = await state.get(user_id, "ticker")
            res, err = await analyze(tf=tf, symbol=symbol)

        if err:
            await cb.message.answer(f"Ошибка: {err}")
        else:
            await send_result(cb.message, res)
            await cb.message.answer("Готов к новому анализу?", reply_markup=market_keyboard())

        await state.clear(user_id)
        await cb.answer("Готово!")
        return

    if data.startswith("back:"):
        await cb.message.edit_text("Выберите рынок:", reply_markup=market_keyboard())
        await state.clear(user_id)
        await cb.answer()
        return

    await cb.answer("Неизвестно")

async def send_result(message: Message, res: dict):
    prob = res["prob"]
    growth_percent = int(prob * 100)
    down_percent = 100 - growth_percent

    # Определяем рекомендацию
    if prob >= 0.65:
        recommendation = "🟢 **BUY** (Покупать)"
        color = "🟢"
    elif prob <= 0.35:
        recommendation = "🔴 **SELL** (Продавать)"
        color = "🔴"
    else:
        recommendation = "⚪ **HOLD** (Держать / Наблюдать)"
        color = "⚪"

    txt = (
        f"📊 **{res['symbol']} | {res['tf']} мин**\n\n"
        f"{color} **Рекомендация:** {recommendation}\n"
        f"Рост (2–3 свечи): **{growth_percent}%**\n"
        f"Падение: **{down_percent}%**\n"
        f"Уверенность: **{res['confidence']}** ({res['confidence_score']})\n"
        f"Режим рынка: {res['regime'].capitalize()}\n"
        f"Источник: {res['source']}\n"
    )

    if res.get("quality", 1.0) < 0.9:
        txt += f"⚠ Качество скрина: {res['quality']:.2f} (может влиять на точность)\n"

    if res["patterns"]:
        txt += f"🔥 Паттерны: {', '.join(res['patterns'])}\n"

    ind = res.get("indicators", {})
    txt += (
        f"\n📈 Индикаторы:\n"
        f"• RSI: {ind.get('rsi', 50):.1f}\n"
        f"• Stoch: {ind.get('stoch', 50):.1f}\n"
        f"• ADX (сила тренда): {ind.get('adx', 20):.1f}\n"
        f"• MACD: {ind.get('macd', 0):.5f}\n"
        f"• Bollinger: {ind.get('bb', 'neutral').capitalize()}\n"
    )

    txt += "\n⚠ **Не финансовая рекомендация! Торгуйте на свой страх и риск.**"

    await message.answer(txt, parse_mode="Markdown")

def main():
    bot = Bot(TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    dp.message.register(start, CommandStart())
    dp.message.register(image_handler, F.content_type.in_({ContentType.PHOTO, ContentType.DOCUMENT}))
    dp.callback_query.register(callback_handler)
    print("Бот запущен — версия со скальпингом и индикаторами!")
    dp.run_polling(bot)

if __name__ == "__main__":
    main()

