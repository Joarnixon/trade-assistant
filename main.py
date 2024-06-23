import telebot
from Coefficients import Static, Dynamic
import asyncio
from LogHandler import Handling
from DayChanges import GetChanges
import MinuteCandles
handle_op = Handling()
Handling.handle_log(handle_op)
bot = telebot.TeleBot(Static.telegram_token)
market_op2 = GetChanges()


asyncio.run(MinuteCandles.main())
# Функция-обработчик сообщений

@bot.channel_post_handler(commands=['изменения'], func=lambda message: message.chat.id == Static.chat_id)
def handle_channel_post(message: telebot.types.Message):
            print('Поручение принял')
            top_gains, top_loses, elapsed_time = asyncio.run(GetChanges.main(self = market_op2))
            response_message = f"{top_gains}\n\n'=================================='\n{top_loses}\n\n" \
                               f"Время выполнения: {elapsed_time:.2f} сек"
            bot.send_message(Static.chat_id, response_message)


# Запуск бота
bot.polling()



