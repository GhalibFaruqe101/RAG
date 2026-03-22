import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import RAG_sys

os.environ["LANGCHAIN_TRACING_V2"] = "false"
load_dotenv(".env.RAG")
tel_token = os.getenv("Telegram_API_KEY")


TOKEN = tel_token

logging.basicConfig(level=logging.INFO)


# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("RAG Bot is ready. Ask me anything.")


# Handle user messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text

    # Run synchronous ask function in a separate thread to prevent blocking event loop
    response = await asyncio.to_thread(RAG_sys.ask, user_query)

    await update.message.reply_text(response)


def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
