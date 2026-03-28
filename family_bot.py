#!/usr/bin/env python3
"""
Family Bot Runner — Nova (Ogechi) and Bolt (Kids)
Usage: python3 family_bot.py nova
       python3 family_bot.py bolt
"""
import os, sys, asyncio
from dotenv import load_dotenv

load_dotenv()

BOT_NAME = sys.argv[1] if len(sys.argv) > 1 else "nova"

if BOT_NAME == "nova":
    TOKEN = os.getenv("NOVA_BOT_TOKEN")
    PERSONA_FILE = os.getenv("NOVA_PERSONA_FILE")
    MODEL = "llama3.1:70b"
    BOT_DISPLAY = "Nova"
elif BOT_NAME == "bolt":
    TOKEN = os.getenv("BOLT_BOT_TOKEN")
    PERSONA_FILE = os.getenv("BOLT_PERSONA_FILE")
    MODEL = "llama3.1:70b"
    BOT_DISPLAY = "Bolt"
else:
    print(f"Unknown bot: {BOT_NAME}")
    sys.exit(1)

if not TOKEN:
    print(f"No token for {BOT_NAME}")
    sys.exit(1)

PERSONA = open(PERSONA_FILE).read() if PERSONA_FILE and os.path.exists(PERSONA_FILE) else ""
OLLAMA_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")

import requests
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

# Per-user conversation history
histories = {}

def chat(user_id: int, text: str) -> str:
    if user_id not in histories:
        histories[user_id] = []
    
    histories[user_id].append({"role": "user", "content": text})
    
    # Keep last 20 messages
    if len(histories[user_id]) > 20:
        histories[user_id] = histories[user_id][-20:]
    
    messages = []
    if PERSONA:
        messages.append({"role": "system", "content": PERSONA})
    messages.extend(histories[user_id])
    
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 1024}
        }, timeout=120)
        r.raise_for_status()
        response = r.json()["message"]["content"]
        histories[user_id].append({"role": "assistant", "content": response})
        return response
    except Exception as e:
        return f"Sorry, I had a technical issue: {e}"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_id = update.effective_user.id
    text = update.message.text
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    response = chat(user_id, text)
    
    # Split long messages
    for i in range(0, len(response), 4000):
        await update.message.reply_text(response[i:i+4000])

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if BOT_NAME == "nova":
        msg = "Hi Ogechi! I'm Nova 🌟 I'm here to help you crush the LCSW exam. We can study together, do practice vignettes, quiz on any topic, or just chat. What do you want to work on? Type /help to see all my study commands."
    else:
        msg = "Hey! I'm Bolt ⚡ I'm here to help you learn and have fun! Who am I talking to — Andrew, Christopher, or Athena? Type /help to see what I can do!"
    await update.message.reply_text(msg)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    response = chat(user_id, "List all your available commands in a clear, friendly way.")
    await update.message.reply_text(response)

async def cmd_vignette(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    response = chat(user_id, "Give me a clinical vignette practice question with 4 answer choices (A, B, C, D). Wait for my answer before explaining.")
    await update.message.reply_text(response)

async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    topic = " ".join(context.args) if context.args else "any topic"
    response = chat(user_id, f"Give me a quiz question about {topic}.")
    await update.message.reply_text(response)

async def cmd_drill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    topic = " ".join(context.args) if context.args else "ethics"
    response = chat(user_id, f"Let's drill on {topic}. Give me a focused practice question.")
    await update.message.reply_text(response)

async def cmd_explain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    topic = " ".join(context.args) if context.args else "the last concept we discussed"
    response = chat(user_id, f"Explain {topic} clearly and thoroughly.")
    await update.message.reply_text(response)

async def cmd_challenge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    response = chat(user_id, "Give me a fun challenge question! Make it competitive and exciting.")
    await update.message.reply_text(response)

async def cmd_story(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    response = chat(user_id, "Tell me a short educational story that teaches something fun!")
    await update.message.reply_text(response)

async def cmd_homework(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    problem = " ".join(context.args) if context.args else ""
    if problem:
        response = chat(user_id, f"Help me with this homework: {problem}")
    else:
        response = "Tell me your homework problem and I'll walk you through it step by step! 📚"
    await update.message.reply_text(response)

def main():
    print(f"Starting {BOT_DISPLAY} bot...", flush=True)
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    
    if BOT_NAME == "nova":
        app.add_handler(CommandHandler("vignette", cmd_vignette))
        app.add_handler(CommandHandler("quiz", cmd_quiz))
        app.add_handler(CommandHandler("drill", cmd_drill))
        app.add_handler(CommandHandler("explain", cmd_explain))
    else:
        app.add_handler(CommandHandler("challenge", cmd_challenge))
        app.add_handler(CommandHandler("story", cmd_story))
        app.add_handler(CommandHandler("homework", cmd_homework))
        app.add_handler(CommandHandler("quiz", cmd_quiz))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print(f"{BOT_DISPLAY} is ready!", flush=True)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
