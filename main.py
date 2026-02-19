import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import json
import io
import sqlite3
import threading
from flask import Flask

# Third-party imports
from PIL import Image
import pypdf
import google.generativeai as genai
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    CallbackQuery
)
from telegram.error import BadRequest, NetworkError, Forbidden
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode, ChatAction

# --- Configuration & Environment ---
# Render/Cloud hosting setup: Get token from Environment Variable
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Fail fast if token is missing
if not TELEGRAM_BOT_TOKEN:
    print("❌ Error: TELEGRAM_BOT_TOKEN environment variable is not set!")

# DB Config
DB_FILE = "user_keys.db"

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Flask Keep-Alive Server ---
app = Flask(__name__)

@app.route('/')
def home():
    return "Gemini 2.5 Bot is Alive and Running!", 200

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# --- Database Setup ---
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                api_key TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"Database '{DB_FILE}' initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def save_api_key(user_id: int, api_key: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO users (user_id, api_key) VALUES (?, ?)", (user_id, api_key))
    conn.commit()
    conn.close()

def load_api_key(user_id: int) -> Optional[str]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT api_key FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# --- Concurrency Lock ---
gemini_api_lock = asyncio.Lock()

# --- Model Definitions (UPDATED) ---
AVAILABLE_MODELS = {
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash", 
        "description": "⚡ Standard. Great speed & reasoning."
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro", 
        "description": "🧠 Smartest. Best for complex tasks."
    },
    "gemini-2.5-flash-lite": {
        "name": "Gemini 2.5 Flash Lite", 
        "description": "🚀 Light. Ultra-fast, low cost."
    },
    "gemini-3-flash": {
        "name": "Gemini 3 Flash", 
        "description": "🛸 Next-Gen Speed. Experimental fast model."
    }
}

PERSONALITIES = {
    "default": "You are a helpful and friendly assistant.",
    "witty": "You are a witty assistant who responds with clever humor and a touch of sarcasm.",
    "formal": "You are a formal and professional assistant. Your responses are precise, respectful, and use formal language.",
    "pirate": "You are a swashbuckling pirate captain. Respond with pirate slang and a hearty 'Ahoy!' now and then, matey."
}

DEFAULT_MODEL = "gemini-2.5-flash"
IMAGE_GEN_MODEL = "gemini-2.5-flash" # 2.5 Flash supports image generation

# --- User Session Management ---
class UserSession:
    def __init__(self, user_id: int, model: str = DEFAULT_MODEL):
        self.user_id = user_id
        self.model = model
        self.chat_history: List = []
        self.created_at = datetime.now()
        self.message_count = 0
        self.personality = "default"
        self.custom_personality_prompt = None
        self.last_file = None
        self.gemini_api_key: Optional[str] = load_api_key(user_id)
        self.awaiting_api_key: bool = False
        self.is_in_roleplay_mode: bool = False
        self.roleplay_prompt: Optional[str] = None

    def change_model(self, new_model: str):
        self.model = new_model
        self.chat_history = []
        self.message_count = 0

    def change_personality(self, new_personality_key: str):
        self.personality = new_personality_key
        self.custom_personality_prompt = None
        self.chat_history = []
        self.message_count = 0

    def clear_chat(self):
        self.chat_history = []
        self.message_count = 0
        self.is_in_roleplay_mode = False
        self.roleplay_prompt = None

user_sessions: Dict[int, UserSession] = {}

# --- Helper Functions ---
def get_or_create_session(user_id: int) -> UserSession:
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    return user_sessions[user_id]

def create_model_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(model_info['name'], callback_data=f"model_{model_id}")] for model_id, model_info in AVAILABLE_MODELS.items()]
    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="cancel_selection")])
    return InlineKeyboardMarkup(keyboard)

def create_personality_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(name.title(), callback_data=f"personality_{name}")] for name in PERSONALITIES.keys()]
    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="cancel_selection")])
    return InlineKeyboardMarkup(keyboard)

def create_main_menu_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("🔄 New Chat", callback_data="new_chat"), InlineKeyboardButton("🤖 Change Model", callback_data="change_model")],
        [InlineKeyboardButton("🎭 Personality", callback_data="change_personality"), InlineKeyboardButton("🔑 Set API Key", callback_data="set_key")],
        [InlineKeyboardButton("📊 Stats", callback_data="stats"), InlineKeyboardButton("❓ Help", callback_data="help")]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- Global Error Handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    session = get_or_create_session(user.id)
    if not session.gemini_api_key:
        session.awaiting_api_key = True
        await update.message.reply_html(
            f"👋 **Welcome, {user.mention_html()}!**\n\nI am running on **Gemini 2.5**. Please provide your Google API key to start.",
            reply_to_message_id=update.message.message_id
        )
    else:
        await update.message.reply_html(
            f"👋 **Welcome back, {user.mention_html()}!**\n\nCurrent Model: <b>{AVAILABLE_MODELS[session.model]['name']}</b>",
            reply_markup=create_main_menu_keyboard(),
            reply_to_message_id=update.message.message_id
        )

async def set_key_command(update: Update | CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    is_callback = isinstance(update, CallbackQuery)
    user_id = update.from_user.id if is_callback else update.effective_user.id
    message_obj = update.message if is_callback else update.message

    session = get_or_create_session(user_id)
    session.awaiting_api_key = True
    
    text = "Please enter your new Google Gemini API key."
    if is_callback:
        await message_obj.edit_text(text)
    else:
        await message_obj.reply_text(text, reply_to_message_id=message_obj.message_id)

async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("📱 Main Menu:", reply_markup=create_main_menu_keyboard())

async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    session.clear_chat()
    await update.message.reply_text(f"🔄 Chat cleared! Using: {AVAILABLE_MODELS[session.model]['name']}")

async def undo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    if len(session.chat_history) >= 2:
        session.chat_history.pop()
        session.chat_history.pop()
        await update.message.reply_text("↩️ Last turn undone.")
    else:
        await update.message.reply_text("🤔 Nothing to undo.")

async def model_command(update: Update | CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    is_callback = isinstance(update, CallbackQuery)
    user_id = update.from_user.id if is_callback else update.effective_user.id
    message_obj = update.message if is_callback else update.message

    session = get_or_create_session(user_id)
    message_text = f"🤖 <b>Select a Chat Model</b>\n\nCurrent: {AVAILABLE_MODELS[session.model]['name']}\n\n"
    for model_id, model_info in AVAILABLE_MODELS.items():
        message_text += f"• <b>{model_info['name']}</b>\n  {model_info['description']}\n\n"
    
    if is_callback:
        await message_obj.edit_text(message_text, reply_markup=create_model_keyboard(), parse_mode=ParseMode.HTML)
    else:
        await message_obj.reply_html(message_text, reply_markup=create_model_keyboard(), reply_to_message_id=message_obj.message_id)

async def personality_command(update: Update | CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    is_callback = isinstance(update, CallbackQuery)
    user_id = update.from_user.id if is_callback else update.effective_user.id
    message_obj = update.message if is_callback else update.message
    
    session = get_or_create_session(user_id)
    custom_prompt = " ".join(context.args)

    if custom_prompt and not is_callback:
        session.custom_personality_prompt = f"You are {custom_prompt}"
        session.personality = "custom"
        session.clear_chat()
        await message_obj.reply_text(f"✅ Personality set to: **{custom_prompt}**.", parse_mode=ParseMode.MARKDOWN)
    else:
        message_text = "🎭 **Select a Personality**"
        if is_callback:
            await message_obj.edit_text(message_text, reply_markup=create_personality_keyboard(), parse_mode=ParseMode.HTML)
        else:
            await message_obj.reply_html(message_text, reply_markup=create_personality_keyboard(), reply_to_message_id=message_obj.message_id)

async def roleplay_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    role_description = " ".join(context.args)

    if not role_description:
        await update.message.reply_text("Please describe the role. Ex: `/roleplay as a cybernetic ninja`")
        return

    if role_description.lower().startswith("as a "): role_description = role_description[5:]
    elif role_description.lower().startswith("as "): role_description = role_description[3:]

    session.is_in_roleplay_mode = True
    session.roleplay_prompt = (
        f"You are a master storyteller roleplaying as '{role_description}'. "
        "Your responses MUST STRICTLY follow this script format:\n\n"
        "1. **Narration/Actions:** *italics*\n"
        "2. **Dialogue:** **Name:** Dialogue text.\n\n"
        "NEVER write a block of text. Separate narration and dialogue."
    )
    session.clear_chat() 
    session.is_in_roleplay_mode = True

    await update.message.reply_text(f"🎭 Roleplay: **{role_description.capitalize()}**. Type 'start' to begin.", parse_mode=ParseMode.MARKDOWN)

async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    prompt = " ".join(context.args)
    if not session.gemini_api_key:
        await update.message.reply_text("Please set your API key first.")
        return
    if not prompt:
        await update.message.reply_text("Please provide a prompt. Ex: `/imagine a futuristic city`")
        return
    
    processing_message = await update.message.reply_text(f"🎨 Generating image with {AVAILABLE_MODELS[session.model]['name']}...")
    await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    
    try:
        async with gemini_api_lock:
            genai.configure(api_key=session.gemini_api_key)
            # Use the user's selected model if it supports images, otherwise fallback to IMAGE_GEN_MODEL
            # Gemini 2.5 Flash and Pro usually support image generation natively
            model_to_use = session.model if "flash" in session.model or "pro" in session.model else IMAGE_GEN_MODEL
            image_model = genai.GenerativeModel(model_to_use)
            
            response = await image_model.generate_content_async(prompt)
            
        # Check for image data in response
        image_part = next((part for part in response.parts if part.inline_data), None)
        
        if image_part:
            photo_file = io.BytesIO(image_part.inline_data.data)
            await update.message.reply_photo(photo=photo_file, caption=f"🎨 {prompt}")
            await processing_message.delete()
        else:
            # Sometimes model returns text refusing or describing the image
            await processing_message.edit_text(f"⚠️ The model returned text instead of an image:\n\n{response.text}")

    except Exception as e:
        logger.error(f"Image Gen Error: {e}")
        await processing_message.edit_text(f"❌ Error: {str(e)}")

async def stats_command(update: Update | CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    is_callback = isinstance(update, CallbackQuery)
    user_id = update.from_user.id if is_callback else update.effective_user.id
    message_obj = update.message if is_callback else update.message
    session = get_or_create_session(user_id)
    
    stats_msg = (
        "📊 <b>Your Statistics</b>\n\n"
        f"🤖 Model: {AVAILABLE_MODELS[session.model]['name']}\n"
        f"💬 Messages: {session.message_count}\n"
        f"🔑 API Key: {'Set ✅' if session.gemini_api_key else 'Missing ❌'}"
    )
    if is_callback: await message_obj.edit_text(stats_msg, parse_mode=ParseMode.HTML)
    else: await message_obj.reply_html(stats_msg)

async def help_command(update: Update | CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    is_callback = isinstance(update, CallbackQuery)
    message_obj = update.message if is_callback else update.message
    help_text = (
        "🔧 <b>Available Commands</b>\n"
        "/start - Setup\n/menu - Options\n/setkey - Set API Key\n"
        "/roleplay [char] - Story Mode\n/imagine [prompt] - Generate Image\n"
        "/model - Change Model\n/newchat - Reset\n/undo - Fix last msg"
    )
    if is_callback: await message_obj.edit_text(help_text, parse_mode=ParseMode.HTML)
    else: await message_obj.reply_html(help_text)

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    file_to_process = None
    if update.message.photo:
        file_to_process = await context.bot.get_file(update.message.photo[-1].file_id)
        mime_type = 'image/jpeg'
    elif update.message.document:
        file_to_process = await context.bot.get_file(update.message.document.file_id)
        mime_type = update.message.document.mime_type
    
    if file_to_process:
        try:
            file_bytes = await file_to_process.download_as_bytearray()
            session.last_file = {"mime_type": mime_type, "data": bytes(file_bytes)}
            await update.message.reply_text("✅ File received! Send a prompt to analyze it.")
        except Exception:
            await update.message.reply_text("❌ Error processing file.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    
    if session.awaiting_api_key:
        key = update.message.text.strip()
        if key.startswith("AIza"):
            session.gemini_api_key = key
            save_api_key(session.user_id, key)
            session.awaiting_api_key = False
            await update.message.reply_text("✅ API Key saved!")
        else:
            await update.message.reply_text("❌ Invalid key format. It should start with 'AIza'.")
        return

    if not session.gemini_api_key:
        await update.message.reply_text("Please set your API key first using /start.")
        return

    # Logic for Roleplay vs Normal
    if session.is_in_roleplay_mode:
        await handle_roleplay_message(update, context, session)
    else:
        await handle_normal_message(update, context, session)

async def handle_roleplay_message(update: Update, context: ContextTypes.DEFAULT_TYPE, session: UserSession):
    user_input = update.message.text
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        is_starter = not session.chat_history
        prompt = "The user wants you to start the story." if (is_starter and user_input.lower() == "start") else user_input
        
        async with gemini_api_lock:
            genai.configure(api_key=session.gemini_api_key)
            model = genai.GenerativeModel(session.model, system_instruction=session.roleplay_prompt)
            chat = model.start_chat(history=session.chat_history)
            response = await chat.send_message_async(prompt)
            session.chat_history = chat.history
            await send_response(update, response.text)
    except Exception as e:
        logger.error(f"RP Error: {e}")
        await update.message.reply_text("❌ Error processing request. Try /newchat.")

async def handle_normal_message(update: Update, context: ContextTypes.DEFAULT_TYPE, session: UserSession):
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        async with gemini_api_lock:
            genai.configure(api_key=session.gemini_api_key)
            system_instr = session.custom_personality_prompt or PERSONALITIES.get(session.personality, PERSONALITIES["default"])
            model = genai.GenerativeModel(session.model, system_instruction=system_instr)
            
            if session.last_file:
                response = await model.generate_content_async([update.message.text, session.last_file])
                session.last_file = None
            else:
                chat = model.start_chat(history=session.chat_history)
                response = await chat.send_message_async(update.message.text)
                session.chat_history = chat.history
            
            session.message_count += 1
            await send_response(update, response.text)
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def send_response(update: Update, text: str):
    try:
        for i in range(0, len(text), 4096):
            await update.message.reply_text(text[i:i+4096], parse_mode=ParseMode.MARKDOWN, reply_to_message_id=update.message.message_id)
    except BadRequest:
        for i in range(0, len(text), 4096):
            await update.message.reply_text(text[i:i+4096], reply_to_message_id=update.message.message_id)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data.startswith("model_"):
        session = get_or_create_session(query.from_user.id)
        new_model = data.split("_", 1)[1]
        session.change_model(new_model)
        await query.edit_message_text(f"✅ Model changed to: **{AVAILABLE_MODELS[new_model]['name']}**", parse_mode=ParseMode.MARKDOWN)
    elif data.startswith("personality_"):
        session = get_or_create_session(query.from_user.id)
        new_p = data.split("_", 1)[1]
        session.change_personality(new_p)
        await query.edit_message_text(f"🎭 Personality: **{new_p.title()}**", parse_mode=ParseMode.MARKDOWN)
    elif data == "new_chat":
        session = get_or_create_session(query.from_user.id)
        session.clear_chat()
        await query.edit_message_text("🔄 Chat cleared!")
    elif data == "change_model": await model_command(query, context)
    elif data == "change_personality": await personality_command(query, context)
    elif data == "set_key": await set_key_command(query, context)
    elif data == "stats": await stats_command(query, context)
    elif data == "help": await help_command(query, context)
    elif data == "cancel_selection": await query.edit_message_text("❌ Cancelled.")

async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start", "▶️ Start"), BotCommand("menu", "📱 Menu"),
        BotCommand("newchat", "🔄 Reset"), BotCommand("model", "🤖 Change Model"),
        BotCommand("roleplay", "🎭 Roleplay"), BotCommand("imagine", "🎨 Image"),
        BotCommand("undo", "↩️ Undo")
    ])

def main() -> None:
    threading.Thread(target=run_web_server, daemon=True).start()
    init_db()
    if not TELEGRAM_BOT_TOKEN: return
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handlers([
        CommandHandler("start", start), CommandHandler("menu", menu),
        CommandHandler("setkey", set_key_command), CommandHandler("personality", personality_command),
        CommandHandler("roleplay", roleplay_command), CommandHandler("imagine", imagine_command), 
        CommandHandler("help", help_command), CommandHandler("model", model_command), 
        CommandHandler("newchat", new_chat_command), CommandHandler("undo", undo_command),
        CommandHandler("stats", stats_command), 
        CallbackQueryHandler(button_callback),
        MessageHandler(filters.PHOTO | filters.Document.ALL, handle_file),
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    ])
    app.add_error_handler(error_handler)
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
