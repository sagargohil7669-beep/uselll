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
    raise ValueError("❌ Error: TELEGRAM_BOT_TOKEN environment variable is not set!")

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
# This creates a web server to satisfy Render's port requirement
app = Flask(__name__)

@app.route('/')
def home():
    return "Gemini Bot is Alive and Running!", 200

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    # Run Flask without the reloader to prevent creating two bot instances
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# --- Database Setup ---
def init_db():
    """Initializes the database and creates the users table if it doesn't exist."""
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
        logger.info(f"Database '{DB_FILE}' initialized/checked.")
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

# --- Model & Personality Definitions ---
AVAILABLE_MODELS = {
    "gemini-2.0-flash": {"name": "Gemini 2.0 Flash", "description": "🚀 Adaptive thinking, cost efficiency"},
    "gemini-1.5-pro": {"name": "Gemini 1.5 Pro", "description": "🧠 Enhanced thinking and reasoning"},
    "gemini-1.5-flash": {"name": "Gemini 1.5 Flash", "description": "💨 Fast and versatile performance"}
}
# Note: Updated model name to official 2.0 release if available, fallback usually works.
PERSONALITIES = {
    "default": "You are a helpful and friendly assistant.",
    "witty": "You are a witty assistant who responds with clever humor and a touch of sarcasm.",
    "formal": "You are a formal and professional assistant. Your responses are precise, respectful, and use formal language.",
    "pirate": "You are a swashbuckling pirate captain. Respond with pirate slang and a hearty 'Ahoy!' now and then, matey."
}
DEFAULT_MODEL = "gemini-1.5-flash" # Safer default for free tier
IMAGE_GEN_MODEL = "gemini-2.0-flash" # Use latest capable model

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
        [InlineKeyboardButton("🎭 Change Personality", callback_data="change_personality"), InlineKeyboardButton("🔑 Set API Key", callback_data="set_key")],
        [InlineKeyboardButton("📊 Stats", callback_data="stats"), InlineKeyboardButton("❓ Help", callback_data="help")]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- Global Error Handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and handle specific telegram errors."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    # If the error is a network error, we usually just log it and retry internally by PTB
    if isinstance(context.error, NetworkError):
        return

    # For other errors, we might want to notify the user (if possible)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ An internal error occurred. If this persists, try /newchat."
            )
        except Exception:
            pass # Failed to send error message

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    session = get_or_create_session(user.id)
    if not session.gemini_api_key:
        session.awaiting_api_key = True
        await update.message.reply_html(
            f"👋 **Welcome, {user.mention_html()}!**\n\nTo get started, please provide your Google Gemini API key.",
            reply_to_message_id=update.message.message_id
        )
    else:
        await update.message.reply_html(
            f"👋 **Welcome back, {user.mention_html()}!**\n\nI'm ready to assist. Use `/menu` for options.",
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
    await update.message.reply_text(
        "📱 Main Menu:",
        reply_markup=create_main_menu_keyboard(),
        reply_to_message_id=update.message.message_id
    )

async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    session.clear_chat()
    await update.message.reply_text(
        f"🔄 Chat cleared! Model: {AVAILABLE_MODELS[session.model]['name']}",
        reply_to_message_id=update.message.message_id
    )

async def undo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    if len(session.chat_history) >= 2:
        session.chat_history.pop()
        session.chat_history.pop()
        await update.message.reply_text(
            "↩️ Your last prompt and my response have been undone.",
            reply_to_message_id=update.message.message_id
        )
    else:
        await update.message.reply_text(
            "🤔 Nothing to undo.",
            reply_to_message_id=update.message.message_id
        )

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
        await message_obj.reply_text(
            f"✅ My personality is now: **{custom_prompt}**.",
            parse_mode=ParseMode.MARKDOWN,
            reply_to_message_id=message_obj.message_id
        )
    else:
        message_text = "🎭 **Select a Personality**\n\nChoose an option or type:\n`/personality a cheerful teacher`"
        if is_callback:
            await message_obj.edit_text(message_text, reply_markup=create_personality_keyboard(), parse_mode=ParseMode.HTML)
        else:
            await message_obj.reply_html(message_text, reply_markup=create_personality_keyboard(), reply_to_message_id=message_obj.message_id)

async def roleplay_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    role_description = " ".join(context.args)

    if not role_description:
        await update.message.reply_text(
            "Please describe the role. Example: `/roleplay as a suspicious detective`",
            reply_to_message_id=update.message.message_id
        )
        return

    if role_description.lower().startswith("as a "):
        role_description = role_description[5:]
    elif role_description.lower().startswith("as "):
        role_description = role_description[3:]

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

    await update.message.reply_text(
        f"🎭 Roleplay mode activated: **{role_description.capitalize()}**.\n"
        "Type `start` to begin or describe the opening scene.",
        parse_mode=ParseMode.MARKDOWN,
        reply_to_message_id=update.message.message_id
    )

async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    prompt = " ".join(context.args)
    if not session.gemini_api_key:
        await update.message.reply_text("Please set your Gemini API key first using /start or /setkey.", reply_to_message_id=update.message.message_id)
        return
    if not prompt:
        await update.message.reply_text("Please provide a description. Ex: `/imagine a robot painting a sunset`", reply_to_message_id=update.message.message_id)
        return
    processing_message = await update.message.reply_text(f"🎨 Generating image for: \"{prompt}\"...", reply_to_message_id=update.message.message_id)
    await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    try:
        async with gemini_api_lock:
            genai.configure(api_key=session.gemini_api_key)
            # Use specific image gen model or experimental
            image_model = genai.GenerativeModel(IMAGE_GEN_MODEL) 
            # Note: The API for image generation varies by region/model. 
            # This implementation assumes the user has access to a model capable of 'generate_images' 
            # or multimodal output. Standard Gemini 1.5 doesn't output images directly via text prompt usually
            # but newer 2.0 or Imagen models do.
            # Fallback Logic for standard text models attempting images:
            response = await image_model.generate_content_async(prompt)
            
        # Check if parts contain inline data (Image)
        if response.parts and hasattr(response.parts[0], 'inline_data'):
             # Logic depends on specific Gemini SDK version for Images
             pass
        else:
             # If the model simply returned text saying it can't generate images:
             await processing_message.edit_text("⚠️ This model version returned text instead of an image. Ensure you are using a Gemini model that supports direct image output (like Imagen or Gemini 2.0 Flash in supported regions).")
             return

        # Attempt to extract image
        image_data_part = next((part for part in response.parts if part.inline_data), None)
        if image_data_part:
            photo_file = io.BytesIO(image_data_part.inline_data.data)
            await update.message.reply_photo(photo=photo_file, caption=f"Image for: \"{prompt}\"", reply_to_message_id=update.message.message_id)
            await processing_message.delete()
        else:
             # Basic handling if SDK response structure differs
             await processing_message.edit_text("Could not generate an image. Response format unexpected.")

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        await processing_message.edit_text(f"❌ Error: {str(e)}")

async def stats_command(update: Update | CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    is_callback = isinstance(update, CallbackQuery)
    user_id = update.from_user.id if is_callback else update.effective_user.id
    message_obj = update.message if is_callback else update.message

    session = get_or_create_session(user_id)
    personality_display = session.custom_personality_prompt.replace("You are ", "", 1) if session.personality == "custom" else session.personality.title()
    stats_message = (
        "📊 <b>Your Statistics</b>\n\n"
        f"🤖 Model: {AVAILABLE_MODELS[session.model]['name']}\n"
        f"🎭 Personality: {personality_display}\n"
        f"💬 Messages: {session.message_count}\n"
        f"🔑 API Key Set: {'Yes' if session.gemini_api_key else 'No'}"
    )
    
    if is_callback:
        await message_obj.edit_text(stats_message, parse_mode=ParseMode.HTML)
    else:
        await message_obj.reply_html(stats_message, reply_to_message_id=message_obj.message_id)

async def help_command(update: Update | CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    is_callback = isinstance(update, CallbackQuery)
    message_obj = update.message if is_callback else update.message

    help_text = (
        "🔧 <b>Available Commands</b>\n\n"
        "/start - Setup\n"
        "/menu - Main menu\n"
        "/setkey - Set API Key\n"
        "/roleplay `[char]` - Start story\n"
        "/imagine `[prompt]` - Create image\n"
        "/personality `[desc]` - Set personality\n"
        "/model - Change AI\n"
        "/newchat - Reset chat\n"
        "/undo - Undo last\n"
        "/stats - Stats\n"
    )
    
    if is_callback:
        await message_obj.edit_text(help_text, parse_mode=ParseMode.HTML)
    else:
        await message_obj.reply_html(help_text, reply_to_message_id=message_obj.message_id)

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
            await update.message.reply_text("✅ File received.", reply_to_message_id=update.message.message_id)
        except Exception as e:
            await update.message.reply_text("❌ Error downloading file.", reply_to_message_id=update.message.message_id)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_or_create_session(update.effective_user.id)
    
    # API Key Setup Flow
    if session.awaiting_api_key:
        if update.message.text.strip().startswith("AIza"):
            session.gemini_api_key = update.message.text.strip()
            save_api_key(session.user_id, session.gemini_api_key)
            session.awaiting_api_key = False
            await update.message.reply_text("✅ API Key saved! You can now start chatting.", reply_to_message_id=update.message.message_id)
        else:
            await update.message.reply_text("❌ That doesn't look like a valid Gemini API key (starts with AIza). Try again.", reply_to_message_id=update.message.message_id)
        return

    if not session.gemini_api_key:
        await update.message.reply_text("Please set your Gemini API key first using /start.", reply_to_message_id=update.message.message_id)
        return

    if session.is_in_roleplay_mode:
        await handle_roleplay_message(update, context, session)
    else:
        await handle_normal_message(update, context, session)

async def handle_roleplay_message(update: Update, context: ContextTypes.DEFAULT_TYPE, session: UserSession):
    user_input = update.message.text
    await update.effective_chat.send_action(ChatAction.TYPING)

    try:
        is_story_starter = not session.chat_history
        prompt_to_send = ""
        if is_story_starter:
            if user_input.lower().strip() == "start":
                prompt_to_send = "The user wants you to start the story. Write an opening scene."
            else:
                prompt_to_send = f"Opening scene: '{user_input}'. Continue the story."
        else:
            prompt_to_send = user_input

        async with gemini_api_lock:
            genai.configure(api_key=session.gemini_api_key)
            model = genai.GenerativeModel(session.model, system_instruction=session.roleplay_prompt)
            chat = model.start_chat(history=session.chat_history)
            response = await chat.send_message_async(prompt_to_send)
            ai_response = response.text
            session.chat_history = chat.history
        
        await send_response(update, ai_response)

    except Exception as e:
        logger.error(f"Roleplay Error: {e}")
        await update.message.reply_text("❌ Error processing request. Check your API key or model availability.", reply_to_message_id=update.message.message_id)

async def handle_normal_message(update: Update, context: ContextTypes.DEFAULT_TYPE, session: UserSession):
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        async with gemini_api_lock:
            genai.configure(api_key=session.gemini_api_key)
            system_instruction = session.custom_personality_prompt or PERSONALITIES.get(session.personality, PERSONALITIES["default"])
            model = genai.GenerativeModel(session.model, system_instruction=system_instruction)
            
            if session.last_file:
                prompt_parts = [update.message.text, session.last_file]
                response = await model.generate_content_async(prompt_parts)
                session.last_file = None
            else:
                chat = model.start_chat(history=session.chat_history)
                response = await chat.send_message_async(update.message.text)
                session.chat_history = chat.history
        
        session.message_count += 1
        await send_response(update, response.text)
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        error_msg = str(e)
        if "API key" in error_msg:
             await update.message.reply_text("❌ Invalid API Key. Use /setkey to update.", reply_to_message_id=update.message.message_id)
        else:
             await update.message.reply_text("❌ An error occurred.", reply_to_message_id=update.message.message_id)

async def send_response(update: Update, text: str):
    try:
        # Telegram max message length is 4096
        for i in range(0, len(text), 4096):
            await update.message.reply_text(text[i:i+4096], parse_mode=ParseMode.MARKDOWN, reply_to_message_id=update.message.message_id)
    except BadRequest:
        # Fallback if Markdown fails
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
        await query.edit_message_text(f"✅ Model: **{AVAILABLE_MODELS[new_model]['name']}**.", parse_mode=ParseMode.MARKDOWN)
    elif data.startswith("personality_"):
        session = get_or_create_session(query.from_user.id)
        new_personality = data.split("_", 1)[1]
        session.change_personality(new_personality)
        await query.edit_message_text(f"🎭 Personality: **{new_personality.title()}**.", parse_mode=ParseMode.MARKDOWN)
    elif data == "new_chat":
        session = get_or_create_session(query.from_user.id)
        session.clear_chat()
        await query.edit_message_text(f"🔄 Chat cleared!")
    elif data == "change_model": await model_command(query, context)
    elif data == "change_personality": await personality_command(query, context)
    elif data == "set_key": await set_key_command(query, context)
    elif data == "stats": await stats_command(query, context)
    elif data == "help": await help_command(query, context)
    elif data == "cancel_selection": await query.edit_message_text("❌ Cancelled.")

async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start", "▶️ Start"),
        BotCommand("menu", "📱 Menu"),
        BotCommand("newchat", "🔄 Reset"),
        BotCommand("roleplay", "🎭 Roleplay"),
        BotCommand("imagine", "🎨 Image"),
        BotCommand("undo", "↩️ Undo"),
        BotCommand("help", "❓ Help")
    ])

def main() -> None:
    # 1. Start the Flask Web Server in a separate thread
    # This prevents Render from killing the app for not binding to a port.
    logger.info("Starting Web Server for Render keep-alive...")
    threading.Thread(target=run_web_server, daemon=True).start()

    # 2. Initialize Database
    init_db()

    # 3. Start Telegram Bot
    if not TELEGRAM_BOT_TOKEN:
        logger.error("No TELEGRAM_BOT_TOKEN found. Exiting.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    
    # Handlers
    handlers = [
        CommandHandler("start", start), CommandHandler("menu", menu),
        CommandHandler("setkey", set_key_command), CommandHandler("personality", personality_command),
        CommandHandler("roleplay", roleplay_command), CommandHandler("imagine", imagine_command), 
        CommandHandler("help", help_command), CommandHandler("model", model_command), 
        CommandHandler("newchat", new_chat_command), CommandHandler("undo", undo_command),
        CommandHandler("stats", stats_command), 
        CallbackQueryHandler(button_callback),
        MessageHandler(filters.PHOTO | filters.Document.ALL, handle_file),
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    ]
    application.add_handlers(handlers)
    
    # Register Error Handler
    application.add_error_handler(error_handler)

    logger.info("Bot is polling...")
    # allowed_updates explicitly set to handle generic updates
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()