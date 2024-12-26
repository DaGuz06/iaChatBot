from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configura el modelo gratuito
model_name = "distilgpt2"  # Modelo ligero y gratuito
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

async def start(update: Update, context):
    await update.message.reply_text("¡Hola! Soy un chatbot personal. Hazme una pregunta.")

async def respond(update: Update, context):
    user_message = update.message.text
    inputs = tokenizer.encode(user_message, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    await update.message.reply_text(response)

# Configura la aplicación de Telegram
app = ApplicationBuilder().token("7403521128:AAHT461KOM7NxjYvbIx881vr_wePS2NJAqk").build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))

app.run_polling()
