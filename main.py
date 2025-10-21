import asyncio
from discord.ext import commands
import discord
import logging


logging.basicConfig(
	filename='logs.txt',  # Name of the log file
	filemode='a',        # 'a' for append (default), 'w' to overwrite each run
	encoding='utf-8',
	level=logging.INFO,  # Minimum level of messages to log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
	format='[%(asctime)s] %(levelname)s: %(message)s' # Format of log messages
)


if __name__ == "__main__":
	intents = discord.Intents.default()
	intents.message_content = True

	bot = commands.Bot(command_prefix=".", intents=intents)

	with open("BOT_TOKEN", "r") as f:
		BOT_TOKEN = f.read()
		
	@bot.event
	async def on_ready():
		logging.info("Logged in as %s (ID: %d)", bot.user, bot.user.id)
		logging.info("------")

	async def main():
		await bot.load_extension("cogs.image_police")
		await bot.start(BOT_TOKEN)
	
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		logging.info("Bot disconnected!")