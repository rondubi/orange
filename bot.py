import os
import discord
import logging

from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent

PREFIX = "!"

# Setup logging
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Import the Mistral agent from the agent.py file
agent = MistralAgent()

# Initialize as None first
last_channel = None

# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")
if token is None:
    raise ValueError("No discord token found")

STARTING_PROMPT = """
🎉 WELCOME TO MEME BOT! 🎉

🟢 Meme Bot is now online and ready to create memes!
"""

DISCONNECT_PROMPT = """
🔴 Meme Bot is going offline. See you later!
"""

INSUFFICIENT_MESSAGE = """
🟡 I'm assuming you didn't want to create a meme here, but if you did, please indicate so!
"""

@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
    """
    global last_channel
    logger.info(f"{bot.user} has connected to Discord!")
    
    # Find the "orange" channel in any of the guilds the bot is in
    for guild in bot.guilds:
        channel = discord.utils.get(guild.text_channels, name='orange')
        if channel:
            last_channel = channel
            await last_channel.send(STARTING_PROMPT)
            break

@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_message
    """
    global last_channel
    last_channel = message.channel  # Update the active channel
    
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops.
    if message.author.bot or message.content.startswith("!"):
        return

    # Process the message with the agent you wrote
    # Open up the agent.py file to customize the agent
    logger.info(f"Processing message from {message.author}: {message.content}")
    response = await agent.run(message)

    if response == "INSUFFICIENT_MESSAGE":
        await message.reply(INSUFFICIENT_MESSAGE)
        return

    if response is None:
        return

    # Send the response back to the channel
    file = discord.File(response, filename = os.path.basename(response))
    embed = discord.Embed()
    embed.set_image(url = f"attachment://{os.path.basename(response)}")
    await message.reply(file = file, embed = embed)

    # Clean up file
    os.remove(response)

# Commands

@bot.event
async def on_disconnect():
    """Called when the bot disconnects from Discord"""
    if last_channel:
        try:
            await last_channel.send(DISCONNECT_PROMPT)
        except:
            pass

@bot.event
async def insufficient_message():
    """Called when the user provides an insufficient message from Discord"""
    if last_channel:
        try:
            await last_channel.send(INSUFFICIENT_MESSAGE)
        except:
            pass

# This example command is here to show you how to add commands to the bot.
# Run !ping with any number of arguments to see the command in action.
# Feel free to delete this if your project will not need commands.
@bot.command(name="ping", help="Pings the bot.")
async def ping(ctx, *, arg=None):
    if arg is None:
        await ctx.send("Pong!")
    else:
        await ctx.send(f"Pong! Your argument was {arg}")

# Start the bot, connecting it to the gateway
bot.run(token)
