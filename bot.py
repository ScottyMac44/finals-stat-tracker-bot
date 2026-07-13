import os
import io
import base64
import discord
import cv2
import numpy as np
import json
import filetype
from discord.ext import commands
from ai import process_image_with_claude
from dotenv import load_dotenv

# Get Discord bot token from env
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Setup Discord bot intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

class StatsVerificationView(discord.ui.View):
    def __init__(self, original_author):
        super().__init__(timeout=300) # 5-minute timeout
        self.original_author = original_author

    @discord.ui.button(label="Data is Correct", style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Ensure only the original user can click the button
        if interaction.user != self.original_author:
            await interaction.response.send_message("Only the uploader can confirm this data.", ephemeral=True)
            return

        # Disable buttons after click
        for child in self.children:
            child.disabled = True
        
        await interaction.response.edit_message(content="Data confirmed! Sending off to the database...", view=self)
        
        # TODO: Implement your database logic here

    @discord.ui.button(label="Needs Correction", style=discord.ButtonStyle.red)
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Ensure only the original user can click the button
        if interaction.user != self.original_author:
            await interaction.response.send_message("Only the uploader can reject this data.", ephemeral=True)
            return

        # Disable buttons after click
        for child in self.children:
            child.disabled = True
            
        await interaction.response.edit_message(content="Data marked as incorrect. Initiating manual correction...", view=self)
        
        # TODO: Implement your manual correction logic here

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} ({bot.user.id})")

@bot.command(name="stats")
async def extract_stats(ctx):
    """Trigger this command by typing !stats and attaching a screenshot."""

    # Check if the user actually attached an image
    if not ctx.message.attachments:
        await ctx.send("Please attach a scoreboard screenshot when using this command!")
        return
    attachment = ctx.message.attachments[0]

    # Check if the attachment is of the right file type
    allowed_filetypes = {"image/png", "image/jpeg", "image/webp"}
    if not attachment.content_type in allowed_filetypes:
        await ctx.send("Unsupported file format! Please use PNG, JPEG or WEBP.")
        return
    
    # Let the user know the bot is working
    async with ctx.typing():
        try:
            # Download image straight into memory
            image_bytes = await attachment.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                await ctx.send(f"An error occurred while decoding the image.")
                return
            
            # Convert to grayscale
            h, w, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Increase contrast / thresholding to make the text pure white, background pure black
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # Encode image from numpy array into .png format and store in memory
            success, encoded_image = cv2.imencode('.png', thresh)
            if not success:
                await ctx.send(f"An error occurred while encoding the processed image.")                
                return

            encoded_bytes = encoded_image.tobytes()
            
            # Send to Claude
            json_output = process_image_with_claude(encoded_bytes)

            json_file = discord.File(io.BytesIO(json_output.encode('utf-8')), filename="match_stats.json")

            view = StatsVerificationView(ctx.author)

            await ctx.send(
                content=f"{ctx.author.mention}, here is the parsed data. Please verify if it is correct:",
                file=json_file,
                view=view
            )

        except Exception as e:
            await ctx.send(f"An error occurred while processing the image: {str(e)}")

# Start the bot
bot.run(TOKEN)
