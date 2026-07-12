import os
import io
import base64
import discord
import cv2
import numpy as np
import json
import filetype
from discord.ext import commands
from anthropic import Anthropic
from dotenv import load_dotenv

# Initialize APIs using Environment Variables for safety
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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


def detect_media_type(image_bytes: bytes) -> str:
    """Detect image media type from file signature bytes."""
    kind = filetype.guess(image_bytes)
    if kind is None or kind.mime not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        raise ValueError("Could not detect a supported image format after encoding.")
    return kind.mime
       
def process_image_with_claude(image_bytes: bytes, media_type: str) -> str:
    """Encodes raw image bytes and sends them to Claude 4.5 Haiku."""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
    Analyze this end-of-match scoreboard from the video game THE FINALS.
    Extract the stats for all visible players across all teams.
    It is critical that the team names and usernames are parsed correctly. A name you will see often is CHEEMSBURBGER, not CHEEMSBURGER.
    The gamemode is at the top-left of the screenshot. Typically this is either "HEAD2HEAD", "FINAL ROUND" or "QUICK CASH".
    The single letter before a username which is either "H", "M" or "L" indicates the class, and can be ignored.
    A username may also have a clan tag before it, which is typically 3-5 small characters in a box, can also be ignored.
    Each username has a hashtag and then a 4-digit ID at the end. This can also be ignored.
    The winning team is the team with the number 1 to the left of the scoreboard.
    After usernames, the scoreboard from left to right has the following columns: E (eliminations), A (assists), D (deaths), R (revives), coins (IGNORE THIS COLUMN), combat score, support score, objective score.
    Return ONLY a valid, raw JSON object matching this schema. Remove the "```json" at the beginning and the "```" at the end.
    {
        "gamemode": "string"
        "winning_team": "string"
        "match_results": [
            {
            "team_name": "string",
            "players": [
                    {
                        "username": "string",
                        "eliminations": integer,
                        "assists": integer,
                        "deaths": integer,
                        "revives": integer
                        "combat_score": integer,
                        "support_score": integer,
                        "objective_score": integer,
                    }
                ]
            }
        ]
    }
    """

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image
                        }
                    },
                    {"type": "text", "text": prompt}
                ],
            }
        ],
    )
    return response.content[0].text

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
            if img is not None:
                # Convert to grayscale
                h, w, _ = img.shape
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Increase contrast / thresholding to make the text pure white, background pure black
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            else:
                await ctx.send(f"An error occurred while decoding the image.")
                return

            success, encoded_image = cv2.imencode('.png', thresh)
            if not success:
                await ctx.send(f"An error occurred while encoding the processed image.")                
                return

            encoded_bytes = encoded_image.tobytes()
            encoded_media_type = detect_media_type(encoded_bytes)
            
            # Send to Claude
            json_output = process_image_with_claude(encoded_bytes, encoded_media_type)

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
