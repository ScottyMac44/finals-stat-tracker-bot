import os
import io
import base64
import discord
import cv2
import numpy as np
import json
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
       
def process_image_with_claude(image_bytes: bytes, media_type: str) -> str:
    """Encodes raw image bytes and sends them to Claude 4.5 Haiku."""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
    Analyze this end-of-match scoreboard from the video game THE FINALS.
    Extract the stats for all visible players across all teams.
    It is critical that the team names and usernames are parsed correctly.
    The single letter before a username which is either "H", "M" or "L" indicates the class, and can be ignored.
    A username may also have a clan tag before it, which is typically 3-5 small characters in a box, can also be ignored.
    Each username has a hashtag and then a 4-digit ID at the end. This can also be ignored.
    The winning team is the team with the number 1 to the left of the scoreboard.
    Return ONLY a valid, raw JSON object matching this schema. Do not wrap it in markdown block quotes.
    {
        "winning_team": "string"
        "match_results": [
            {
            "team_name": "string",
            "players": [
                    {
                        "username": "string",
                        "combat_score": integer,
                        "support_score": integer,
                        "objective_score": integer,
                        "eliminations": integer,
                        "assists": integer,
                        "deaths": integer,
                        "revives": integer
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
                # 1. Crop to the scoreboard area (adjust bounding box ratios for your resolution)
                h, w, _ = img.shape
                scoreboard_crop = img[int(h*0):int(h*0.8), int(w*0):int(w*0.7)]
                
                # 2. Convert to grayscale
                gray = cv2.cvtColor(scoreboard_crop, cv2.COLOR_BGR2GRAY)
                
                # 3. Increase contrast / thresholding to make the text pure white, background pure black
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            else:
                await ctx.send(f"An error occurred while decoding the image.")

            success, encoded_image = cv2.imencode('.png', img)
            if not success:
                await ctx.send(f"An error occurred while encoding the processed image.")                
                return
            
            # Send to Claude
            json_output = process_image_with_claude(encoded_image.tobytes(), "image/png")

            # Send JSON output back to Discord channel inside a code block
            # Note: Discord has a 2000 character limit per message, a standard JSON will easily fit.
            await ctx.send(f"```json\n{json_output}\n```")

        except Exception as e:
            await ctx.send(f"An error occurred while processing the image: {str(e)}")

# Start the bot
bot.run(TOKEN)
