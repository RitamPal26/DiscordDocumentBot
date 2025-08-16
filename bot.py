# bot.py

import discord
from discord.ext import commands
import os
import asyncio
import logging
from dotenv import load_dotenv
import re
from typing import Optional
from urllib.parse import urlparse

# Import your RAG pipeline
from rag_pipeline import EnhancedRAGPipeline, Config

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN not found in environment variables. Please add it to your .env file.")

# Initialize RAG pipeline
config = Config()
rag_pipeline = EnhancedRAGPipeline(config)

# Bot setup with necessary intents
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Global state management
learning_in_progress = False
current_documentation_domain = ""

# <<< IMPROVEMENT: Helper function for creating consistent embeds
def create_embed(title: str, description: str, color: discord.Color) -> discord.Embed:
    """Creates a standardized Discord embed."""
    embed = discord.Embed(title=title, description=description, color=color)
    embed.set_footer(text="Powered by Firecrawl & Ollama")
    return embed

@bot.event
async def on_ready():
    """Startup confirmation event."""
    print(f"🤖 Bot is ready and online! Logged in as {bot.user}")
    print(f"📊 Connected to {len(bot.guilds)} server(s)")
    
    if rag_pipeline.has_knowledge_base():
        print(f"📚 {rag_pipeline.get_knowledge_base_info()}")
    else:
        print("📚 No existing knowledge base found")
    
    try:
        synced = await bot.tree.sync()
        print(f"⚡ Synced {len(synced)} command(s)")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

# <<< IMPROVEMENT: Added a central /help command
@bot.tree.command(name="help", description="Show all available commands and how to use them")
async def help(interaction: discord.Interaction):
    """Displays a help message with all commands."""
    embed = create_embed(
        "🤖 How to Use the Docs Bot",
        "Here are the commands you can use to interact with me:",
        discord.Color.blue()
    )
    embed.add_field(
        name="`/learn-docs [url]`",
        value="Teach me from a documentation website. I'll crawl the site and build a persistent knowledge base.",
        inline=False
    )
    embed.add_field(
        name="`@me [your question]`",
        value="Once I've learned from a doc, mention me and ask a question to get an answer based on the context.",
        inline=False
    )
    embed.add_field(
        name="`/status`",
        value="Check my current status to see if I'm learning or what documentation I have loaded.",
        inline=False
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="learn-docs", description="Learn from a documentation URL to answer questions")
async def learn_docs(interaction: discord.Interaction, url: str):
    """Slash command to learn from documentation using Firecrawl."""
    global learning_in_progress, current_documentation_domain
    
    url = url.strip()
    if not _is_valid_url(url):
        embed = create_embed("❌ Invalid URL", "Please provide a valid HTTP/HTTPS URL.", discord.Color.red())
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    if learning_in_progress:
        embed = create_embed("🤔 Already Learning", "I'm already busy learning. Please wait for that to complete.", discord.Color.orange())
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return
    
    await interaction.response.defer()
    
    learning_in_progress = True
    domain = urlparse(url).netloc
    current_documentation_domain = domain
    
    try:
        start_embed = create_embed(
            "🚀 Learning Started",
            f"I'm now learning from `{domain}` using Firecrawl for fast scraping.\n\n*This may take 30-60 seconds...*",
            discord.Color.blue()
        )
        await interaction.followup.send(embed=start_embed)
        
        success = await asyncio.to_thread(rag_pipeline.create_vector_store, url)
        
        if success:
            knowledge_info = rag_pipeline.get_knowledge_base_info()
            success_embed = create_embed(
                "✅ Learning Complete!",
                f"Successfully learned from `{domain}`. The knowledge base is now persistent and ready!\n\n**{knowledge_info}**",
                discord.Color.green()
            )
            success_embed.add_field(name="How to Ask", value=f"Just mention me with your question, like `@{bot.user.display_name} How do I get started?`")
            await interaction.followup.send(embed=success_embed)
        else:
            fail_embed = create_embed(
                "❌ Learning Failed",
                f"Failed to learn from `{domain}`. Please check that the URL is a valid and accessible documentation site.",
                discord.Color.red()
            )
            await interaction.followup.send(embed=fail_embed)
            
    except Exception as e:
        logger.error(f"Error during learning: {e}")
        error_embed = create_embed("💥 An Unexpected Error Occurred", f"```{str(e)[:1000]}```", discord.Color.dark_red())
        await interaction.followup.send(embed=error_embed)
    finally:
        learning_in_progress = False

@bot.event
async def on_message(message):
    """Handle incoming messages and respond to mentions."""
    if message.author == bot.user or bot.user not in message.mentions:
        return
    
     # --- Start of New "Decision Logger" Feature ---
    # Check if the bot was mentioned and the trigger phrase "save this" is in the message
    if bot.user in message.mentions and "save this" in message.content.lower():
        
        # 1. Ensure this is a reply to another message
        if not message.reference or not message.reference.message_id:
            await message.reply("To save a decision, please use this command as a **reply** to the message you want to archive.", ephemeral=True, delete_after=15)
            return

        try:
            # 2. Fetch the original message that is being replied to
            original_message = await message.channel.fetch_message(message.reference.message_id)

            # 3. Find the dedicated archive channel
            channel_name = os.getenv('DECISION_LOG_CHANNEL', 'project-decisions')
            archive_channel = discord.utils.get(message.guild.text_channels, name=channel_name)

            if not archive_channel:
                await message.reply(f"I couldn't find the archive channel `#{channel_name}`. Please create it or check the name in the config.", ephemeral=True, delete_after=15)
                return

            # 4. Create a rich embed to store the decision for clarity
            decision_embed = discord.Embed(
                title="🎯 Decision Logged",
                description=f"**Decision:**\n>>> {original_message.content}",
                color=discord.Color.green(),
                timestamp=original_message.created_at
            )
            decision_embed.set_author(name=original_message.author.display_name, icon_url=original_message.author.avatar.url)
            decision_embed.add_field(name="Logged By", value=message.author.mention, inline=True)
            decision_embed.add_field(name="Original Context", value=f"[Jump to Message]({original_message.jump_url})", inline=True)
            decision_embed.set_footer(text=f"From #{original_message.channel.name}")

            # 5. Send the embed to the archive channel and confirm to the user
            await archive_channel.send(embed=decision_embed)
            await message.add_reaction("✅") # Add a checkmark to the user's "save this" message

            # Stop processing here so it doesn't try to answer it as a question
            return 
            
        except discord.NotFound:
            await message.reply("I couldn't find the original message to save. It might have been deleted.", ephemeral=True, delete_after=15)
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
            await message.reply("Sorry, something went wrong while trying to save that decision.", ephemeral=True, delete_after=15)
            return
    
    if not rag_pipeline.has_knowledge_base():
        reply_description = (
            f"I'm currently learning from `{current_documentation_domain}`. Please try again in a moment."
            if learning_in_progress
            else "I haven't learned any documentation yet! An admin needs to use the `/learn-docs <url>` command first."
        )
        reply_title = "🔄 Still Learning..." if learning_in_progress else "📚 No Knowledge Base"
        reply_color = discord.Color.orange() if learning_in_progress else discord.Color.red()
        
        embed = create_embed(reply_title, reply_description, reply_color)
        await message.reply(embed=embed)
        return
    
    question = _extract_question(message.content, bot.user.id)
    if not question:
        embed = create_embed("🤷 Need a Question", f"Please mention me with a specific question!\n\n**Example**: `@{bot.user.display_name} How do I install this?`", discord.Color.orange())
        await message.reply(embed=embed)
        return
    
    async with message.channel.typing():
        try:
            answer = await asyncio.to_thread(rag_pipeline.query_rag, question)
            
            if len(answer) > 2000:
                chunks = _split_long_message(answer)
                for i, chunk in enumerate(chunks):
                    # <<< IMPROVEMENT: Create embed for the first chunk
                    if i == 0:
                        embed = create_embed(f"📖 Answer for: \"{question[:50]}...\"", chunk, discord.Color.dark_grey())
                        await message.reply(embed=embed)
                    else:
                        await message.channel.send(chunk) # Send subsequent chunks as plain text
            else:
                embed = create_embed(f"📖 Answer for: \"{question[:50]}...\"", answer, discord.Color.dark_grey())
                await message.reply(embed=embed)
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            embed = create_embed("💥 Processing Error", "Sorry, I encountered an error. Please try rephrasing your question.", discord.Color.red())
            await message.reply(embed=embed)

def _is_valid_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False

def _extract_question(message_content: str, bot_user_id: int) -> str:
    """
    Extract the question from a message that mentions the bot,
    ignoring the 'save this' command.
    """
    # This pattern correctly removes the bot's mention
    mention_pattern = f"<@!?{bot_user_id}>"
    cleaned_content = re.sub(mention_pattern, "", message_content).strip()
    
    # <<< CHANGE IS HERE
    # Check if the core content is the save command. If so, return an
    # empty string so it's not treated as a question for the RAG pipeline.
    if "save this" in cleaned_content.lower():
        return ""
    
    # Normalize spacing and return the clean question
    return ' '.join(cleaned_content.split())

def _split_long_message(text: str, max_length: int = 1980) -> list:
    """
    Intelligently splits a long message.
    - Tries to split by paragraphs first.
    - Then by sentences.
    - Finally, does a hard split by words if a single paragraph is too long.
    """
    chunks = []
    
    # Split by paragraphs to maintain formatting
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_length:
            chunks.append(current_chunk)
            current_chunk = ""
            
        if len(para) > max_length:
            # If a single paragraph is too long, split it by words
            words = para.split(' ')
            temp_para_chunk = ""
            for word in words:
                if len(temp_para_chunk) + len(word) + 1 > max_length:
                    chunks.append(temp_para_chunk)
                    temp_para_chunk = word
                else:
                    temp_para_chunk += f" {word}"
            current_chunk = temp_para_chunk.strip()
        else:
            current_chunk += f"{para}\n\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

@bot.tree.command(name="status", description="Check bot status and current documentation")
async def status(interaction: discord.Interaction):
    """Checks and reports the bot's current status."""
    status_msg = ""
    color = discord.Color.orange()

    if learning_in_progress:
        status_msg = f"I'm currently crawling and indexing `{current_documentation_domain}`."
        title = "🔄 Learning in Progress"
        color = discord.Color.blue()
    
    elif rag_pipeline.has_knowledge_base():
        # 1. Get the new, detailed knowledge base info
        kb_info = rag_pipeline.get_knowledge_base_info()
        count = kb_info.get("count", 0)
        sources = kb_info.get("sources", [])
        
        # 2. Build the descriptive status message
        status_msg = f"My knowledge base is loaded and persistent.\nKnowledge base loaded with **{count}** documents"
        
        # 3. If sources were found, add them to the message
        if sources:
            source_text = ", ".join([f"{source} documents" for source in sources])
            status_msg += f" which includes **{source_text}**."
        
        title = "✅ Ready to Help!"
        color = discord.Color.green()

    else:
        title = "📚 No Documentation Loaded"
        status_msg = "Use `/learn-docs <url>` to get started!"

    embed = create_embed(title, status_msg, color)
    await interaction.response.send_message(embed=embed, ephemeral=True)

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ Error: DISCORD_TOKEN not found.")
    else:
        try:
            print("🚀 Starting the RAG Discord bot...")
            bot.run(DISCORD_TOKEN)
        except discord.LoginFailure:
            print("❌ Error: Invalid Discord token. Please check your .env file.")
        except Exception as e:
            print(f"❌ Error starting bot: {e}")