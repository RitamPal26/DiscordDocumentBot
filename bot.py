# bot.py

import discord
from discord import app_commands
from discord.ext import commands
from discord.ui import Button, View
from dotenv import load_dotenv
import os
import asyncio
import logging
import re
from urllib.parse import urlparse

# Import your RAG pipeline
from rag_pipeline import EnhancedRAGPipeline, Config
import db_manager

# --- Initial Setup ---
load_dotenv()
db_manager.init_db()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Bot Configuration ---
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DECISION_LOG_CHANNEL = os.getenv('DECISION_LOG_CHANNEL', 'project-decisions')
RIFT_API_KEY = os.getenv("RIFT_API_KEY")

if not DISCORD_TOKEN or not RIFT_API_KEY:
    raise ValueError("DISCORD_TOKEN and RIFT_API_KEY must be in your .env file.")

# Initialize RAG pipeline
config = Config()
rag_pipeline_instance = EnhancedRAGPipeline(config)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Global state management
learning_in_progress = False
current_documentation_domain = ""

# --- Helper Functions ---
def create_embed(title: str, description: str, color: discord.Color) -> discord.Embed:
    """Creates a standardized Discord embed."""
    embed = discord.Embed(title=title, description=description, color=color)
    embed.set_footer(text="Powered by Firecrawl & CloudRift")
    return embed

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# --- The Main Interactive View Class ---
class PlanView(View):
    def __init__(self, tasks=None, message_id=None):
        super().__init__(timeout=None)
        if tasks and message_id:
            # Create a set of buttons for each task
            for original_id, description in tasks:
                unique_task_id = f"{message_id}-{original_id}"
                
                self.add_item(Button(label=f"Claim {original_id}", style=discord.ButtonStyle.secondary, emoji="👀", custom_id=f"claim:{unique_task_id}"))
                self.add_item(Button(label="Complete", style=discord.ButtonStyle.success, emoji="✅", custom_id=f"complete:{unique_task_id}"))
                self.add_item(Button(label="Help", style=discord.ButtonStyle.primary, emoji="❓", custom_id=f"help:{unique_task_id}"))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        custom_id = interaction.data['custom_id']
        action, unique_task_id = custom_id.split(":", 1)

        if action == "claim":
            await self.handle_claim(interaction, unique_task_id)
        elif action == "complete":
            await self.handle_complete(interaction, unique_task_id)
        elif action == "help":
            await self.handle_help(interaction, unique_task_id)
        
        return False

    async def handle_claim(self, interaction: discord.Interaction, task_id: str):
        db_manager.update_task_status(task_id, 'in_progress', interaction.user.id)
        await interaction.response.send_message(f"You've claimed task `{task_id.split('-')[1]}`!", ephemeral=True, delete_after=5)
        await self.update_plan_message(interaction)

    async def handle_complete(self, interaction: discord.Interaction, task_id: str):
        db_manager.update_task_status(task_id, 'completed', interaction.user.id)
        await interaction.response.send_message(f"You've completed task `{task_id.split('-')[1]}`! 🎉", ephemeral=True, delete_after=5)
        await self.update_plan_message(interaction)

    async def handle_help(self, interaction: discord.Interaction, task_id: str):
        await interaction.response.defer(thinking=True, ephemeral=True)
        task_description = db_manager.get_task_description(task_id)
        if not task_description:
            await interaction.followup.send("I couldn't find this task.", ephemeral=True)
            return

        # This context will be invalid until the 404 error is fixed
        context = await asyncio.to_thread(rag_pipeline_instance.query_rag, f"How to implement: {task_description}")
        guidance = await asyncio.to_thread(rag_pipeline_instance.generate_guidance, task_description, context)
    
        # <<< FIX: Split long guidance messages into chunks >>>
        if len(guidance) > 1950:
            chunks = [guidance[i:i + 1950] for i in range(0, len(guidance), 1950)]
            await interaction.followup.send(f"### 💡 Guidance for `{task_id.split('-')[1]}`\n{chunks[0]}", ephemeral=True)
            for chunk in chunks[1:]:
                # Send subsequent chunks in new followup messages
                await interaction.followup.send(chunk, ephemeral=True)
        else:
            await interaction.followup.send(f"### 💡 Guidance for `{task_id.split('-')[1]}`\n{guidance}", ephemeral=True)

    async def update_plan_message(self, interaction: discord.Interaction):
        try:
            message_id_match = re.search(r"`(\d+)`", interaction.message.content)
            if not message_id_match:
                return
            message_id = int(message_id_match.group(1))
        except (AttributeError, IndexError, TypeError):
            logger.error("Could not parse message_id from control message content.")
            return

        original_plan_text, tasks = db_manager.get_plan_and_tasks(message_id)
        
        status_map = {task[0]: (task[2], task[3]) for task in tasks}
        
        task_regex = re.compile(r"-\s*\[\s*\]\s*\((TID-[FB]\d+)\)\s*(.*)")
        updated_lines = []
        for line in original_plan_text.split('\n'):
            match = task_regex.search(line)
            if match:
                original_id, description = match.groups()
                status, assignee_id = status_map.get(original_id, ('pending', None))
                
                if status == 'in_progress':
                    prefix = "👀"
                    assignee_mention = f" (Claimed by <@{assignee_id}>)" if assignee_id else ""
                    updated_lines.append(f"- {prefix} ~`({original_id}) {description.strip()}`~{assignee_mention}")
                elif status == 'completed':
                    prefix = "✅"
                    assignee_mention = f" (Completed by <@{assignee_id}>)" if assignee_id else ""
                    updated_lines.append(f"- {prefix} ~~`({original_id}) {description.strip()}`~~{assignee_mention}")
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        try:
            plan_message = await interaction.channel.fetch_message(message_id)
            await plan_message.edit(content='\n'.join(updated_lines))
        except discord.NotFound:
            logger.error(f"Could not find original plan message with ID {message_id} to update.")
        except discord.Forbidden:
            logger.error(f"Missing permissions to edit message {message_id}.")


# --- Bot Setup and Events ---
@bot.event
async def on_ready():
    print(f"🤖 BluePrint is online! Logged in as {bot.user}")
    
    if not hasattr(bot, 'persistent_views_added'):
        bot.add_view(PlanView())
        bot.persistent_views_added = True
        print("✅ Persistent view registered.")

    try:
        synced = await bot.tree.sync()
        print(f"⚡ Synced {len(synced)} command(s).")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

# --- Bot Commands ---
class MyBot:
    tree = None # Placeholder for the app_commands.CommandTree
# bot = MyBot() 

@bot.tree.command(name="help", description="Shows a complete guide on how to use the bot")
async def help(interaction: discord.Interaction):
    """Shows a complete guide on how to use the bot."""

    # 1. Create the main embed object
    # The color can be customized to your bot's branding.
    embed = discord.Embed(
        title="🚀 BluePrint Help Guide",
        description=(
            "Hello! I'm BluePrint, your AI-powered project assistant. "
            "I help you generate technical plans, learn from documentation, "
            "and manage your project knowledge base right here in Discord."
        ),
        color=discord.Color.from_rgb(52, 152, 219) # A nice blue color
    )

    # 2. Add the Core Workflow section
    # This gives users a quick start guide.
    embed.add_field(
        name="🎯 Core Workflow",
        value=(
            "1. **Plan a Feature**: Use `/plan` to get a detailed, step-by-step technical plan for any feature you can describe.\n"
            "2. **Build Your Knowledge**: Use `/learn-docs` to have me read and understand a documentation website.\n"
            "3. **Save Important Info**: Mention me (`@BluePrint`) in any message to save it directly to your knowledge base."
        ),
        inline=False
    )

    # 3. Add the Commands section
    # Using `>` (blockquote) and `*Example:*` makes the command explanations clearer.
    embed.add_field(
        name="Available Commands",
        value=(
            "**/plan `[feature]`**\n"
            "> Creates a detailed, step-by-step technical plan for a new feature.\n"
            "> *Example: `/plan create a user authentication system using OAuth`*\n\n"
            
            "**/learn-docs `[url]`**\n"
            "> Ingests and learns from a public documentation website.\n"
            "> *Example: `/learn-docs https://discordpy.readthedocs.io/en/stable/`*\n\n"
            
            "**/add-docs `[url]`**\n"
            "> Adds a new documentation website to your existing knowledge base.\n\n"
            
            "**/status**\n"
            "> Checks the current status of your knowledge base.\n\n"

            "**/help**\n"
            "> Shows this guide."
        ),
        inline=False
    )

    # 4. Add the Special Features section
    # Highlighting the @mention functionality is important as it's not a slash command.
    embed.add_field(
        name="✨ Special Features",
        value=(
            "**Quick Save**: Simply **@BluePrint** at the start of any message to save its content to your knowledge base. "
            "This is perfect for saving code snippets, important decisions, or team notes without a formal command."
        ),
        inline=False
    )

    # 5. Set a footer
    embed.set_footer(text="BluePrint | Your AI Project Partner")

    # 6. Send the response
    # ephemeral=True makes the message visible only to the user who ran the command.
    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="plan", description="Generate a technical plan for a new feature.")
@app_commands.describe(feature_request="Describe the feature you want to build.")
async def plan(interaction: discord.Interaction, feature_request: str):
    await interaction.response.defer(thinking=True)

    technical_plan = rag_pipeline_instance.generate_plan(feature_request)

    task_regex = re.compile(r"-\s*\[\s*\]\s*\((TID-[FB]\d+)\)\s*(.*)")
    parsed_tasks = task_regex.findall(technical_plan)

    if not parsed_tasks:
        await interaction.followup.send("The generated plan didn't contain any actionable tasks. Please try again.")
        return

    # <<< FIX: Split the plan into chunks if it's too long >>>
    plan_message = None
    if len(technical_plan) <= 2000:
        plan_message = await interaction.followup.send(technical_plan)
    else:
        plan_chunks = [technical_plan[i:i + 1990] for i in range(0, len(technical_plan), 1990)]
        for i, chunk in enumerate(plan_chunks):
            if i == 0:
                plan_message = await interaction.followup.send(chunk)
            else:
                await interaction.channel.send(chunk)
    # <<< END FIX >>>

    db_manager.add_plan(plan_message.id, plan_message.channel.id, feature_request, technical_plan)

    for original_id, description in parsed_tasks:
        unique_task_id = f"{plan_message.id}-{original_id}"
        db_manager.add_task(unique_task_id, plan_message.id, original_id, description)

    task_chunks = list(chunks(parsed_tasks, 4))

    for i, chunk in enumerate(task_chunks):
        view = PlanView(tasks=chunk, message_id=plan_message.id)
        target = plan_message.thread or interaction.channel
        await target.send(f"**Interactive Controls for Plan `{plan_message.id}` (Part {i+1})**", view=view)

# ... (keep your existing learn-docs, add-docs, on_message, and status commands) ...
@bot.tree.command(name="learn-docs", description="Deletes old data and learns from a new documentation URL")
async def learn_docs(interaction: discord.Interaction, url: str):
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
    await interaction.response.defer(ephemeral=True)
    learning_in_progress = True
    domain = urlparse(url).netloc
    current_documentation_domain = domain
    try:
        start_embed = create_embed("🚀 Learning Started", f"Replacing my knowledge base with docs from `{domain}`...", discord.Color.blue())
        await interaction.followup.send(embed=start_embed)
        success = await asyncio.to_thread(rag_pipeline_instance.create_vector_store, url)
        if success:
            kb_info = rag_pipeline_instance.get_knowledge_base_info()
            count = kb_info.get("count", 0)
            success_embed = create_embed("✅ Learning Complete!", f"Successfully learned from `{domain}`. My knowledge base is now ready with **{count}** documents.", discord.Color.green())
            await interaction.followup.send(embed=success_embed)
        else:
            fail_embed = create_embed("❌ Learning Failed", f"Failed to learn from `{domain}`. Please check the URL and my console logs.", discord.Color.red())
            await interaction.followup.send(embed=fail_embed)
    except Exception as e:
        logger.error(f"Error during learning: {e}", exc_info=True)
        error_embed = create_embed("💥 An Unexpected Error Occurred", "Something went wrong. Please check the logs.", discord.Color.dark_red())
        await interaction.followup.send(embed=error_embed)
    finally:
        learning_in_progress = False

@bot.tree.command(name="add-docs", description="Adds new documentation to the existing knowledge base")
async def add_docs(interaction: discord.Interaction, url: str):
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
    await interaction.response.defer(ephemeral=True)
    learning_in_progress = True
    domain = urlparse(url).netloc
    current_documentation_domain = domain
    try:
        start_embed = create_embed("➕ Adding Knowledge", f"Adding documents from `{domain}` to my knowledge base...", discord.Color.blue())
        await interaction.followup.send(embed=start_embed)
        success = await asyncio.to_thread(rag_pipeline_instance.add_to_vector_store, url)
        if success:
            kb_info = rag_pipeline_instance.get_knowledge_base_info()
            count = kb_info.get("count", 0)
            success_embed = create_embed("✅ Knowledge Added!", f"Successfully added documents from `{domain}`. I now have **{count}** total documents.", discord.Color.green())
            await interaction.followup.send(embed=success_embed)
        else:
            fail_embed = create_embed("❌ Add Failed", f"Failed to add documents from `{domain}`. Please check the URL and my console logs.", discord.Color.red())
            await interaction.followup.send(embed=fail_embed)
    except Exception as e:
        logger.error(f"Error while adding docs: {e}", exc_info=True)
        error_embed = create_embed("💥 An Unexpected Error Occurred", "Something went wrong. Please check the logs.", discord.Color.dark_red())
        await interaction.followup.send(embed=error_embed)
    finally:
        learning_in_progress = False

@bot.event
async def on_message(message):
    if message.author == bot.user or bot.user not in message.mentions:
        return
    if "save this" in message.content.lower():
        if not message.reference or not message.reference.message_id:
            await message.reply("To save a decision, please use this command as a **reply** to the message you want to archive.", delete_after=15)
            return
        try:
            original_message = await message.channel.fetch_message(message.reference.message_id)
            archive_channel = discord.utils.get(message.guild.text_channels, name=DECISION_LOG_CHANNEL)
            if not archive_channel:
                await message.reply(f"I couldn't find the `#{DECISION_LOG_CHANNEL}` channel. An admin may need to create it.", delete_after=15)
                return
            decision_embed = discord.Embed(title="🎯 Decision Logged", description=f">>> {original_message.content}", color=discord.Color.green(), timestamp=original_message.created_at)
            decision_embed.set_author(name=original_message.author.display_name, icon_url=original_message.author.avatar.url if original_message.author.avatar else None)
            decision_embed.add_field(name="Logged By", value=message.author.mention, inline=True)
            decision_embed.add_field(name="Original Context", value=f"[Jump to Message]({original_message.jump_url})", inline=True)
            decision_embed.set_footer(text=f"From #{original_message.channel.name}")
            await archive_channel.send(embed=decision_embed)
            await message.add_reaction("✅")
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
            await message.add_reaction("❌")
        return
    question = _extract_question(message.content, bot.user.id)
    if not question:
        embed = create_embed("👋 Hello!", f"Please ask me a question after the mention.\n\n**Example**: `@{bot.user.display_name} How do I get started?`", discord.Color.blue())
        await message.reply(embed=embed)
        return
    if not rag_pipeline_instance.has_knowledge_base():
        embed = create_embed("📚 No Knowledge Base", "My knowledge base is empty. An admin needs to use `/learn-docs` to teach me.", discord.Color.orange())
        await message.reply(embed=embed)
        return
    async with message.channel.typing():
        try:
            answer = await asyncio.to_thread(rag_pipeline_instance.query_rag, question)
            if len(answer) > 1900:
                chunks = _split_long_message(answer)
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        embed = create_embed(f"📖 Answer for: \"{question[:50]}...\"", chunk, discord.Color.dark_grey())
                        await message.reply(embed=embed)
                    else:
                        await message.channel.send(chunk)
            else:
                embed = create_embed(f"📖 Answer for: \"{question[:50]}...\"", answer, discord.Color.dark_grey())
                await message.reply(embed=embed)
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            embed = create_embed("💥 Processing Error", "Sorry, an error occurred. Please try rephrasing or check the logs.", discord.Color.red())
            await message.reply(embed=embed)

@bot.tree.command(name="status", description="Check bot status and current documentation")
async def status(interaction: discord.Interaction):
    if learning_in_progress:
        title = "🔄 Learning in Progress"
        status_msg = f"I'm currently processing docs from `{current_documentation_domain}`."
        color = discord.Color.blue()
    elif rag_pipeline_instance.has_knowledge_base():
        kb_info = rag_pipeline_instance.get_knowledge_base_info()
        count = kb_info.get("count", 0)
        sources = kb_info.get("sources", [])
        title = "✅ Ready to Help!"
        status_msg = f"My knowledge base is loaded with **{count}** documents"
        if sources:
            source_text = ", ".join([f"**{source}**" for source in sources])
            status_msg += f" from: {source_text}."
        color = discord.Color.green()
    else:
        title = "📚 No Documentation Loaded"
        status_msg = "Use `/learn-docs <url>` to get started!"
        color = discord.Color.orange()
    embed = create_embed(title, status_msg, color)
    await interaction.response.send_message(embed=embed, ephemeral=True)

# --- Utility Functions ---
def _is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False

def _extract_question(message_content: str, bot_user_id: int) -> str:
    mention_pattern = f"<@!?{bot_user_id}>"
    cleaned_content = re.sub(mention_pattern, "", message_content).strip()
    if "save this" in cleaned_content.lower():
        return ""
    return ' '.join(cleaned_content.split())

def _split_long_message(text: str, max_length: int = 2000) -> list:
    chunks = []
    while len(text) > max_length:
        split_point = text.rfind('\n\n', 0, max_length)
        if split_point == -1: split_point = text.rfind('\n', 0, max_length)
        if split_point == -1: split_point = text.rfind('. ', 0, max_length)
        if split_point == -1: split_point = max_length
        chunks.append(text[:split_point])
        text = text[split_point:].lstrip()
    chunks.append(text)
    return chunks

# --- Main Execution ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ Error: DISCORD_TOKEN not found.")
    else:
        try:
            print("🚀 Starting the AI Tech Lead bot...")
            bot.run(DISCORD_TOKEN)
        except discord.LoginFailure:
            print("❌ Error: Invalid Discord token. Please check your .env file.")
        except Exception as e:
            print(f"❌ Error starting bot: {e}")