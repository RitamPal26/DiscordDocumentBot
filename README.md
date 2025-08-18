# 🚀 BluePrint: Your AI Project Architect in Discord

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![discord.py](https://img.shields.io/badge/discord.py-2.3.2-7289DA?logo=discord&logoColor=white)](https://github.com/Rapptz/discord.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BluePrint is an AI-powered Discord bot designed to streamline your development workflow. It generates technical plans, learns from documentation, and acts as a centralized knowledge base for your team, all without leaving Discord.**

---

## ✨ Key Features

- **🤖 AI-Powered Planning**: Describe a feature, and BluePrint generates a detailed, step-by-step technical plan to guide your development.
- **📚 Documentation Genius**: Provide a documentation URL, and BluePrint will learn from documentation website, ready to answer questions.
- **🧠 Centralized Knowledge Base**: Save important messages, code snippets, and decisions directly to the bot's memory with a simple `@BluePrint`.
- **✅ Status Checks**: Quickly check what documentation and knowledge the bot has already learned.
- **slash Commands**: A clean, modern, and intuitive user experience using Discord's latest slash commands.

## 🎬 Demo

<iframe src="https://player.vimeo.com/video/1110819082" width="640" height="360" frameborder="0" allowfullscreen></iframe>

---

## 🛠️ Tech Stack

- **Language**: Python
- **Framework**: `discord.py` for Discord API interaction
- **AI**: Powered by a CloudRift's Llama-3.1-70B for planning and document analysis.
- **Hosting**: Railway

---

## ⚙️ Getting Started

Follow these instructions to get a local copy of BluePrint up and running for development and testing purposes.

### Prerequisites

- [Python 3.10 or higher](https://www.python.org/downloads/)
- A Discord Bot Token
    - Create a bot in the [Discord Developer Portal](https://discord.com/developers/applications).
    - Enable the **Message Content Intent** under the "Bot" tab.

### Installation & Configuration

1. **Clone the repository:**
    ```sh
    git clone [https://github.com/RitamPal26/BluePrint.git](https://github.com/RitamPal26/BluePrint.git)
    cd BluePrint
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up your environment variables:**
    -   Create a file named `.env` in the root directory of the project.
    -   Add your bot token and any other required API keys to this file.

    Your `.env` file should look like this:
    ```env
    # .env.example
    DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN_HERE
    RIFT_API_KEY=YOUR_RIFT_API_KEY_HERE 
    FIRECRAWL_API_KEY=YOUR_FIRECRAWL_API_KEY
    DECISION_LOG_CHANNEL=project-decisions
    ```

4. **Run the bot:**
    ```sh
    venv\Scripts\activate
    python bot.py
    ```

5. **Invite the bot to your server:**
    Use the following pre-configured link to add the bot to your server with the correct permissions.
    > [https://discord.com/oauth2/authorize?client_id=1405836550264062042&scope=bot%20applications.commands&permissions=397537941568](https://discord.com/oauth2/authorize?client_id=1405836550264062042&scope=bot%20applications.commands&permissions=397537941568)

---

## 🤖 Bot Commands

BluePrint uses slash commands for all its features.

| Command             | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `/plan [feature]`   | Creates a detailed technical plan for a new feature.         |
| `/learn-docs [url]` | Ingests and learns from a public documentation website.      |
| `/add-docs [url]`   | Adds a new documentation website to the existing knowledge.  |
| `/status`           | Checks the current status of the bot's knowledge base.       |
| `/help`             | Shows a complete guide on how to use the bot.                |

### Quick Save Feature

- `@BluePrint [your message]`
    > Mention the bot at the start of any message to save its content to the DECISION_LOG_CHANNEL channel or ask any questions.

---

## 🙏 Acknowledgements

A huge thanks to the following projects, tools, and communities that made BluePrint possible:

- **discord.py**: For powering seamless Discord API interactions.
- **CloudRift**: For providing access to the Llama-3.1-70B model and hosting their hackathon (August 15-17, 2025), which inspired this project.
- **Firecrawl API**: For enabling efficient documentation crawling and ingestion.
- **Railway**: For reliable hosting and deployment support.
- **The open-source community** on GitHub and Discord for invaluable resources, tutorials, and feedback.

Special shoutout to all contributors, testers, and hackathon participants—your input drives innovation! If you'd like to contribute, fork this repository or create a PR.

---

## 📄 License

This project is distributed under the MIT License. See `LICENSE` for more information.