# Claude MCP Test Repository

## Overview
This repository demonstrates the integration between Claude Desktop and GitHub using the Model Context Protocol (MCP). The following workflow was executed automatically through Claude:

## Workflow Steps
✅ Create a simple HTML page  
✅ Initialize repository "claude-mcp-test"  
✅ Push HTML page to repository  
✅ Add CSS styling and update repository  
✅ Create issue for content enhancement  
✅ Create feature branch and implement changes  
✅ Prepare pull request for main branch  
✅ Create README.md with instructions/documentation  

## Repository Structure
```
.
├── README.md
└── index.html       # Main webpage with CSS styling
```

## Features
- Clean, responsive HTML page
- Integrated CSS styling
- Navigation menu
- Multiple content sections
- Contact footer

## MCP Installation Guide for macOS

### Prerequisites
- macOS operating system
- Homebrew package manager
- Terminal access

### Step 1: Install Node Version Manager (nvm)
```bash
# Install nvm using Homebrew
brew install nvm

# Create nvm working directory
mkdir ~/.nvm
```

### Step 2: Configure Shell Profile
Add the following to your shell profile (e.g., `~/.profile`, `~/.zshrc`, or `~/.bash_profile`):
```bash
export NVM_DIR="$HOME/.nvm"
[ -s "/opt/homebrew/opt/nvm/nvm.sh" ] && \. "/opt/homebrew/opt/nvm/nvm.sh"  # This loads nvm
[ -s "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm" ] && \. "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm"  # This loads nvm bash_completion
```

### Step 3: Install Node.js
```bash
nvm install node

# Verify installation
node -v
npm -v
```

### Step 4: Configure Brave Search
1. Follow the [Brave Search setup guide](https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search)
2. Create/modify Claude Desktop configuration:
```json
{
    "mcpServers": {
      "brave-search": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-brave-search"
        ],
        "env": {
          "BRAVE_API_KEY": "your-api-key-here"
        }
      }
    }
}
```

### Step 5: Configure GitHub Integration
1. Follow the [GitHub setup guide](https://github.com/modelcontextprotocol/servers/tree/main/src/github)
2. Update Claude Desktop configuration:
```json
{
    "mcpServers": {
      "brave-search": {
        // Previous Brave search configuration
      },
      "github": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-github"
        ],
        "env": {
          "GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-token-here"
        }
      }
    }
}
```

### Brave Search and Git hub together:

```json
{
    "mcpServers": {
      "brave-search": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-brave-search"
        ],
        "env": {
          "BRAVE_API_KEY": ""
        }
      },
      "github": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-github"
        ],
        "env": {
          "GITHUB_PERSONAL_ACCESS_TOKEN": ""
        }
      }
    }
```

### Configuration File Location
The Claude Desktop configuration file is located at:
```bash
~/Library/Application\ Support/Claude/claude_desktop_config.json
```

### Troubleshooting
```bash
tail -n 20 -f ~/Library/Logs/Claude/mcp*.log
```

### Ensuring that MCP tools are used
- ❌  Get Tesla stock price today
- ✔️  Use Brave search to find Tesla stock price today

### Important Notes
- ⚠️ Keep your API keys and tokens secure
- 📄 Back up your configuration file
- ⚙️ Verify installations using provided command-line tools
- ❓ For troubleshooting, refer to the official documentation

---

### Resources
📚 [GitHub MCP Server Documentation](https://github.com/modelcontextprotocol/servers/tree/main/src/github)  
📚 [Brave Search MCP Documentation](https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search)

---

*Created and maintained through Claude Desktop using GitHub MCP*
