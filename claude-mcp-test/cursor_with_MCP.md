# TypeScript Project Setup Guide

## What is This Project?
This is a Model Context Protocol (MCP) server implementation that extends Claude's capabilities in the Cursor IDE. MCP is Anthropic's protocol that allows Claude to interact with external tools and services. This server provides three tools:

1. Screenshot Tool - Takes screenshots of web pages with customizable viewport sizes (uses Puppeteer)
2. Architect Tool - Uses OpenAI's GPT-4 model to help plan coding tasks
3. Code Review Tool - Shows git diff output for code changes (currently uses git diff only, no LLM)

## How It Works
- **Claude Integration**: 
  - When you chat with Claude in Cursor, it can access these tools through the MCP protocol
  - Claude knows how to use these tools and can call them based on your requests
  - The communication between Claude and the tools is handled by the MCP server

- **MCP Protocol**:
  - MCP (Model Context Protocol) is Anthropic's protocol for tool integration
  - It allows Claude to:
    - Discover available tools and their capabilities
    - Call tools with appropriate parameters
    - Receive and interpret tool results
    - Handle errors and provide feedback
  - The protocol uses stdio for communication between Claude and the tools

- **Tool Architecture**:
  - Each tool is registered with the MCP server
  - Tools can use any technology or API under the hood:
    - Direct system calls (Screenshot tool using Puppeteer)
    - External APIs (Architect tool using OpenAI)
    - Local commands (Code Review tool using git)
  - Claude handles the natural language processing and decides when and how to use each tool

- **Server Lifecycle**:
  1. Server starts and registers available tools
  2. Claude connects to the server through Cursor
  3. Claude discovers available tools
  4. User interacts with Claude in natural language
  5. Claude calls appropriate tools as needed
  6. Tools execute and return results to Claude
  7. Claude interprets results and responds to user

## Claude and MCP: Detailed Interaction

### Communication Flow
1. **User to Claude**:
   - You chat with Claude in the Cursor IDE
   - Claude processes your natural language request
   - Claude determines if it needs to use any tools

2. **Claude to MCP Server**:
   - Claude sends a tool request through the MCP protocol
   - The request includes tool name and parameters
   - Communication happens via stdio (standard input/output)

3. **MCP Server to Tools**:
   - Server validates the request
   - Calls the appropriate tool with parameters
   - Tool executes and returns results

4. **Results Back to User**:
   - Tool results flow back through MCP to Claude
   - Claude interprets the results
   - Claude provides human-friendly response

### Example Interaction Flow
When you ask: "Take a screenshot of Google with tablet size viewport"

1. **Claude's Processing**:
   - Recognizes this as a screenshot request
   - Determines required parameters:
     - URL: "https://www.google.com"
     - Viewport: tablet size (768x1024)
     - Output filename

2. **MCP Protocol Steps**:
   ```
   Claude → MCP: Tool discovery request
   MCP → Claude: List of available tools and their schemas
   Claude → MCP: Screenshot tool request with parameters
   MCP → Tool: Execute screenshot command
   Tool → MCP: Screenshot result/confirmation
   MCP → Claude: Tool execution result
   Claude → User: Confirmation and next steps
   ```

3. **Behind the Scenes**:
   - MCP server maintains tool registry
   - Handles parameter validation
   - Manages tool lifecycle
   - Provides error handling
   - Logs operations (in ~/mcp-server.log)

### MCP Protocol Features
- **Tool Discovery**: Claude can query available tools
- **Schema Validation**: Each tool defines its required parameters
- **Error Handling**: Structured error reporting back to Claude
- **Asynchronous Operation**: Tools can run long operations
- **State Management**: Server maintains tool state
- **Logging**: Detailed logs for debugging

### Security and Permissions
- MCP runs locally on your machine
- Tools have access to your local environment
- Claude can only access what the tools expose
- API keys (like OpenAI) are managed locally

### Debugging MCP Interactions
- Watch server logs: `tail -f ~/mcp-server.log`
- Check tool-specific outputs
- Monitor process status: `ps -aef | grep mcp`
- Debug mode: `DEBUG=* node dist/tools/index.js`

## Integration Details
- **MCP Server**: Implements the Model Context Protocol to communicate with Claude in Cursor
- **Claude Integration**: The tools are exposed to Claude through the MCP server, allowing Claude to use them during conversations
- **Mixed Model Usage**: While the MCP server talks to Claude, individual tools can use different technologies:
  - The Architect tool uses OpenAI's API for code analysis
  - The Screenshot tool uses Puppeteer (no LLM)
  - The Code Review tool uses git commands (no LLM)

## Tools and Technologies

### Screenshot Tool
- **Technology**: Uses Puppeteer (headless Chrome browser)
- **No LLM/AI**: Pure browser automation tool
- **Features**:
  - Customizable viewport sizes for different devices
  - Full-page screenshots
  - Support for both URLs and local paths
  - Configurable timeouts and browser settings

### Architect Tool
- **Technology**: OpenAI API
- **Model**: GPT-4 (configurable)
- **Features**:
  - Code analysis and planning
  - Implementation suggestions
  - Can be configured to use different OpenAI models:
    - GPT-4 (current default, best quality)
    - GPT-3.5-turbo (faster, more cost-effective)
    - Other OpenAI models as they become available

### Code Review Tool
- **Technology**: Git command-line interface
- **Current Implementation**: Uses `git diff` command
- **No LLM Currently**: Just shows raw diff output
- **Potential Future Enhancements**:
  - Add LLM analysis of diff output
  - Integrate with OpenAI for intelligent code review
  - Add automated fix suggestions

## Initial Setup

### 1. Environment Setup
1. Make sure Node.js and npm are installed
2. Add this alias to your `.bash_profile` or `.zshrc`:
```bash
alias start-mcp-server='cd /Users/soumukhe/pythonsScripts/cursor/mcp && source ~/.nvm/nvm.sh && nvm use v18.17.0 && (tsc --project tsconfig.json && node dist/tools/index.js > ~/mcp-server.log 2>&1 &)'
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Compile TypeScript
```bash
tsc --project tsconfig.json
```

## Tool Usage Examples

### 1. Screenshot Tool
Takes screenshots of web pages with customizable viewport sizes.

Examples:
```
// Desktop viewport (default: 1280x800)
Take a screenshot of https://www.google.com and save it as google-desktop.png

// Tablet viewport (768x1024)
Take a screenshot of Google's homepage with a tablet viewport size of 768x1024 and save it as google-tablet.png

// Mobile viewport (375x667)
Take a screenshot of Google's homepage with an iPhone viewport size of 375x667 and save it as google-mobile.png
```

### 2. Architect Tool
Uses OpenAI's GPT models to analyze code and provide implementation steps.

Example:
```
Use the architect tool to plan how to add error handling to the screenshot function
```

### 3. Code Review Tool
Reviews code changes against the main branch.

Example:
```
Review the changes in the current directory
```

## Configuration

### OpenAI API Key
1. The OpenAI API key is stored in `src/env/keys.js`
2. To update the API key:
   ```javascript
   // src/env/keys.js
   export default {
     OPENAI_API_KEY: "your-api-key-here"
   };
   ```

### Changing OpenAI Models
1. The model is configured in `src/tools/architect.ts`
2. To change the model:
   ```typescript
   // src/tools/architect.ts
   const response = await openai.chat.completions.create({
     model: "gpt-4", // Change this to your desired model
     messages: [
       { role: "system", content: systemPrompt },
       { role: "user", content: userPrompt },
     ],
   });
   ```
   Available models include: "gpt-4", "gpt-3.5-turbo", etc.

## Starting and Stopping the Server

### Start the Server
```bash
start-mcp-server
```

### Check Server Status
```bash
ps -aef | grep mcp
```

### Stop the Server
```bash
pkill -f "pythonsScripts/cursor/mcp/dist/tools/index.js"
```

## Troubleshooting

### If the server isn't responding:
1. Check if it's running:
   ```bash
   ps -aef | grep mcp
   ```
2. Kill existing instances:
   ```bash
   pkill -f "pythonsScripts/cursor/mcp/dist/tools/index.js"
   ```
3. Restart the server:
   ```bash
   start-mcp-server
   ```

### If you get TypeScript errors:
1. Make sure all dependencies are installed:
   ```bash
   npm install
   ```
2. Recompile the TypeScript code:
   ```bash
   tsc --project tsconfig.json
   ```

### Check Server Logs
```bash
tail -f ~/mcp-server.log
```

## Advanced Usage and Examples

### How Claude Decides Which Tool to Use
Claude analyzes your request and matches it against the tools' descriptions and capabilities:

1. **Screenshot Tool Selection**:
   - Triggered by keywords like "screenshot", "capture", "webpage"
   - When viewport sizes are mentioned (desktop, mobile, tablet)
   - For visual documentation requests

2. **Architect Tool Selection**:
   - When planning or architecture questions are asked
   - For code analysis and implementation strategy
   - When breaking down complex tasks

3. **Code Review Tool Selection**:
   - When reviewing code changes is mentioned
   - For git diff related requests
   - When comparing versions

### Error Handling in MCP
The MCP protocol handles errors at multiple levels:

1. **Protocol Level**:
   ```
   Claude → MCP: Invalid tool request
   MCP → Claude: Error with schema validation details
   Claude → User: Human-friendly error explanation
   ```

2. **Tool Level**:
   - Each tool has its own error handling
   - Errors are logged to `~/mcp-server.log`
   - Claude receives structured error responses

3. **Recovery Strategies**:
   - Claude can retry with different parameters
   - Fallback to alternative approaches
   - Clear error messages to user

### Real-World Examples

#### 1. Screenshot Tool - Complex Scenarios

**Example 1: Multi-device Testing**
```
User: "Take screenshots of our login page for desktop, tablet, and mobile"
Claude: I'll help you capture screenshots in different viewport sizes.

1. Desktop (1280x800):
   [Takes screenshot with default size]

2. Tablet (768x1024):
   [Takes screenshot with tablet dimensions]

3. Mobile (375x667):
   [Takes screenshot with mobile dimensions]
```

**Example 2: Error Recovery**
```
User: "Screenshot https://internal-site.company.com"
Claude: The screenshot failed because the site isn't accessible. Let me try:
1. Checking if it's a local URL
2. Verifying server status
3. Suggesting alternatives
```

#### 2. Architect Tool - Advanced Planning

**Example 1: Complex Feature Planning**
```
User: "Plan how to add user authentication to our Express app"
Claude: Let me analyze the requirements and existing code.

[Architect tool response includes]:
1. Required dependencies
2. Database schema changes
3. New API endpoints
4. Security considerations
5. Testing strategy
```

**Example 2: Code Refactoring**
```
User: "How should we refactor this monolithic function?"
Claude: I'll analyze the code and suggest a modular approach.

[Architect tool provides]:
1. Code structure analysis
2. Separation of concerns
3. New function boundaries
4. Data flow diagram
5. Implementation steps
```

#### 3. Code Review Tool - Detailed Analysis

**Example 1: Feature Branch Review**
```
User: "Review the changes in the auth-feature branch"
Claude: I'll analyze the git diff and provide feedback.

[Code Review tool shows]:
1. Files changed
2. Lines added/removed
3. Potential issues
4. Suggested improvements
```

**Example 2: Security Review**
```
User: "Check if the recent changes exposed any API keys"
Claude: I'll review the diff for sensitive information.

[Code Review tool helps identify]:
1. Exposed credentials
2. Security-sensitive changes
3. Best practice violations
```

### Tool Interaction Patterns

1. **Sequential Tool Usage**:
   ```
   User: "Update the login page and show me how it looks"
   Claude: 
   1. Uses Architect tool to plan changes
   2. Waits for user to implement
   3. Uses Screenshot tool to verify
   ```

2. **Complementary Analysis**:
   ```
   User: "Review the new feature implementation"
   Claude:
   1. Uses Code Review tool for diff
   2. Uses Architect tool for improvement suggestions
   ```

3. **Iterative Refinement**:
   ```
   User: "Optimize the mobile layout"
   Claude:
   1. Takes initial screenshot
   2. Uses Architect tool for suggestions
   3. Takes new screenshot to verify changes
   ```

## Project Structure
```
src/
  ├── tools/
  │   ├── index.ts        (main file that brings everything together)
  │   ├── screenshot.ts   (screenshot tool implementation)
  │   ├── architect.ts    (AI architect tool implementation)
  │   └── codeReview.ts   (code review tool implementation)
  └── env/
      └── keys.js         (contains API keys)
```

## Configuration Files
1. `tsconfig.json`: TypeScript configuration
2. `package.json`: Project dependencies

## Important Notes
- The screenshot tool supports custom viewport sizes for different device types
- Always keep your OpenAI API key secure and never commit it to version control
- The server logs are written to `~/mcp-server.log`
- When modifying TypeScript files, remember to recompile using `tsc --project tsconfig.json` 
