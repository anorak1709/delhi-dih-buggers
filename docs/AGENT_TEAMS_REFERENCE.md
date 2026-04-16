# Agent Teams Master Reference Guide

> Comprehensive reference for building effective Claude Code agent teams and subagents.
> Source: https://code.claude.com/docs/en/agent-teams + https://code.claude.com/docs/en/sub-agents

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [When to Use Agent Teams vs Subagents](#when-to-use-agent-teams-vs-subagents)
3. [Enabling Agent Teams](#enabling-agent-teams)
4. [Starting a Team](#starting-a-team)
5. [Team Architecture](#team-architecture)
6. [Controlling Your Team](#controlling-your-team)
7. [Display Modes](#display-modes)
8. [Task Management](#task-management)
9. [Communication Patterns](#communication-patterns)
10. [Quality Gates with Hooks](#quality-gates-with-hooks)
11. [Subagent Configuration](#subagent-configuration)
12. [Subagent Frontmatter Reference](#subagent-frontmatter-reference)
13. [Tool Access Control](#tool-access-control)
14. [Permission Modes](#permission-modes)
15. [MCP Servers in Subagents](#mcp-servers-in-subagents)
16. [Persistent Memory](#persistent-memory)
17. [Hooks for Subagents](#hooks-for-subagents)
18. [Invocation Patterns](#invocation-patterns)
19. [Effective Prompt Patterns](#effective-prompt-patterns)
20. [Use Case Examples](#use-case-examples)
21. [Example Subagent Definitions](#example-subagent-definitions)
22. [Best Practices](#best-practices)
23. [Troubleshooting](#troubleshooting)
24. [Known Limitations](#known-limitations)

---

## Quick Start

```json
// .claude/settings.local.json — enable agent teams
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

Then tell Claude:

```
Create an agent team with 3 teammates to refactor the auth module.
One on backend logic, one on tests, one on API docs.
```

---

## When to Use Agent Teams vs Subagents

|                       | Subagents                                          | Agent Teams                                           |
|-----------------------|----------------------------------------------------|-------------------------------------------------------|
| **Context**           | Own context window; results return to caller        | Own context window; fully independent                 |
| **Communication**     | Report results back to main agent only              | Teammates message each other directly                 |
| **Coordination**      | Main agent manages all work                         | Shared task list with self-coordination               |
| **Best for**          | Focused tasks where only the result matters         | Complex work requiring discussion and collaboration   |
| **Token cost**        | Lower: results summarized back to main context      | Higher: each teammate is a separate Claude instance   |
| **Nested spawning**   | Cannot spawn other subagents                        | Cannot spawn nested teams                             |

### Use Agent Teams When:
- Multiple workers need to **communicate with each other**
- Research benefits from **competing hypotheses** and debate
- Work spans **multiple independent modules** (frontend, backend, tests)
- Tasks require **parallel exploration** that converges on findings
- You need workers that can **challenge each other's conclusions**

### Use Subagents When:
- Task is **self-contained** and only the result matters
- You want to **isolate verbose output** from your main context
- You need **quick, focused workers** that report back
- Work is **sequential** or has many dependencies
- You want to **enforce specific tool restrictions**

### Use Neither (Single Session) When:
- Task is **sequential** with many file-level dependencies
- Frequent **back-and-forth** or iterative refinement is needed
- Work involves **same-file edits** by multiple workers
- The task is simple enough that coordination overhead isn't worth it

---

## Enabling Agent Teams

Agent teams are **experimental and disabled by default**. Requires Claude Code v2.1.32+.

### Via settings.json (any scope):
```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

### Via shell environment:
```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

---

## Starting a Team

Tell Claude to create a team in natural language. Describe the task and team structure:

```
I'm designing a CLI tool that helps developers track TODO comments.
Create an agent team to explore this from different angles:
one teammate on UX, one on technical architecture, one playing devil's advocate.
```

### How Teams Get Started:
1. **You request a team** — explicitly ask for an agent team
2. **Claude proposes a team** — Claude suggests one if it determines the task benefits from parallel work; you confirm before it proceeds

Claude won't create a team without your approval.

---

## Team Architecture

An agent team consists of four components:

| Component      | Role                                                                         |
|----------------|------------------------------------------------------------------------------|
| **Team lead**  | The main Claude Code session that creates the team, spawns teammates, coordinates work |
| **Teammates**  | Separate Claude Code instances that each work on assigned tasks              |
| **Task list**  | Shared list of work items that teammates claim and complete                  |
| **Mailbox**    | Messaging system for communication between agents                           |

### Storage Locations:
- **Team config**: `~/.claude/teams/{team-name}/config.json`
- **Task list**: `~/.claude/tasks/{team-name}/`

The team config contains a `members` array with each teammate's name, agent ID, and agent type. Teammates can read this file to discover other team members.

---

## Controlling Your Team

### Specify Teammates and Models
```
Create a team with 4 teammates to refactor these modules in parallel.
Use Sonnet for each teammate.
```

### Require Plan Approval Before Implementation
```
Spawn an architect teammate to refactor the authentication module.
Require plan approval before they make any changes.
```

When a teammate finishes planning, it sends a plan approval request to the lead. The lead reviews and approves or rejects with feedback. Influence the lead's judgment with criteria:
- "only approve plans that include test coverage"
- "reject plans that modify the database schema"

### Talk to Teammates Directly
Each teammate is a full, independent Claude Code session.

- **In-process mode**: `Shift+Down` to cycle through teammates, type to message. `Enter` to view a session, `Escape` to interrupt. `Ctrl+T` for task list.
- **Split-pane mode**: Click into a teammate's pane.

### Shut Down a Teammate
```
Ask the researcher teammate to shut down
```
The teammate can approve (exits gracefully) or reject with an explanation.

### Clean Up the Team
```
Clean up the team
```
Always use the **lead** to clean up. Shut down all teammates first. Teammates should never run cleanup.

---

## Display Modes

| Mode           | Description                                    | Requirements        |
|----------------|------------------------------------------------|---------------------|
| `"auto"`       | Split panes if inside tmux; in-process otherwise | Default             |
| `"in-process"` | All teammates in main terminal                 | Any terminal        |
| `"tmux"`       | Each teammate gets own pane                    | tmux or iTerm2      |

### Set in settings.json:
```json
{
  "teammateMode": "in-process"
}
```

### Set per session:
```bash
claude --teammate-mode in-process
```

---

## Task Management

### Task States
Tasks have three states: **pending**, **in progress**, and **completed**.

Tasks can depend on other tasks — a pending task with unresolved dependencies cannot be claimed until those dependencies are completed.

### Assignment Models
- **Lead assigns**: Tell the lead which task to give to which teammate
- **Self-claim**: After finishing a task, a teammate picks up the next unassigned, unblocked task

Task claiming uses **file locking** to prevent race conditions.

### Sizing Tasks
- **Too small**: Coordination overhead exceeds the benefit
- **Too large**: Teammates work too long without check-ins
- **Just right**: Self-contained units that produce a clear deliverable (a function, a test file, a review)
- **Guideline**: 5-6 tasks per teammate keeps everyone productive

---

## Communication Patterns

Each teammate has its own context window. When spawned, a teammate loads the same project context as a regular session (CLAUDE.md, MCP servers, skills) plus the spawn prompt from the lead. The lead's conversation history does NOT carry over.

### Message Types:
- **message**: Send to one specific teammate
- **broadcast**: Send to all teammates simultaneously (use sparingly — costs scale with team size)

### Automatic Behaviors:
- **Automatic message delivery**: Messages delivered automatically to recipients
- **Idle notifications**: When a teammate finishes, it automatically notifies the lead
- **Shared task list**: All agents see task status and claim available work

---

## Quality Gates with Hooks

### TeammateIdle Hook
Runs when a teammate is about to go idle after finishing its turn.

**Input schema:**
```json
{
  "session_id": "abc123",
  "hook_event_name": "TeammateIdle",
  "teammate_name": "researcher",
  "team_name": "my-project",
  "cwd": "/path/to/project",
  "permission_mode": "default"
}
```

**Behavior:**
- Exit code 2 → teammate receives stderr message as feedback and **continues working**
- JSON `{"continue": false, "stopReason": "..."}` → **stops the teammate entirely**
- Does NOT support matchers (fires on every occurrence)

**Example — Build Artifact Validation:**
```bash
#!/bin/bash
if [ ! -f "./dist/output.js" ]; then
  echo "Build artifact missing. Run the build before stopping." >&2
  exit 2
fi
exit 0
```

### TaskCompleted Hook
Runs when a task is being marked as completed.

**Input schema:**
```json
{
  "session_id": "abc123",
  "hook_event_name": "TaskCompleted",
  "task_id": "task-001",
  "task_subject": "Implement user authentication",
  "task_description": "Add login and signup endpoints",
  "teammate_name": "implementer",
  "team_name": "my-project"
}
```

**Behavior:**
- Exit code 2 → task is NOT marked as completed, stderr fed back as feedback
- JSON `{"continue": false, "stopReason": "..."}` → stops the teammate entirely
- Does NOT support matchers

**Example — Test Suite Validation:**
```bash
#!/bin/bash
INPUT=$(cat)
TASK_SUBJECT=$(echo "$INPUT" | jq -r '.task_subject')

if ! npm test 2>&1; then
  echo "Tests not passing. Fix failing tests before completing: $TASK_SUBJECT" >&2
  exit 2
fi
exit 0
```

### Hook Configuration in settings.json:
```json
{
  "hooks": {
    "TeammateIdle": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/check-build-artifacts.sh"
          }
        ]
      }
    ],
    "TaskCompleted": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/run-tests-on-complete.sh"
          }
        ]
      }
    ]
  }
}
```

---

## Subagent Configuration

### File Format
Subagent files are Markdown with YAML frontmatter:

```markdown
---
name: code-reviewer
description: Reviews code for quality and best practices
tools: Read, Glob, Grep
model: sonnet
---

You are a code reviewer. When invoked, analyze the code and provide
specific, actionable feedback on quality, security, and best practices.
```

### Scope and Priority

| Location                     | Scope                    | Priority    |
|------------------------------|--------------------------|-------------|
| `--agents` CLI flag          | Current session          | 1 (highest) |
| `.claude/agents/`            | Current project          | 2           |
| `~/.claude/agents/`          | All your projects        | 3           |
| Plugin's `agents/` directory | Where plugin is enabled  | 4 (lowest)  |

### CLI-Defined Subagents (Session Only)
```bash
claude --agents '{
  "code-reviewer": {
    "description": "Expert code reviewer. Use proactively after code changes.",
    "prompt": "You are a senior code reviewer. Focus on code quality, security, and best practices.",
    "tools": ["Read", "Grep", "Glob", "Bash"],
    "model": "sonnet"
  },
  "debugger": {
    "description": "Debugging specialist for errors and test failures.",
    "prompt": "You are an expert debugger. Analyze errors, identify root causes, and provide fixes."
  }
}'
```

### Managing Subagents
- `/agents` — Interactive interface to view, create, edit, delete subagents
- `claude agents` — List all configured subagents from CLI

---

## Subagent Frontmatter Reference

| Field             | Required | Description                                                                                          |
|-------------------|----------|------------------------------------------------------------------------------------------------------|
| `name`            | Yes      | Unique identifier using lowercase letters and hyphens                                                |
| `description`     | Yes      | When Claude should delegate to this subagent                                                         |
| `tools`           | No       | Allowlist of tools. Inherits all if omitted                                                          |
| `disallowedTools` | No       | Denylist of tools removed from inherited/specified list                                              |
| `model`           | No       | `sonnet`, `opus`, `haiku`, full model ID, or `inherit` (default)                                     |
| `permissionMode`  | No       | `default`, `acceptEdits`, `dontAsk`, `bypassPermissions`, or `plan`                                  |
| `maxTurns`        | No       | Maximum agentic turns before the subagent stops                                                      |
| `skills`          | No       | Skills to load into context at startup (full content injected)                                       |
| `mcpServers`      | No       | MCP servers available to this subagent (inline definitions or references)                            |
| `hooks`           | No       | Lifecycle hooks scoped to this subagent                                                              |
| `memory`          | No       | Persistent memory scope: `user`, `project`, or `local`                                               |
| `background`      | No       | `true` to always run as background task (default: `false`)                                           |
| `effort`          | No       | Effort level: `low`, `medium`, `high`, `max` (Opus 4.6 only)                                        |
| `isolation`       | No       | `worktree` for isolated git worktree copy                                                            |

---

## Tool Access Control

### Allowlist (tools field)
Only these tools are available:
```yaml
tools: Read, Grep, Glob, Bash
```

### Denylist (disallowedTools field)
Inherit everything except these:
```yaml
disallowedTools: Write, Edit
```

If both set: `disallowedTools` applied first, then `tools` resolved against remaining pool.

### Restrict Subagent Spawning
```yaml
# Only allow spawning worker and researcher subagents
tools: Agent(worker, researcher), Read, Bash

# Allow spawning any subagent
tools: Agent, Read, Bash

# Omit Agent entirely to prevent spawning any subagents
tools: Read, Bash
```

### Disable Specific Subagents
```json
{
  "permissions": {
    "deny": ["Agent(Explore)", "Agent(my-custom-agent)"]
  }
}
```

### Built-in Subagents

| Agent             | Model    | Tools          | Purpose                               |
|-------------------|----------|----------------|---------------------------------------|
| **Explore**       | Haiku    | Read-only      | File discovery, codebase exploration  |
| **Plan**          | Inherit  | Read-only      | Codebase research for planning        |
| **General-purpose** | Inherit | All            | Complex multi-step tasks              |
| **Bash**          | Inherit  | Terminal        | Running commands in separate context  |
| **statusline-setup** | Sonnet | -           | Configure status line                 |
| **Claude Code Guide** | Haiku | -           | Answer Claude Code feature questions  |

---

## Permission Modes

| Mode                  | Behavior                                                 |
|-----------------------|----------------------------------------------------------|
| `default`             | Standard permission checking with prompts                |
| `acceptEdits`         | Auto-accept file edits                                   |
| `dontAsk`             | Auto-deny permission prompts (explicitly allowed tools still work) |
| `bypassPermissions`   | Skip permission prompts (use with caution)               |
| `plan`                | Plan mode (read-only exploration)                        |

**Important**: If the parent uses `bypassPermissions`, this takes precedence and cannot be overridden. Teammates start with the lead's permission settings.

---

## MCP Servers in Subagents

```yaml
---
name: browser-tester
description: Tests features in a real browser using Playwright
mcpServers:
  # Inline definition: scoped to this subagent only
  - playwright:
      type: stdio
      command: npx
      args: ["-y", "@playwright/mcp@latest"]
  # Reference by name: reuses an already-configured server
  - github
---

Use the Playwright tools to navigate, screenshot, and interact with pages.
```

Inline servers connect when the subagent starts, disconnect when it finishes. To keep an MCP server out of the main conversation entirely, define it inline in the subagent.

**Security note**: Plugin subagents do NOT support `hooks`, `mcpServers`, or `permissionMode` fields.

---

## Persistent Memory

### Scopes

| Scope     | Location                                       | Use when                                       |
|-----------|-------------------------------------------------|------------------------------------------------|
| `user`    | `~/.claude/agent-memory/<name>/`                | Learnings apply across all projects            |
| `project` | `.claude/agent-memory/<name>/`                  | Project-specific, shareable via version control |
| `local`   | `.claude/agent-memory-local/<name>/`            | Project-specific, NOT in version control       |

### Example:
```yaml
---
name: code-reviewer
description: Reviews code for quality and best practices
memory: user
---

You are a code reviewer. As you review code, update your agent memory with
patterns, conventions, and recurring issues you discover.
```

### How It Works:
- System prompt includes instructions for reading/writing to the memory directory
- First 200 lines of `MEMORY.md` are included in context
- Read, Write, and Edit tools are automatically enabled for memory management

### Tips:
- Ask the subagent to consult memory before starting: "Check your memory for patterns you've seen before"
- Ask it to update memory after completing work: "Save what you learned to your memory"
- Include memory instructions in the markdown body for proactive maintenance

---

## Hooks for Subagents

### In Subagent Frontmatter (scoped to that subagent):
```yaml
---
name: code-reviewer
description: Review code changes with automatic linting
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-command.sh"
  PostToolUse:
    - matcher: "Edit|Write"
      hooks:
        - type: command
          command: "./scripts/run-linter.sh"
---
```

### In settings.json (project-level lifecycle hooks):
```json
{
  "hooks": {
    "SubagentStart": [
      {
        "matcher": "db-agent",
        "hooks": [
          { "type": "command", "command": "./scripts/setup-db-connection.sh" }
        ]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [
          { "type": "command", "command": "./scripts/cleanup-db-connection.sh" }
        ]
      }
    ]
  }
}
```

### Hook Events for Subagents

| Event             | Matcher input     | When it fires                         |
|-------------------|-------------------|---------------------------------------|
| `PreToolUse`      | Tool name         | Before the subagent uses a tool       |
| `PostToolUse`     | Tool name         | After the subagent uses a tool        |
| `Stop`            | (none)            | When subagent finishes (→ SubagentStop) |
| `SubagentStart`   | Agent type name   | When a subagent begins execution      |
| `SubagentStop`    | Agent type name   | When a subagent completes             |
| `TeammateIdle`    | (no matcher)      | When teammate about to go idle        |
| `TaskCompleted`   | (no matcher)      | When task being marked complete       |

### Hook Types
1. **command** — Runs a shell command
2. **prompt** — Evaluates a condition with LLM (tool events only)
3. **agent** — Runs an agent with tools (tool events only)

### Exit Code Behavior
- **Exit 0**: Hook passes, continue normally
- **Exit 2**: Block the action, feed stderr back as feedback

---

## Invocation Patterns

### Natural Language
```
Use the test-runner subagent to fix failing tests
```

### @-Mention (Guaranteed Invocation)
```
@"code-reviewer (agent)" look at the auth changes
```

### Session-Wide Agent
```bash
claude --agent code-reviewer
```

Or in settings:
```json
{ "agent": "code-reviewer" }
```

### Foreground vs Background
- **Foreground**: Blocks main conversation; permission prompts passed through
- **Background**: Runs concurrently; permissions pre-approved at launch
- Ask Claude to "run this in the background" or press `Ctrl+B`
- Disable with `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`

### Resuming Subagents
```
Use the code-reviewer subagent to review the authentication module
[Agent completes]

Continue that code review and now analyze the authorization logic
[Claude resumes with full context from previous conversation]
```

---

## Effective Prompt Patterns

### Give Teammates Enough Context
Bad:
```
Spawn a security reviewer.
```
Good:
```
Spawn a security reviewer teammate with the prompt: "Review the authentication module
at src/auth/ for security vulnerabilities. Focus on token handling, session
management, and input validation. The app uses JWT tokens stored in
httpOnly cookies. Report any issues with severity ratings."
```

### Adversarial/Debate Pattern
```
Spawn 5 agent teammates to investigate different hypotheses. Have them talk to
each other to try to disprove each other's theories, like a scientific debate.
Update the findings doc with whatever consensus emerges.
```

### Prevent Lead from Doing Work
```
Wait for your teammates to complete their tasks before proceeding.
```

### Influence Plan Approval
```
Only approve plans that include test coverage.
Reject plans that modify the database schema.
```

---

## Use Case Examples

### Parallel Code Review
```
Create an agent team to review PR #142. Spawn three reviewers:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review and report findings.
```

### Competing Hypotheses Debugging
```
Users report the app exits after one message instead of staying connected.
Spawn 5 agent teammates to investigate different hypotheses. Have them talk to
each other to try to disprove each other's theories, like a scientific debate.
Update the findings doc with whatever consensus emerges.
```

### New Feature with Independent Modules
```
Create an agent team to implement the new notification system:
- One teammate for the backend API endpoints
- One for the React notification components
- One for the WebSocket real-time delivery
- One for the test suite
```

### Cross-Layer Coordination
```
Create a team to add user preferences:
- Backend teammate: database schema + API endpoints
- Frontend teammate: settings UI components
- Test teammate: integration and unit tests
```

---

## Example Subagent Definitions

### Code Reviewer (Read-Only)
```markdown
---
name: code-reviewer
description: Expert code review specialist. Proactively reviews code for quality, security, and maintainability. Use immediately after writing or modifying code.
tools: Read, Grep, Glob, Bash
model: inherit
---

You are a senior code reviewer ensuring high standards of code quality and security.

When invoked:
1. Run git diff to see recent changes
2. Focus on modified files
3. Begin review immediately

Review checklist:
- Code is clear and readable
- Functions and variables are well-named
- No duplicated code
- Proper error handling
- No exposed secrets or API keys
- Input validation implemented
- Good test coverage
- Performance considerations addressed

Provide feedback organized by priority:
- Critical issues (must fix)
- Warnings (should fix)
- Suggestions (consider improving)

Include specific examples of how to fix issues.
```

### Debugger (Read + Write)
```markdown
---
name: debugger
description: Debugging specialist for errors, test failures, and unexpected behavior. Use proactively when encountering any issues.
tools: Read, Edit, Bash, Grep, Glob
---

You are an expert debugger specializing in root cause analysis.

When invoked:
1. Capture error message and stack trace
2. Identify reproduction steps
3. Isolate the failure location
4. Implement minimal fix
5. Verify solution works

Debugging process:
- Analyze error messages and logs
- Check recent code changes
- Form and test hypotheses
- Add strategic debug logging
- Inspect variable states

For each issue, provide:
- Root cause explanation
- Evidence supporting the diagnosis
- Specific code fix
- Testing approach
- Prevention recommendations

Focus on fixing the underlying issue, not the symptoms.
```

### Data Scientist
```markdown
---
name: data-scientist
description: Data analysis expert for SQL queries, BigQuery operations, and data insights. Use proactively for data analysis tasks and queries.
tools: Bash, Read, Write
model: sonnet
---

You are a data scientist specializing in SQL and BigQuery analysis.

When invoked:
1. Understand the data analysis requirement
2. Write efficient SQL queries
3. Use BigQuery command line tools (bq) when appropriate
4. Analyze and summarize results
5. Present findings clearly

Key practices:
- Write optimized SQL queries with proper filters
- Use appropriate aggregations and joins
- Include comments explaining complex logic
- Format results for readability
- Provide data-driven recommendations
```

### Database Query Validator (Hook-Gated)
```markdown
---
name: db-reader
description: Execute read-only database queries. Use when analyzing data or generating reports.
tools: Bash
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-readonly-query.sh"
---

You are a database analyst with read-only access. Execute SELECT queries to answer questions about the data.

You cannot modify data. If asked to INSERT, UPDATE, DELETE, or modify schema, explain that you only have read access.
```

**Validation script** (`./scripts/validate-readonly-query.sh`):
```bash
#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
  exit 0
fi

if echo "$COMMAND" | grep -iE '\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE)\b' > /dev/null; then
  echo "Blocked: Write operations not allowed. Use SELECT queries only." >&2
  exit 2
fi

exit 0
```

---

## Best Practices

### Team Size
- Start with **3-5 teammates** for most workflows
- **5-6 tasks per teammate** keeps everyone productive
- Three focused teammates often outperform five scattered ones
- Scale up only when work genuinely benefits from parallelism

### Task Design
- Each task should produce a **clear deliverable** (function, test file, review)
- Avoid tasks that require **same-file edits** by multiple teammates
- Break the work so each teammate **owns a different set of files**

### Context Management
- Teammates **don't inherit** the lead's conversation history
- Include **task-specific details** in spawn prompts
- Teammates load CLAUDE.md, MCP servers, and skills automatically

### Subagent Design
- Each subagent should **excel at one specific task**
- Write **detailed descriptions** — Claude uses them to decide when to delegate
- Limit tool access to **only necessary permissions**
- Check project subagents into **version control**

### Monitoring
- Check in on teammates' progress regularly
- Redirect approaches that aren't working
- If the lead starts implementing itself: "Wait for your teammates to complete their tasks"

### Token Efficiency
- Token usage **scales linearly** with number of active teammates
- For research/review/new features, extra tokens are usually worthwhile
- For routine tasks, a single session is more cost-effective

### Start Simple
- Start with **research and review** tasks (clear boundaries, no code writing)
- Graduate to **parallel implementation** once comfortable with coordination

---

## Troubleshooting

### Teammates Not Appearing
- In-process mode: Press `Shift+Down` to cycle — they may be running but not visible
- Check that the task was complex enough to warrant a team
- For split panes, verify tmux is installed: `which tmux`
- For iTerm2, verify `it2` CLI is installed and Python API is enabled

### Too Many Permission Prompts
Pre-approve common operations in permission settings before spawning teammates:
```json
{
  "permissions": {
    "allow": ["Bash(npm:*)", "Bash(git:*)", "Read", "Edit"]
  }
}
```

### Teammates Stopping on Errors
- Check output using `Shift+Down` or click the pane
- Give additional instructions directly, or spawn a replacement

### Lead Shuts Down Prematurely
Tell the lead: "Keep going" or "Wait for teammates to finish before proceeding"

### Orphaned tmux Sessions
```bash
tmux ls
tmux kill-session -t <session-name>
```

### Task Status Lagging
Teammates sometimes fail to mark tasks completed, blocking dependent tasks. Check if work is actually done and update manually or tell the lead to nudge the teammate.

---

## Known Limitations

| Limitation | Details |
|---|---|
| **No session resumption** | `/resume` and `/rewind` do not restore in-process teammates |
| **Task status lag** | Teammates sometimes fail to mark tasks completed |
| **Slow shutdown** | Teammates finish current request before shutting down |
| **One team per session** | Clean up before starting a new team |
| **No nested teams** | Teammates cannot spawn their own teams |
| **Fixed lead** | Cannot promote teammate to lead or transfer leadership |
| **Permissions at spawn** | All teammates start with lead's mode; change individually after |
| **Split panes** | Not supported in VS Code terminal, Windows Terminal, or Ghostty |
| **Subagents can't spawn subagents** | Nesting is not supported |

---

## Quick Decision Matrix

| Scenario | Approach |
|---|---|
| Quick focused lookup | Single Explore subagent |
| Verbose test output isolation | Single subagent |
| Independent module implementation | Agent team (3-5 teammates) |
| Competing hypothesis debugging | Agent team with debate prompt |
| Code review from multiple angles | Agent team (3 reviewers) |
| Sequential multi-step workflow | Chain subagents from main conversation |
| Same-file edits needed | Single session |
| Simple targeted change | Main conversation directly |
