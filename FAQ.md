# ❓ Blackbox FAQ

## General Questions

### What is Blackbox?
Blackbox is a self-hosted workflow automation engine that uses declarative YAML files (`.bbx`) to automate complex tasks without writing code.

### Is Blackbox free?
Yes, for **non-commercial use**. Commercial use requires a paid license.

### What can I build with Blackbox?
API orchestration, chatbots, monitoring systems, data pipelines, CI/CD workflows, and scheduled tasks.

---

## Installation

### How do I install?
```bash
pip install -r requirements.txt
python cli.py run hello.bbx
```

### System requirements?
Python 3.8+, 256MB RAM minimum

---

## Workflows

### How do I create a workflow?
Create a `.bbx` file or use the visual editor.

### Can steps run in parallel?
Yes! Use `parallel: true` and `depends_on`

---

## Adapters

### What adapters are available?
- HTTP, Telegram, Logger (built-in)
- Firebase, GitHub, Stripe, MongoDB (via MCP Bridge)

### How do I use Telegram?
```yaml
steps:
  send:
    use: telegram.send_message
    args:
      bot_token: "YOUR_TOKEN"
      chat_id: "YOUR_CHAT_ID"
      text: "Hello!"
```

---

## Troubleshooting

### "Unknown MCP type: http"
Use supported adapter names: `http`, `logger`, `telegram`

### Telegram not working?
Check: bot token, sent `/start`, correct chat ID

### Variable substitution fails?
Use `save:` and reference as `${step.step_id.field}`

### Timeout errors?
Increase timeout: `timeout: 30s`

---

## Getting Help

- 📚 [Getting Started](GETTING_STARTED.md)
- 📖 [Tutorial](TUTORIAL.md)
- 💬 GitHub Issues
