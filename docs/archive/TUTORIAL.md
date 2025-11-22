# Blackbox Tutorial

This tutorial will guide you through creating a real-world automation workflow using Blackbox.

## Goal
We will create a workflow that:
1. Checks if a website is up.
2. If it's up, logs a success message.
3. If it's down (or error), logs a failure.

## Step 1: Use the Wizard
Run the wizard to generate a template:
```bash
python cli.py wizard
```
1. **Name:** Website Monitor
2. **Template:** API Monitor (Option 3)
3. **URL:** https://google.com

This will generate `website_monitor.bbx`.

## Step 2: Examine the Code
Open `website_monitor.bbx` in VS Code. You'll see:

```yaml
steps:
  check_api:
    use: http.get
    args:
      url: "https://google.com"
      
  log_success:
    use: logger.info
    args:
      message: "API is UP: ${check_api.output}"
    when: "${check_api.status} == 'success'"
```

## Step 3: Run via CLI
```bash
python cli.py run website_monitor.bbx
```

## Step 4: Run via Dashboard
1. Start the server: `python api_server.py`
2. Open `http://localhost:8000`
3. Click "Run" on `website_monitor.bbx`.
4. Watch the logs appear in real-time!

## Advanced: Subflows
Try creating a `master.bbx` that calls this monitor:

```yaml
steps:
  run_monitor:
    use: flow.run
    args:
      path: website_monitor.bbx
```

Run `master.bbx` and watch it execute the monitor as a sub-step.
