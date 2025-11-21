# Getting Started with Blackbox

## Prerequisites
- Python 3.8+
- VS Code (recommended for IntelliSense)

## 1. Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/kurokie1337/bbx.git
cd bbx
pip install -r requirements.txt
```

## 2. Configure VS Code
For the best experience, enable autocomplete:
1. Run `python cli.py schema` to generate `bbx.schema.json`.
2. Create `.vscode/settings.json`:
   ```json
   {
     "json.schemas": [
       {
         "fileMatch": ["*.bbx"],
         "url": "./bbx.schema.json"
       }
     ]
   }
   ```

## 3. Your First Workflow
The easiest way to start is using the Wizard:
```bash
python cli.py wizard
```
Select "Empty Workflow" or a template like "Web Scraper".

## 4. Running Workflows
### via CLI
```bash
python cli.py run my_workflow.bbx
```

### via Dashboard
1. Run `python api_server.py`
2. Go to `http://localhost:8000`
3. Click "Run" on your workflow.

## 5. Next Steps
- Learn about [Composability](README.md#composability-train-cars)
- Explore [Adapters](docs/RUNTIME_INTERNALS.md)
