# User Workflows

## 🎨 Purpose

**Dynamic user-generated automations and AI-generated workflows**

These workflows are:
- ⚡ **Dynamic** - Can be modified/regenerated frequently
- 🤖 **AI-Generated** - Created by BBX AI from natural language
- 🔄 **Experimental** - Test ideas and automation concepts
- 👤 **Personal** - User-specific automation tasks

---

## 🎯 What Goes Here

### AI-Generated Workflows
```
- generated.bbx          (AI-generated from "Deploy to AWS")
- deploy_nextjs.bbx      (AI-generated from "Deploy Next.js app")
- test_pipeline.bbx      (AI-generated from "Run pytest tests")
```

### Custom Automations
```
- personal_backup.bbx
- data_processing.bbx
- notification_script.bbx
```

### Experimental Workflows
```
- test_new_feature.bbx
- prototype_deployment.bbx
```

---

## 🤖 Generating Workflows with AI

### Using CLI:
```bash
bbx generate "Deploy React app to Vercel"
# Generates: generated.bbx

# Rename to something meaningful:
mv generated.bbx deploy_react_vercel.bbx
```

### Using MCP (Claude Code):
```
Ask AI: "Generate a BBX workflow for deploying to AWS S3"
AI will use bbx_generate tool automatically
```

---

## ✨ Workflow Lifecycle

1. **Generate** - Use AI or create manually
2. **Test** - Validate and dry-run
3. **Refine** - Modify and improve
4. **Promote** - Move stable workflows to `core/` if needed
5. **Delete** - Remove obsolete workflows

---

## 📝 Best Practices

- **Name clearly** - Use descriptive names: `deploy_nextjs_prod.bbx`
- **Add comments** - Document what the workflow does
- **Version control** - Commit useful workflows to git
- **Clean up** - Delete obsolete/failed experiments

---

## 🔗 See Also

- [Core Workflows](../core/README.md) - Stable system workflows
- [Meta Workflows](../meta/README.md) - BBX system workflows
- [MCP Server](../../MCP_SERVER.md) - AI agent integration
