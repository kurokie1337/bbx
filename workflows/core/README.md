# Core Workflows

## 📦 Purpose

**Stable workflows for BBX system development and infrastructure**

These workflows are:
- ✅ **Stable** - Don't modify frequently
- ✅ **Production-ready** - Tested and reliable
- ✅ **System-critical** - Core infrastructure operations

---

## 🎯 What Goes Here

### CI/CD Pipelines
```
- build_and_test.bbx
- deploy_production.bbx
- rollback_deployment.bbx
```

### Infrastructure Management
```
- provision_environment.bbx
- scale_services.bbx
- backup_data.bbx
```

### Monitoring & Health
```
- health_check.bbx
- collect_metrics.bbx
- alert_on_failure.bbx
```

---

## ⚠️ Important

- **DO NOT** delete core workflows without team approval
- **TEST** thoroughly before modifying
- **DOCUMENT** any changes in git commits
- **VERSION** using workflow versioning system

---

## 📝 Creating New Core Workflows

1. Plan the workflow thoroughly
2. Test in staging environment
3. Review with team
4. Document usage and dependencies
5. Add to this README

---

## 🔗 See Also

- [User Workflows](../user/README.md) - Dynamic user automations
- [Meta Workflows](../meta/README.md) - BBX system workflows
