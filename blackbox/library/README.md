# BBX Universal Adapter Library

## 📚 Overview
This directory contains YAML definitions for the Universal Adapter architecture. Each file defines how to execute a specific CLI tool inside Docker without writing Python code.

## 🎯 Philosophy
**"Zero-Code Adapters"** - Replace 1,400+ lines of Python boilerplate with simple, declarative YAML configurations.

## 📁 Available Definitions

### Cloud Providers
- **[aws_cli.yaml](aws_cli.yaml)** - AWS CLI
- **[gcloud.yaml](gcloud.yaml)** - Google Cloud SDK
- **[az.yaml](az.yaml)** - Azure CLI

### Infrastructure as Code
- **[terraform.yaml](tf_plan.yaml)** - Terraform operations
- **[ansible.yaml](ansible.yaml)** - Ansible playbooks
- **[helm.yaml](helm.yaml)** - Helm chart management

### Container Orchestration
- **[k8s_apply.yaml](k8s_apply.yaml)** - Kubernetes apply
- **[docker_cli.yaml](docker_cli.yaml)** - Docker CLI (DinD)

### Databases
- **[psql.yaml](psql.yaml)** - PostgreSQL client
- **[mongosh.yaml](mongosh.yaml)** - MongoDB shell
- **[redis.yaml](redis.yaml)** - Redis CLI

## 🚀 Quick Start

### Example 1: Use in Workflow (Dynamic Mode)
```yaml
- id: deploy_to_k8s
  mcp: universal
  method: run
  inputs:
    uses: docker://bitnami/kubectl:latest
    auth:
      type: kubeconfig
    cmd:
      - kubectl
      - apply
      - -f
      - deployment.yaml
      - -n
      - production
```

### Example 2: Reference Library Definition
```yaml
# First, reference the definition
- id: deploy_to_k8s
  mcp: universal
  method: run
  definition: blackbox/library/k8s_apply.yaml
  inputs:
    file: deployment.yaml
    namespace: production
```

## 📖 Schema Reference

### Required Fields
- **id**: Unique identifier
- **uses**: Docker image (format: `docker://image:tag`)
- **cmd**: Command template (Jinja2 list)

### Optional Fields
- **auth**: Authentication provider config
- **env**: Environment variables
- **volumes**: Volume mounts
- **workdir**: Working directory
- **output_parser**: Output parsing config

## 🔐 Authentication Providers

### Kubernetes
```yaml
auth:
  type: kubeconfig  # Auto-mounts ~/.kube/config
```

### AWS
```yaml
auth:
  type: aws_credentials  # Injects AWS_* env vars + mounts ~/.aws
```

### GCP
```yaml
auth:
  type: gcp_credentials  # Mounts service account JSON
```

## 📊 Impact Metrics

| Adapter | Old (Python) | New (YAML) | Reduction |
|---------|--------------|------------|-----------|
| Terraform | 371 lines | 15 lines | **96%** |
| Kubernetes | 591 lines | 12 lines | **98%** |
| AWS | 474 lines | 11 lines | **98%** |
| **Total** | **1,436 lines** | **38 lines** | **97.4%** |

## 🎓 Creating New Definitions

See [universal_adapter_guide.md](../../.gemini/antigravity/brain/45781443-87d1-492a-be3e-a56ebcc0c718/universal_adapter_guide.md) for detailed instructions.

### Quick Template
```yaml
id: my_tool
uses: docker://namespace/image:tag
auth:
  type: provider_name  # Optional
cmd:
  - tool
  - {{ inputs.param }}
  - {% if inputs.optional %}--flag={{ inputs.optional }}{% endif %}
env:
  VAR: "{{ secrets.SECRET }}"
output_parser:
  type: json|text
  query: "jmespath"  # Optional, for JSON
```

## 🧪 Testing

Test any definition with:
```bash
python cli.py run examples/poc_universal_kubectl.bbx
```

Or create your own test workflow:
```yaml
name: Test My Definition
version: 1.0
steps:
  - id: test
    mcp: universal
    method: run
    inputs:
      uses: docker://your/image:tag
      cmd: [your, command]
```

## 🔧 Troubleshooting

### "No Docker image specified"
**Solution:** Ensure `uses` is present in definition or workflow inputs.

### "Missing input for template"
**Solution:** Check that all `{{ inputs.x }}` variables are provided.

### "Failed to parse JSON output"
**Solution:** Verify CLI actually outputs JSON. Add `--output=json` flag.

### Subprocess still hanging
**Solution:** Pre-pull large images: `docker pull image:tag`

## 📚 Additional Resources

- [Phase 1 Completion Report](../../.gemini/antigravity/brain/45781443-87d1-492a-be3e-a56ebcc0c718/phase1_completion.md)
- [Universal Adapter Guide](../../.gemini/antigravity/brain/45781443-87d1-492a-be3e-a56ebcc0c718/universal_adapter_guide.md)
- [Technical Proof of Concept](../../.gemini/antigravity/brain/45781443-87d1-492a-be3e-a56ebcc0c718/technical_proof_of_concept.md)

## 🌟 Contributing

1. Create YAML file in this directory
2. Test with example workflow
3. Document in this README
4. Submit for review

## 📄 License
Apache 2.0 - Copyright 2025 Ilya Makarov, Krasnoyarsk
