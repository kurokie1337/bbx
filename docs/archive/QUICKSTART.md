# BBX Universal Adapter - Quick Start Guide

## 🚀 Installation

```bash
cd workflow_test
pip install -r requirements.txt
```

## 📦 Package Management

### List Available Packages
```bash
python cli.py package list
```

### Install a Package
```bash
python cli.py package install terraform
python cli.py package install kubectl
python cli.py package install aws_cli
```

### View Package Info
```bash
python cli.py package info terraform
```

### Validate All Packages
```bash
python cli.py package validate
```

## 🎯 Using Universal Adapter

### Method 1: Direct Usage (Dynamic Mode)

Create a workflow file `my_workflow.bbx`:
```yaml
name: My Workflow
version: 1.0

steps:
  - id: run_terraform
    mcp: universal
    method: run
    inputs:
      uses: docker://hashicorp/terraform:latest
      cmd:
        - terraform
        - version
      output_parser:
        type: text
```

Run it:
```bash
python cli.py run my_workflow.bbx
```

### Method 2: Using Library Definitions

```yaml
name: Kubernetes Deployment
version: 1.0

steps:
  - id: apply_config
    mcp: universal
    method: run
    definition: blackbox/library/k8s_apply.yaml
    inputs:
      file: deployment.yaml
      namespace: production
```

## 🔍 Security Scanning

Scan images before using them:
```bash
python cli.py security-scan python:3.11
python cli.py security-scan hashicorp/terraform:latest
python cli.py security-scan --severity CRITICAL alpine:latest
```

## 📊 Audit Logs

View execution history:
```bash
# Last 7 days (all adapters)
python cli.py audit

# Last 30 days
python cli.py audit --days 30

# Filter by adapter
python cli.py audit --adapter terraform --days 7
```

## 🔐 Private Registries

### GitHub Container Registry
```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
python cli.py registry-login ghcr.io --token $GITHUB_TOKEN
```

### AWS ECR
```bash
# Ensure AWS credentials are configured
aws configure

# Login (automatically uses AWS CLI)
python -c "from blackbox.core.registry_auth import PrivateRegistryAuth; PrivateRegistryAuth().login_ecr('us-east-1')"
```

### Google Container Registry
```bash
# Ensure gcloud is authenticated
gcloud auth login

# Login
python -c "from blackbox.core.registry_auth import PrivateRegistryAuth; PrivateRegistryAuth().login_gcr('my-project-id')"
```

## 🔄 Multi-Step Workflows

Create complex pipelines in a single definition:

```yaml
name: CI/CD Pipeline
version: 1.0

steps:
  - id: build_and_test
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      steps:
        - name: "Lint"
          cmd: ["echo", "Running linter..."]
        - name: "Test"
          cmd: ["echo", "Running tests..."]
          continue_on_error: false
        - name: "Build"
          cmd: ["echo", "Building..."]
      resources:
        cpu: "2"
        memory: "1g"
```

## 📈 Performance Tips

### 1. Pre-pull Images
```bash
docker pull hashicorp/terraform:latest
docker pull bitnami/kubectl:latest
docker pull amazon/aws-cli:latest
```

### 2. Use Specific Tags
Instead of:
```yaml
uses: docker://python:latest
```

Use:
```yaml
uses: docker://python:3.11-alpine
```

### 3. Enable Package Caching
All installed packages are automatically cached by the package manager.

## 🛠️ Creating Custom Definitions

1. Create a YAML file in `blackbox/library/`:

```yaml
# blackbox/library/mytool.yaml
id: mytool
uses: docker://myorg/mytool:latest
cmd:
  - mytool
  - {{ inputs.action }}
  - --config={{ inputs.config }}
output_parser:
  type: json
```

2. Validate it:
```bash
python cli.py package validate
```

3. Use it:
```yaml
- id: run_mytool
  mcp: universal
  method: run
  inputs:
    action: deploy
    config: config.yaml
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/test_universal_e2e.py -v
```

Run specific tests:
```bash
pytest tests/test_universal_e2e.py::test_package_manager_install -v
pytest tests/test_universal_e2e.py::test_multi_step_execution -v
```

## 📚 Examples

Check out the `examples/` directory:
- `poc_simple_alpine.bbx` - Simple test
- `poc_universal_kubectl.bbx` - Kubernetes version check
- `universal_database_migration.bbx` - PostgreSQL migrations
- `universal_gcp_database.bbx` - GCP Cloud SQL operations
- `pipeline_cicd.bbx` - Full CI/CD pipeline
- `universal_multi_step.bbx` - Terraform multi-step deployment

## 🔥 Common Workflows

### AWS CLI
```yaml
- id: list_s3_buckets
  mcp: universal
  method: run
  inputs:
    uses: docker://amazon/aws-cli:latest
    auth:
      type: aws_credentials
    cmd:
      - aws
      - s3
      - ls
      - --output=json
    output_parser:
      type: json
```

### Terraform
```yaml
- id: terraform_plan
  mcp: universal
  method: run
  inputs:
    uses: docker://hashicorp/terraform:latest
    auth:
      type: aws_credentials
    cmd:
      - terraform
      - plan
      - -out=tfplan
    working_dir: ./terraform
```

### Kubernetes
```yaml
- id: get_pods
  mcp: universal
  method: run
  inputs:
    uses: docker://bitnami/kubectl:latest
    auth:
      type: kubeconfig
    cmd:
      - kubectl
      - get
      - pods
      - -n
      - default
      - -o
      - json
    output_parser:
      type: json
```

## ❓ Troubleshooting

### Issue: "Docker image not found"
**Solution:** Pre-pull the image or check your internet connection.

### Issue: "Authentication failed"
**Solution:** Check your credentials are set in environment variables or files.

### Issue: "Subprocess hanging"
**Solution:** Use `universal_v2.py` which has real-time streaming, or pre-pull large images.

### Issue: "Permission denied"
**Solution:** Ensure Docker daemon is running and you have permissions.

## 📖 Full Documentation

- [Phase 1 Report](../.gemini/antigravity/brain/45781443-87d1-492a-be3e-a56ebcc0c718/phase1_completion.md)
- [Phase 2 Report](../.gemini/antigravity/brain/45781443-87d1-492a-be3e-a56ebcc0c718/phase2_completion.md)
- [Phase 3 Report](../.gemini/antigravity/brain/45781443-87d1-492a-be3e-a56ebcc0c718/phase3_completion.md)
- [Universal Adapter Guide](../.gemini/antigravity/brain/45781443-87d1-492a-be3e-a56ebcc0c718/universal_adapter_guide.md)
- [CLI Integration](CLI_INTEGRATION.md)

## 🎯 Next Steps

1. Install your first package: `python cli.py package install terraform`
2. Run a simple test: `python cli.py run examples/poc_simple_alpine.bbx`
3. Scan an image: `python cli.py security-scan alpine:latest`
4. Check audit logs: `python cli.py audit`
5. Create your own definition in `blackbox/library/`

---

**BBX Universal Adapter - Zero Python. Infinite Possibilities.** 🚀
