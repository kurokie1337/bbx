# BBX Standard Library

**The "Zero-Code" Recipe Collection for Cloud & Infrastructure Automation**

This is the BBX Standard Library - a curated collection of ready-to-use YAML recipes that replace the old Python adapters. Instead of maintaining hundreds of Python classes, BBX now uses the **Universal Adapter** with official Docker images.

## Philosophy

> **"No Magic, Just YAML"**

- ✅ **Declarative**: You describe *what* you want, not *how* to do it
- ✅ **Portable**: Uses official Docker images (no custom code)
- ✅ **Transparent**: Every command is visible in the YAML
- ✅ **AI-Friendly**: LLMs can generate and understand YAML easily
- ✅ **Composable**: Mix and match recipes like LEGO blocks

## Structure

```
library/
├── terraform/       # Infrastructure as Code
│   ├── init.yaml
│   ├── plan.yaml
│   ├── apply.yaml
│   └── output.yaml
│
├── kubernetes/      # Container Orchestration
│   ├── apply.yaml
│   ├── get.yaml
│   ├── rollout.yaml
│   └── scale.yaml
│
├── aws/            # AWS Cloud
│   ├── s3-upload.yaml
│   ├── ec2-list.yaml
│   └── lambda-invoke.yaml
│
├── git/            # Version Control
│   ├── clone.yaml
│   ├── tag.yaml
│   └── commit.yaml
│
├── docker/         # Containers
└── database/       # Database Operations
```

## How to Use

### Option 1: Copy-Paste (Simplest)

Open any recipe file (e.g., `terraform/init.yaml`) and copy the example usage into your workflow:

```yaml
steps:
  - id: init_terraform
    mcp: universal
    method: run
    inputs:
      uses: docker://hashicorp/terraform:latest
      working_dir: "./infrastructure"
      cmd:
        - terraform
        - init
        - -upgrade
```

### Option 2: Reference (Future Feature)

In the future, BBX will support `include:` directive:

```yaml
steps:
  - id: init
    include: library/terraform/init.yaml
    inputs:
      working_dir: "./infra"
```

## Examples

### Terraform Pipeline

```yaml
workflow:
  id: terraform_deploy

  steps:
    # Initialize
    - id: init
      mcp: universal
      method: run
      inputs:
        uses: docker://hashicorp/terraform:latest
        working_dir: "./terraform"
        cmd: [terraform, init, -upgrade]

    # Plan
    - id: plan
      mcp: universal
      method: run
      inputs:
        uses: docker://hashicorp/terraform:latest
        working_dir: "./terraform"
        cmd: [terraform, plan, -out=tfplan]
      depends_on: [init]

    # Apply
    - id: apply
      mcp: universal
      method: run
      inputs:
        uses: docker://hashicorp/terraform:latest
        working_dir: "./terraform"
        cmd: [terraform, apply, -auto-approve, tfplan]
      depends_on: [plan]
```

### Kubernetes Deployment

```yaml
workflow:
  id: k8s_deploy

  steps:
    - id: apply_manifest
      mcp: universal
      method: run
      inputs:
        uses: docker://bitnami/kubectl:latest
        cmd:
          - kubectl
          - apply
          - -f
          - k8s/deployment.yaml
          - -n
          - production

    - id: wait_rollout
      mcp: universal
      method: run
      inputs:
        uses: docker://bitnami/kubectl:latest
        cmd:
          - kubectl
          - rollout
          - status
          - deployment/backend
          - -n
          - production
      depends_on: [apply_manifest]
```

### AWS S3 Upload

```yaml
steps:
  - id: upload_build
    mcp: universal
    method: run
    inputs:
      uses: docker://amazon/aws-cli:latest
      cmd:
        - aws
        - s3
        - cp
        - ./dist/app.zip
        - s3://my-bucket/releases/v1.0.0/app.zip
        - --region
        - us-west-2
      auth:
        type: aws_credentials
```

## Docker Images Reference

| Tool       | Official Image                        | Description              |
|------------|---------------------------------------|--------------------------|
| Terraform  | `hashicorp/terraform:latest`          | Infrastructure as Code   |
| Kubectl    | `bitnami/kubectl:latest`              | Kubernetes CLI           |
| AWS CLI    | `amazon/aws-cli:latest`               | AWS Command Line         |
| GCP CLI    | `google/cloud-sdk:latest`             | Google Cloud SDK         |
| Azure CLI  | `mcr.microsoft.com/azure-cli:latest`  | Azure CLI                |
| Git        | `alpine/git:latest`                   | Git version control      |
| Docker     | `docker:latest`                       | Docker CLI               |

## Migration Guide

### Old Way (Python Adapter)

```yaml
- id: terraform_init
  mcp: bbx.terraform
  method: init
  inputs:
    working_dir: "./terraform"
    upgrade: true
```

### New Way (Universal Adapter)

```yaml
- id: terraform_init
  mcp: universal
  method: run
  inputs:
    uses: docker://hashicorp/terraform:latest
    working_dir: "./terraform"
    cmd:
      - terraform
      - init
      - -upgrade
```

**What Changed?**
- ❌ No more `bbx.terraform` Python class
- ✅ Direct Docker image reference
- ✅ Explicit command template
- ✅ Fully transparent and configurable

## Benefits

### For Users
- **No Installation**: Just need Docker
- **Always Latest**: Use any version of any tool
- **No Vendor Lock-in**: Standard Docker images
- **Copy-Paste Ready**: Examples work out of the box

### For AI Agents
- **Token Efficient**: YAML is compact
- **Easy to Generate**: No complex Python syntax
- **Self-Documented**: Every command is visible
- **Composable**: Mix tools freely

### For Maintainers
- **Less Code**: ~8 Python files deleted
- **No Breaking Changes**: Update Docker image, not code
- **Community Driven**: Anyone can add recipes
- **Zero Bugs**: No custom logic to maintain

## Contributing

Want to add a new recipe?

1. Pick a tool with an official Docker image
2. Create `library/<category>/<operation>.yaml`
3. Follow the template format
4. Add usage examples
5. Submit a PR

## Future Roadmap

- [ ] `include:` directive for recipe imports
- [ ] Recipe validation tool
- [ ] Community recipe hub
- [ ] Auto-generation from Docker images
- [ ] Version pinning and compatibility matrix

---

**BBX Standard Library**: From "Python Classes" to "YAML Recipes"

*"Make infrastructure automation boring again."*
