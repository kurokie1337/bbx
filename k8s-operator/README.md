# BBX Kubernetes Operator

Native Kubernetes integration for BBX workflows.

## Installation

\`\`\`bash
helm install bbx-operator ./helm/bbx-operator
\`\`\`

## Usage

\`\`\`yaml
apiVersion: bbx.io/v1alpha1
kind: Workflow
metadata:
  name: my-workflow
spec:
  workflowSpec:
    workflow:
      id: my_workflow
      version: "6.0"
      steps:
        - id: step1
          mcp: bbx.http
          method: get
          inputs:
            url: https://api.example.com
\`\`\`

## Check status

\`\`\`bash
kubectl get workflows
kubectl describe workflow my-workflow
\`\`\`
