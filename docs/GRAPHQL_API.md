# GraphQL API

BBX provides a GraphQL API alongside REST API.

## Endpoint

\`\`\`
POST /graphql
\`\`\`

## Playground

\`\`\`
http://localhost:8000/graphql/playground
\`\`\`

## Example Queries

### Get all workflows
\`\`\`graphql
query {
    workflows {
        id
        name
        version
        steps {
            id
            mcp
            method
        }
    }
}
\`\`\`

### Execute workflow
\`\`\`graphql
mutation {
    executeWorkflow(workflowId: "my_workflow", inputs: "{}") {
        id
        status
        startedAt
    }
}
\`\`\`

### Subscribe to updates
\`\`\`graphql
subscription {
    workflowUpdates(workflowId: "my_workflow") {
        id
        status
        completedAt
    }
}
\`\`\`

## Python Client

\`\`\`python
import requests

query = "{ workflows { id name } }"
response = requests.post(
    "http://localhost:8000/graphql",
    json={"query": query}
)
\`\`\`
