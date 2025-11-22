# BBX Go SDK

\`\`\`go
import "github.com/kurokie1337/bbx-go-sdk"

client := bbx.NewClient("http://localhost:8000", "")
execution, err := client.ExecuteWorkflow("my_workflow.bbx", map[string]interface{}{
    "param1": "value1",
})
\`\`\`
