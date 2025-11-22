package bbx

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

type Client struct {
    BaseURL string
    APIKey  string
    client  *http.Client
}

type WorkflowExecution struct {
    ID         string `json:"id"`
    WorkflowID string `json:"workflow_id"`
    Status     string `json:"status"`
    Outputs    map[string]interface{} `json:"outputs"`
}

func NewClient(baseURL, apiKey string) *Client {
    return &Client{
        BaseURL: baseURL,
        APIKey:  apiKey,
        client:  &http.Client{},
    }
}

func (c *Client) ExecuteWorkflow(workflowPath string, inputs map[string]interface{}) (*WorkflowExecution, error) {
    data := map[string]interface{}{
        "workflow_path": workflowPath,
        "inputs":        inputs,
    }
    body, _ := json.Marshal(data)

    req, _ := http.NewRequest("POST", c.BaseURL+"/api/workflows/execute", bytes.NewBuffer(body))
    req.Header.Set("Content-Type", "application/json")
    if c.APIKey != "" {
        req.Header.Set("Authorization", "Bearer "+c.APIKey)
    }

    resp, err := c.client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var execution WorkflowExecution
    if err := json.NewDecoder(resp.Body).Decode(&execution); err != nil {
        return nil, err
    }

    return &execution, nil
}
