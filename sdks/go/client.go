// Blackbox Go SDK
package blackbox

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
	Token      string
}

type WorkflowCreate struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	BBXYaml     string `json:"bbx_yaml"`
}

type WorkflowResponse struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
}

type ExecutionResponse struct {
	ID         string                 `json:"id"`
	WorkflowID string                 `json:"workflow_id"`
	Status     string                 `json:"status"`
	StartedAt  time.Time              `json:"started_at"`
	Outputs    map[string]interface{} `json:"outputs"`
}

// NewClient creates a new Blackbox API client
func NewClient(baseURL string) *Client {
	return &Client{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Authenticate logs in and sets the auth token
func (c *Client) Authenticate(username, password string) error {
	body := map[string]string{
		"username": username,
		"password": password,
	}
	
	jsonBody, _ := json.Marshal(body)
	resp, err := c.HTTPClient.Post(
		c.BaseURL+"/auth/login",
		"application/json",
		bytes.NewBuffer(jsonBody),
	)
	
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return err
	}
	
	c.Token = result["access_token"].(string)
	return nil
}

// CreateWorkflow creates a new workflow
func (c *Client) CreateWorkflow(workflow WorkflowCreate) (*WorkflowResponse, error) {
	jsonBody, _ := json.Marshal(workflow)
	
	req, _ := http.NewRequest("POST", c.BaseURL+"/api/workflows", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.Token)
	
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var workflowResp WorkflowResponse
	if err := json.NewDecoder(resp.Body).Decode(&workflowResp); err != nil {
		return nil, err
	}
	
	return &workflowResp, nil
}

// ExecuteWorkflow executes a workflow
func (c *Client) ExecuteWorkflow(workflowID string, inputs map[string]interface{}) (*ExecutionResponse, error) {
	body := map[string]interface{}{
		"inputs": inputs,
	}
	
	jsonBody, _ := json.Marshal(body)
	
	req, _ := http.NewRequest(
		"POST",
		fmt.Sprintf("%s/api/execute/%s", c.BaseURL, workflowID),
		bytes.NewBuffer(jsonBody),
	)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.Token)
	
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var execution ExecutionResponse
	if err := json.NewDecoder(resp.Body).Decode(&execution); err != nil {
		return nil, err
	}
	
	return &execution, nil
}

// GetExecutionStatus gets the status of a workflow execution
func (c *Client) GetExecutionStatus(executionID string) (*ExecutionResponse, error) {
	req, _ := http.NewRequest(
		"GET",
		fmt.Sprintf("%s/api/executions/%s", c.BaseURL, executionID),
		nil,
	)
	req.Header.Set("Authorization", "Bearer "+c.Token)
	
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var execution ExecutionResponse
	if err := json.NewDecoder(resp.Body).Decode(&execution); err != nil {
		return nil, err
	}
	
	return &execution, nil
}

/* Example usage:

package main

import (
	"fmt"
	"log"
	"github.com/blackbox/go-sdk"
)

func main() {
	client := blackbox.NewClient("https://api.blackbox.dev")
	
	// Authenticate
	if err := client.Authenticate("user", "password"); err != nil {
		log.Fatal(err)
	}
	
	// Create workflow
	workflow, err := client.CreateWorkflow(blackbox.WorkflowCreate{
		Name: "My Workflow",
		Description: "Test workflow",
		BBXYaml: `
id: test_workflow
steps:
  fetch_data:
    use: http.get
    args:
      url: https://api.example.com/data
`,
	})
	
	if err != nil {
		log.Fatal(err)
	}
	
	// Execute workflow
	execution, err := client.ExecuteWorkflow(workflow.ID, map[string]interface{}{
		"param": "value",
	})
	
	if err != nil {
		log.Fatal(err)
	}
	
	fmt.Printf("Execution ID: %s\n", execution.ID)
}
*/
