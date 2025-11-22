use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowExecution {
    pub id: String,
    pub workflow_id: String,
    pub status: String,
    pub outputs: Option<HashMap<String, serde_json::Value>>,
}

pub struct Client {
    base_url: String,
    api_key: Option<String>,
    client: HttpClient,
}

impl Client {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self {
            base_url,
            api_key,
            client: HttpClient::new(),
        }
    }

    pub async fn execute_workflow(
        &self,
        workflow_path: &str,
        inputs: HashMap<String, serde_json::Value>,
    ) -> Result<WorkflowExecution, Box<dyn std::error::Error>> {
        let url = format!("{}/api/workflows/execute", self.base_url);

        let mut request = self.client
            .post(&url)
            .json(&serde_json::json!({
                "workflow_path": workflow_path,
                "inputs": inputs
            }));

        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        let response = request.send().await?;
        let execution = response.json::<WorkflowExecution>().await?;

        Ok(execution)
    }
}
