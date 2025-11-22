# BBX Rust SDK

\`\`\`rust
use bbx_sdk::Client;
use std::collections::HashMap;

#[tokio::main]
async fn main() {
    let client = Client::new("http://localhost:8000".to_string(), None);
    let inputs = HashMap::new();

    let execution = client.execute_workflow("my_workflow.bbx", inputs).await.unwrap();
    println!("Status: {}", execution.status);
}
\`\`\`
