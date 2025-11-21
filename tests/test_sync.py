import pytest
from unittest.mock import Mock, patch, mock_open
from bbx_sync import WorkflowSyncer

@pytest.fixture
def mock_bbx_client():
    return Mock()

def test_load_local_workflow():
    syncer = WorkflowSyncer(Mock())
    yaml_content = """
    bbx_version: "5.0"
    type: workflow
    workflow:
      id: "test_id"
      name: "Test Name"
    """
    
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        with patch("pathlib.Path.exists", return_value=True):
            result = syncer.load_local_workflow("test.bbx")
            
    assert result["data"]["workflow"]["id"] == "test_id"
    assert result["data"]["workflow"]["name"] == "Test Name"

def test_sync_file_create(mock_bbx_client):
    # Setup
    syncer = WorkflowSyncer(mock_bbx_client)
    yaml_content = """
    bbx_version: "5.0"
    type: workflow
    workflow:
      id: "new_flow"
      name: "New Flow"
    """
    
    # Mock get_workflow to raise exception (not found)
    mock_bbx_client.get_workflow.side_effect = Exception("Not found")
    
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        with patch("pathlib.Path.exists", return_value=True):
            syncer.sync_file_to_blackbox("new.bbx")
            
    # Verify create_workflow was called
    mock_bbx_client.create_workflow.assert_called_once()
    args = mock_bbx_client.create_workflow.call_args[0][0]
    assert args.name == "New Flow"

def test_sync_file_update(mock_bbx_client):
    # Setup
    syncer = WorkflowSyncer(mock_bbx_client)
    yaml_content = """
    bbx_version: "5.0"
    type: workflow
    workflow:
      id: "existing_flow"
      name: "Updated Flow"
    """
    
    # Mock get_workflow to return existing
    mock_bbx_client.get_workflow.return_value = Mock(id="existing_flow")
    
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        with patch("pathlib.Path.exists", return_value=True):
            syncer.sync_file_to_blackbox("existing.bbx")
            
    # Verify update_workflow was called
    mock_bbx_client.update_workflow.assert_called_once()
    args = mock_bbx_client.update_workflow.call_args[0][1]
    assert args.name == "Updated Flow"
