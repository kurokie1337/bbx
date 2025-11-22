import React, { useCallback, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
} from 'reactflow';
import 'reactflow/dist/style.css';
import yaml from 'js-yaml';

interface WorkflowNode extends Node {
  data: {
    label: string;
    mcp: string;
    method: string;
    inputs: Record<string, any>;
  };
}

export function WorkflowDesigner() {
  const [nodes, setNodes, onNodesChange] = useNodesState<WorkflowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [workflowId, setWorkflowId] = useState('my_workflow');
  const [workflowName, setWorkflowName] = useState('My Workflow');

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const exportToYAML = () => {
    const workflow = {
      workflow: {
        id: workflowId,
        name: workflowName,
        version: '6.0',
        steps: nodes.map((node) => ({
          id: node.id,
          mcp: node.data.mcp,
          method: node.data.method,
          inputs: node.data.inputs,
          depends_on: edges
            .filter((e) => e.target === node.id)
            .map((e) => e.source),
        })),
      },
    };

    const yamlStr = yaml.dump(workflow);
    const blob = new Blob([yamlStr], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `{}.bbx`;
    a.click();
  };

  const addHttpStep = () => {
    const newNode: WorkflowNode = {
      id: `step_{}`,
      type: 'default',
      position: { x: 100 + nodes.length * 50, y: 100 },
      data: {
        label: 'HTTP Request',
        mcp: 'bbx.http',
        method: 'get',
        inputs: { url: 'https://api.example.com' },
      },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <div style={{ padding: '10px', background: '#f0f0f0' }}>
        <input
          placeholder="Workflow ID"
          value={workflowId}
          onChange={(e) => setWorkflowId(e.target.value)}
        />
        <input
          placeholder="Workflow Name"
          value={workflowName}
          onChange={(e) => setWorkflowName(e.target.value)}
        />
        <button onClick={addHttpStep}>Add HTTP Step</button>
        <button onClick={exportToYAML}>Export to YAML</button>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
      >
        <Controls />
        <Background />
      </ReactFlow>
    </div>
  );
}
