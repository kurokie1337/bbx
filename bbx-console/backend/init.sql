-- BBX Console Database Initialization Script
-- This script runs automatically when PostgreSQL container starts

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pg_trgm for text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create custom types
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'execution_status') THEN
        CREATE TYPE execution_status AS ENUM (
            'pending', 'running', 'completed', 'failed', 'cancelled', 'timeout'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'task_status') THEN
        CREATE TYPE task_status AS ENUM (
            'backlog', 'todo', 'in_progress', 'review', 'done', 'blocked'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'task_priority') THEN
        CREATE TYPE task_priority AS ENUM (
            'critical', 'high', 'medium', 'low'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'log_level') THEN
        CREATE TYPE log_level AS ENUM (
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        );
    END IF;
END
$$;

-- Executions table
CREATE TABLE IF NOT EXISTS executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_name VARCHAR(255) NOT NULL,
    workflow_path VARCHAR(1024),
    status execution_status DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    inputs JSONB DEFAULT '{}',
    outputs JSONB DEFAULT '{}',
    error TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_workflow_name ON executions(workflow_name);
CREATE INDEX IF NOT EXISTS idx_executions_started_at ON executions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_executions_created_at ON executions(created_at DESC);

-- Step logs table
CREATE TABLE IF NOT EXISTS step_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    step_id VARCHAR(255) NOT NULL,
    step_name VARCHAR(255),
    status execution_status DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    inputs JSONB DEFAULT '{}',
    outputs JSONB DEFAULT '{}',
    error TEXT,
    logs TEXT[],
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for step logs
CREATE INDEX IF NOT EXISTS idx_step_logs_execution_id ON step_logs(execution_id);
CREATE INDEX IF NOT EXISTS idx_step_logs_step_id ON step_logs(step_id);
CREATE INDEX IF NOT EXISTS idx_step_logs_status ON step_logs(status);

-- Tasks table (for Task Manager)
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    status task_status DEFAULT 'backlog',
    priority task_priority DEFAULT 'medium',
    assigned_agent VARCHAR(255),
    parent_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    execution_id UUID REFERENCES executions(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    due_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for tasks
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_agent ON tasks(assigned_agent);
CREATE INDEX IF NOT EXISTS idx_tasks_parent_id ON tasks(parent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC);

-- Agent metrics table
CREATE TABLE IF NOT EXISTS agent_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tasks_completed INTEGER DEFAULT 0,
    tasks_failed INTEGER DEFAULT 0,
    avg_execution_time_ms FLOAT DEFAULT 0,
    total_execution_time_ms BIGINT DEFAULT 0,
    memory_usage_mb FLOAT DEFAULT 0,
    cpu_usage_percent FLOAT DEFAULT 0,
    tokens_used BIGINT DEFAULT 0,
    tokens_input BIGINT DEFAULT 0,
    tokens_output BIGINT DEFAULT 0,
    error_rate FLOAT DEFAULT 0,
    success_rate FLOAT DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for agent metrics
CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent_id ON agent_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_timestamp ON agent_metrics(timestamp DESC);

-- Partitioning for agent metrics (for high volume)
-- CREATE TABLE IF NOT EXISTS agent_metrics_y2024m01 PARTITION OF agent_metrics
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DROP TRIGGER IF EXISTS update_executions_updated_at ON executions;
CREATE TRIGGER update_executions_updated_at
    BEFORE UPDATE ON executions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;
CREATE TRIGGER update_tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for development
INSERT INTO tasks (title, description, status, priority) VALUES
    ('Setup BBX Console', 'Initialize the BBX Console project with all components', 'done', 'high'),
    ('Implement WebSocket', 'Add real-time updates via WebSocket', 'done', 'high'),
    ('Create DAG Visualizer', 'Build workflow DAG visualization component', 'in_progress', 'medium'),
    ('Add Agent Monitoring', 'Implement agent status and metrics monitoring', 'todo', 'medium'),
    ('Integrate A2A Protocol', 'Add Agent-to-Agent protocol support', 'backlog', 'low')
ON CONFLICT DO NOTHING;

-- Grant permissions (if using separate users)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO bbx;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO bbx;

-- Vacuum analyze for performance
ANALYZE executions;
ANALYZE step_logs;
ANALYZE tasks;
ANALYZE agent_metrics;
