//! Task (Process) Definition for BBX OS
//!
//! A Task in BBX represents an agent's "thought" or execution context

use alloc::string::String;
use alloc::vec::Vec;

/// Task ID (like PID)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

/// Task state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Task is ready to run
    Ready,
    /// Task is currently running
    Running,
    /// Task is waiting for I/O or event
    Waiting,
    /// Task is blocked on resource
    Blocked,
    /// Task has completed
    Completed,
    /// Task is dead/terminated
    Dead,
}

/// Task priority (like AgentRing priorities)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskPriority {
    /// Low priority - background tasks
    Low = 0,
    /// Normal priority - regular tasks
    Normal = 1,
    /// High priority - important tasks
    High = 2,
    /// Realtime priority - critical tasks
    Realtime = 3,
}

/// Task structure
#[derive(Debug)]
pub struct Task {
    /// Unique task ID
    pub id: TaskId,

    /// Task name (agent name or workflow step)
    pub name: String,

    /// Current state
    pub state: TaskState,

    /// Priority level
    pub priority: TaskPriority,

    /// Time slice remaining (in ticks)
    pub time_slice: u32,

    /// Parent task ID
    pub parent: Option<TaskId>,

    /// Child task IDs
    pub children: Vec<TaskId>,

    /// Exit code (when completed)
    pub exit_code: Option<i32>,

    /// CPU time used (in ticks)
    pub cpu_time: u64,

    /// Memory usage (in bytes)
    pub memory_usage: usize,

    // === BBX-specific fields ===

    /// Workflow ID if this task is executing a workflow
    pub workflow_id: Option<String>,

    /// Current step in workflow
    pub current_step: Option<String>,

    /// Task dependencies (DAG-style)
    pub depends_on: Vec<TaskId>,

    /// Tasks depending on this one
    pub dependents: Vec<TaskId>,
}

impl Task {
    /// Create a new task
    pub fn new(id: TaskId, name: &str, priority: TaskPriority) -> Self {
        Task {
            id,
            name: String::from(name),
            state: TaskState::Ready,
            priority,
            time_slice: Self::time_slice_for_priority(priority),
            parent: None,
            children: Vec::new(),
            exit_code: None,
            cpu_time: 0,
            memory_usage: 0,
            workflow_id: None,
            current_step: None,
            depends_on: Vec::new(),
            dependents: Vec::new(),
        }
    }

    /// Get time slice for priority level
    fn time_slice_for_priority(priority: TaskPriority) -> u32 {
        match priority {
            TaskPriority::Low => 5,
            TaskPriority::Normal => 10,
            TaskPriority::High => 20,
            TaskPriority::Realtime => 50,
        }
    }

    /// Get default time slice for this task
    pub fn default_time_slice(&self) -> u32 {
        Self::time_slice_for_priority(self.priority)
    }

    /// Check if task can run (all dependencies satisfied)
    pub fn can_run(&self, completed_tasks: &[TaskId]) -> bool {
        self.depends_on.iter().all(|dep| completed_tasks.contains(dep))
    }

    /// Add dependency (DAG-style)
    pub fn add_dependency(&mut self, task_id: TaskId) {
        if !self.depends_on.contains(&task_id) {
            self.depends_on.push(task_id);
        }
    }

    /// Check if task is runnable
    pub fn is_runnable(&self) -> bool {
        matches!(self.state, TaskState::Ready | TaskState::Running)
    }
}
