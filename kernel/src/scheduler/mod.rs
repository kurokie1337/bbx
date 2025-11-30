//! BBX Scheduler - Agent-Centric Task Scheduling
//!
//! Inspired by BBX DAG Engine and AgentRing, but at kernel level

pub mod task;
pub mod ring;

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use spin::Mutex;
use lazy_static::lazy_static;

use task::{Task, TaskId, TaskState, TaskPriority};

/// Maximum number of concurrent tasks
const MAX_TASKS: usize = 1024;

lazy_static! {
    /// Global scheduler instance
    pub static ref SCHEDULER: Mutex<Scheduler> = Mutex::new(Scheduler::new());
}

/// BBX Scheduler
///
/// Agent-centric scheduler inspired by BBX's DAG engine
pub struct Scheduler {
    /// All tasks
    tasks: Vec<Option<Task>>,

    /// Ready queue (priority-based, like AgentRing)
    ready_queues: [VecDeque<TaskId>; 4],

    /// Currently running task
    current_task: Option<TaskId>,

    /// Next task ID
    next_id: u64,

    /// Tick counter
    ticks: u64,
}

impl Scheduler {
    /// Create new scheduler
    pub fn new() -> Self {
        Scheduler {
            tasks: Vec::with_capacity(MAX_TASKS),
            ready_queues: Default::default(),
            current_task: None,
            next_id: 1,
            ticks: 0,
        }
    }

    /// Create a new task (like spawning an agent thought)
    pub fn spawn(&mut self, name: &str, priority: TaskPriority) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;

        let task = Task::new(id, name, priority);

        // Find empty slot or push new
        let mut found = false;
        for slot in &mut self.tasks {
            if slot.is_none() {
                *slot = Some(task);
                found = true;
                break;
            }
        }
        if !found {
            self.tasks.push(Some(task));
        }

        // Add to ready queue
        self.ready_queues[priority as usize].push_back(id);

        crate::kprintln!("[SCHED] Spawned task {} ({})", id.0, name);

        id
    }

    /// Kill a task (like abandoning a thought)
    pub fn kill(&mut self, id: TaskId) {
        for slot in &mut self.tasks {
            if let Some(ref mut task) = slot {
                if task.id == id {
                    task.state = TaskState::Dead;
                    crate::kprintln!("[SCHED] Killed task {}", id.0);
                    break;
                }
            }
        }

        // Remove from ready queues
        for queue in &mut self.ready_queues {
            queue.retain(|&task_id| task_id != id);
        }
    }

    /// Get next task to run (priority-based selection like AgentRing)
    fn select_next(&mut self) -> Option<TaskId> {
        // Check queues from highest priority to lowest
        // REALTIME (3) -> HIGH (2) -> NORMAL (1) -> LOW (0)
        for priority in (0..4).rev() {
            if let Some(id) = self.ready_queues[priority].pop_front() {
                // Verify task is still runnable
                if let Some(task) = self.get_task(id) {
                    if task.state == TaskState::Ready {
                        return Some(id);
                    }
                }
            }
        }
        None
    }

    /// Get task by ID
    pub fn get_task(&self, id: TaskId) -> Option<&Task> {
        for slot in &self.tasks {
            if let Some(ref task) = slot {
                if task.id == id {
                    return Some(task);
                }
            }
        }
        None
    }

    /// Get mutable task by ID
    pub fn get_task_mut(&mut self, id: TaskId) -> Option<&mut Task> {
        for slot in &mut self.tasks {
            if let Some(ref mut task) = slot {
                if task.id == id {
                    return Some(task);
                }
            }
        }
        None
    }

    /// Scheduler tick (called on timer interrupt)
    pub fn tick(&mut self) {
        self.ticks += 1;

        // Check if we need to preempt current task
        if let Some(current_id) = self.current_task {
            if let Some(task) = self.get_task_mut(current_id) {
                task.time_slice -= 1;

                if task.time_slice == 0 {
                    // Time slice expired, move to ready queue
                    task.state = TaskState::Ready;
                    task.time_slice = task.default_time_slice();

                    self.ready_queues[task.priority as usize].push_back(current_id);
                    self.current_task = None;
                }
            }
        }

        // If no current task, select next
        if self.current_task.is_none() {
            if let Some(next_id) = self.select_next() {
                if let Some(task) = self.get_task_mut(next_id) {
                    task.state = TaskState::Running;
                }
                self.current_task = Some(next_id);
            }
        }
    }

    /// Get current task
    pub fn current(&self) -> Option<&Task> {
        self.current_task.and_then(|id| self.get_task(id))
    }

    /// List all tasks (like `bbx ps`)
    pub fn list_tasks(&self) -> Vec<&Task> {
        self.tasks.iter()
            .filter_map(|slot| slot.as_ref())
            .filter(|task| task.state != TaskState::Dead)
            .collect()
    }
}

/// Initialize scheduler
pub fn init() {
    // Spawn init task
    let mut sched = SCHEDULER.lock();
    sched.spawn("init", TaskPriority::Normal);
    sched.spawn("idle", TaskPriority::Low);
}

/// Scheduler tick (called from timer interrupt)
pub fn tick() {
    SCHEDULER.lock().tick();
}

/// Spawn a new task
pub fn spawn(name: &str, priority: TaskPriority) -> TaskId {
    SCHEDULER.lock().spawn(name, priority)
}

/// Kill a task
pub fn kill(id: TaskId) {
    SCHEDULER.lock().kill(id);
}

/// Get current task
pub fn current() -> Option<TaskId> {
    SCHEDULER.lock().current_task
}
