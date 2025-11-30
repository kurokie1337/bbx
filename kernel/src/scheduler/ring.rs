//! AgentRing - Kernel-level io_uring-inspired batch operations
//!
//! This is the kernel implementation of BBX's AgentRing concept

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use spin::Mutex;
use lazy_static::lazy_static;

/// Operation ID
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpId(pub u64);

/// Operation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum OpPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Realtime = 3,
}

/// Operation type
#[derive(Debug, Clone, Copy)]
pub enum OpType {
    /// Read from file/device
    Read,
    /// Write to file/device
    Write,
    /// Network operation
    Network,
    /// Timer/sleep
    Timer,
    /// Memory operation
    Memory,
    /// Agent communication
    AgentCall,
    /// Custom operation
    Custom(u32),
}

/// Operation state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpState {
    /// Submitted, waiting in queue
    Pending,
    /// Currently being processed
    Processing,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed,
    /// Cancelled
    Cancelled,
}

/// Operation in the ring
#[derive(Debug)]
pub struct Operation {
    pub id: OpId,
    pub op_type: OpType,
    pub priority: OpPriority,
    pub state: OpState,
    pub data: u64,         // Operation-specific data
    pub result: i64,       // Result or error code
    pub user_data: u64,    // User context
}

/// Completion entry
#[derive(Debug, Clone)]
pub struct Completion {
    pub op_id: OpId,
    pub result: i64,
    pub user_data: u64,
}

lazy_static! {
    /// Global kernel AgentRing
    pub static ref AGENT_RING: Mutex<KernelRing> = Mutex::new(KernelRing::new());
}

/// Kernel-level AgentRing (io_uring-inspired)
pub struct KernelRing {
    /// Submission queue
    submission_queue: VecDeque<Operation>,

    /// Completion queue
    completion_queue: VecDeque<Completion>,

    /// Processing operations
    processing: Vec<Operation>,

    /// Next operation ID
    next_id: u64,

    /// Statistics
    stats: RingStats,
}

/// Ring statistics
#[derive(Debug, Default)]
pub struct RingStats {
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
}

impl KernelRing {
    /// Create new kernel ring
    pub fn new() -> Self {
        KernelRing {
            submission_queue: VecDeque::with_capacity(4096),
            completion_queue: VecDeque::with_capacity(4096),
            processing: Vec::with_capacity(64),
            next_id: 1,
            stats: RingStats::default(),
        }
    }

    /// Submit operation to ring
    pub fn submit(&mut self, op_type: OpType, priority: OpPriority, data: u64, user_data: u64) -> OpId {
        let id = OpId(self.next_id);
        self.next_id += 1;

        let op = Operation {
            id,
            op_type,
            priority,
            state: OpState::Pending,
            data,
            result: 0,
            user_data,
        };

        // Insert by priority (higher priority = front)
        let mut inserted = false;
        for i in 0..self.submission_queue.len() {
            if self.submission_queue[i].priority < priority {
                self.submission_queue.insert(i, op);
                inserted = true;
                break;
            }
        }
        if !inserted {
            self.submission_queue.push_back(op);
        }

        self.stats.submitted += 1;
        id
    }

    /// Submit batch of operations
    pub fn submit_batch(&mut self, ops: Vec<(OpType, OpPriority, u64, u64)>) -> Vec<OpId> {
        ops.into_iter()
            .map(|(op_type, priority, data, user_data)| {
                self.submit(op_type, priority, data, user_data)
            })
            .collect()
    }

    /// Process pending operations
    pub fn process(&mut self) {
        // Move pending ops to processing (up to worker limit)
        while self.processing.len() < 32 && !self.submission_queue.is_empty() {
            if let Some(mut op) = self.submission_queue.pop_front() {
                op.state = OpState::Processing;
                self.processing.push(op);
            }
        }

        // Process operations (in real kernel, would dispatch to drivers)
        let mut completed = Vec::new();
        for op in &mut self.processing {
            // Simulate operation completion
            op.state = OpState::Completed;
            op.result = 0; // Success

            completed.push(op.id);

            // Add to completion queue
            self.completion_queue.push_back(Completion {
                op_id: op.id,
                result: op.result,
                user_data: op.user_data,
            });

            self.stats.completed += 1;
        }

        // Remove completed from processing
        self.processing.retain(|op| !completed.contains(&op.id));
    }

    /// Wait for completion
    pub fn wait(&mut self, op_id: OpId) -> Option<Completion> {
        // Process until our op is done
        loop {
            self.process();

            // Check completion queue
            for i in 0..self.completion_queue.len() {
                if self.completion_queue[i].op_id == op_id {
                    return self.completion_queue.remove(i);
                }
            }

            // Check if op is still pending/processing
            let still_active = self.submission_queue.iter().any(|op| op.id == op_id)
                || self.processing.iter().any(|op| op.id == op_id);

            if !still_active {
                return None;
            }

            // In real kernel, would yield or sleep here
        }
    }

    /// Peek completions without removing
    pub fn peek_completions(&self) -> &VecDeque<Completion> {
        &self.completion_queue
    }

    /// Drain all completions
    pub fn drain_completions(&mut self) -> Vec<Completion> {
        self.completion_queue.drain(..).collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &RingStats {
        &self.stats
    }

    /// Get queue sizes
    pub fn queue_sizes(&self) -> (usize, usize, usize) {
        (
            self.submission_queue.len(),
            self.processing.len(),
            self.completion_queue.len(),
        )
    }
}

// Public API

/// Submit operation to kernel ring
pub fn submit(op_type: OpType, priority: OpPriority, data: u64, user_data: u64) -> OpId {
    AGENT_RING.lock().submit(op_type, priority, data, user_data)
}

/// Submit batch
pub fn submit_batch(ops: Vec<(OpType, OpPriority, u64, u64)>) -> Vec<OpId> {
    AGENT_RING.lock().submit_batch(ops)
}

/// Wait for completion
pub fn wait(op_id: OpId) -> Option<Completion> {
    AGENT_RING.lock().wait(op_id)
}

/// Process ring (called periodically)
pub fn process() {
    AGENT_RING.lock().process();
}
