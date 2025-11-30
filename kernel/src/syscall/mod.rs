//! BBX System Calls
//!
//! Syscall interface for agents running on BBX OS
//! Inspired by BBX MCP tools but at kernel level

use alloc::string::String;
use alloc::vec::Vec;

/// System call numbers
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum SyscallNum {
    // === Process/Task Management (like BBX execution_manager) ===
    /// Spawn new task
    Spawn = 0,
    /// Exit current task
    Exit = 1,
    /// Get current task ID
    GetPid = 2,
    /// Wait for task completion
    Wait = 3,
    /// Kill task
    Kill = 4,
    /// Get task info (like `bbx ps`)
    GetTaskInfo = 5,

    // === Memory Management (like BBX context_tiering) ===
    /// Allocate memory
    Mmap = 10,
    /// Free memory
    Munmap = 11,
    /// Get memory info
    MemInfo = 12,

    // === I/O Operations (like BBX AgentRing) ===
    /// Submit I/O operation to ring
    IoSubmit = 20,
    /// Wait for I/O completion
    IoWait = 21,
    /// Submit batch of I/O operations
    IoSubmitBatch = 22,
    /// Cancel I/O operation
    IoCancel = 23,

    // === State/Storage (like BBX state adapter) ===
    /// Get state value
    StateGet = 30,
    /// Set state value
    StateSet = 31,
    /// Delete state value
    StateDel = 32,
    /// List state keys
    StateList = 33,

    // === Inter-Agent Communication (like BBX A2A) ===
    /// Send message to agent
    AgentSend = 40,
    /// Receive message
    AgentRecv = 41,
    /// Call agent skill
    AgentCall = 42,
    /// Discover agents
    AgentDiscover = 43,

    // === Workflow Execution (like BBX runtime) ===
    /// Run workflow
    WorkflowRun = 50,
    /// Get workflow status
    WorkflowStatus = 51,
    /// Cancel workflow
    WorkflowCancel = 52,

    // === System Info ===
    /// Get system time
    Time = 60,
    /// Get uptime
    Uptime = 61,
    /// Get system info
    SysInfo = 62,

    // === Console I/O ===
    /// Write to console
    Write = 70,
    /// Read from console
    Read = 71,

    // === Misc ===
    /// Yield CPU
    Yield = 100,
    /// Sleep
    Sleep = 101,
}

/// Syscall result
pub type SyscallResult = i64;

/// Error codes
pub const E_OK: i64 = 0;
pub const E_INVAL: i64 = -1;      // Invalid argument
pub const E_NOMEM: i64 = -2;      // Out of memory
pub const E_NOTFOUND: i64 = -3;   // Not found
pub const E_PERM: i64 = -4;       // Permission denied
pub const E_BUSY: i64 = -5;       // Resource busy
pub const E_TIMEOUT: i64 = -6;    // Operation timed out
pub const E_IO: i64 = -7;         // I/O error
pub const E_NOSYS: i64 = -38;     // Function not implemented

/// Initialize syscall interface
pub fn init() {
    // Syscall handler is set up in interrupts module (int 0x80)
}

/// Dispatch syscall (called from interrupt handler)
pub fn dispatch(num: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64) -> SyscallResult {
    match num {
        // Process/Task
        0 => sys_spawn(arg1 as *const u8, arg2 as u8),
        1 => sys_exit(arg1 as i32),
        2 => sys_getpid(),
        3 => sys_wait(arg1),
        4 => sys_kill(arg1),

        // Memory
        10 => sys_mmap(arg1, arg2),
        11 => sys_munmap(arg1, arg2),

        // I/O Ring
        20 => sys_io_submit(arg1 as u8, arg2 as u8, arg3, arg4),
        21 => sys_io_wait(arg1),

        // State
        30 => sys_state_get(arg1 as *const u8, arg2 as *mut u8, arg3),
        31 => sys_state_set(arg1 as *const u8, arg2 as *const u8, arg3),

        // Agent Communication
        40 => sys_agent_send(arg1, arg2 as *const u8, arg3),
        42 => sys_agent_call(arg1 as *const u8, arg2 as *const u8, arg3),

        // Workflow
        50 => sys_workflow_run(arg1 as *const u8),
        51 => sys_workflow_status(arg1),

        // System
        60 => sys_time(),
        61 => sys_uptime(),
        70 => sys_write(arg1 as *const u8, arg2),
        100 => sys_yield(),
        101 => sys_sleep(arg1),

        _ => E_NOSYS,
    }
}

//=============================================================================
// Syscall Implementations
//=============================================================================

/// Spawn new task
fn sys_spawn(name_ptr: *const u8, priority: u8) -> SyscallResult {
    use crate::scheduler::{spawn, task::TaskPriority};

    // In real implementation, would read name from user space
    let priority = match priority {
        0 => TaskPriority::Low,
        1 => TaskPriority::Normal,
        2 => TaskPriority::High,
        3 => TaskPriority::Realtime,
        _ => return E_INVAL,
    };

    let id = spawn("user_task", priority);
    id.0 as i64
}

/// Exit current task
fn sys_exit(code: i32) -> SyscallResult {
    // In real implementation, would terminate current task
    crate::kprintln!("[SYS] Task exit with code {}", code);
    E_OK
}

/// Get current task ID
fn sys_getpid() -> SyscallResult {
    match crate::scheduler::current() {
        Some(id) => id.0 as i64,
        None => E_NOTFOUND,
    }
}

/// Wait for task
fn sys_wait(task_id: u64) -> SyscallResult {
    // In real implementation, would block until task completes
    E_OK
}

/// Kill task
fn sys_kill(task_id: u64) -> SyscallResult {
    use crate::scheduler::{kill, task::TaskId};
    kill(TaskId(task_id));
    E_OK
}

/// Memory map
fn sys_mmap(addr: u64, len: u64) -> SyscallResult {
    // In real implementation, would allocate virtual memory
    E_NOSYS
}

/// Memory unmap
fn sys_munmap(addr: u64, len: u64) -> SyscallResult {
    E_NOSYS
}

/// Submit I/O operation to ring
fn sys_io_submit(op_type: u8, priority: u8, data: u64, user_data: u64) -> SyscallResult {
    use crate::scheduler::ring::{submit, OpType, OpPriority};

    let op_type = match op_type {
        0 => OpType::Read,
        1 => OpType::Write,
        2 => OpType::Network,
        3 => OpType::Timer,
        4 => OpType::Memory,
        5 => OpType::AgentCall,
        _ => OpType::Custom(op_type as u32),
    };

    let priority = match priority {
        0 => OpPriority::Low,
        1 => OpPriority::Normal,
        2 => OpPriority::High,
        3 => OpPriority::Realtime,
        _ => OpPriority::Normal,
    };

    let op_id = submit(op_type, priority, data, user_data);
    op_id.0 as i64
}

/// Wait for I/O completion
fn sys_io_wait(op_id: u64) -> SyscallResult {
    use crate::scheduler::ring::{wait, OpId};

    match wait(OpId(op_id)) {
        Some(completion) => completion.result,
        None => E_NOTFOUND,
    }
}

/// Get state value
fn sys_state_get(key_ptr: *const u8, value_ptr: *mut u8, max_len: u64) -> SyscallResult {
    // In real implementation, would read from state store
    E_NOSYS
}

/// Set state value
fn sys_state_set(key_ptr: *const u8, value_ptr: *const u8, len: u64) -> SyscallResult {
    E_NOSYS
}

/// Send message to agent
fn sys_agent_send(agent_id: u64, msg_ptr: *const u8, len: u64) -> SyscallResult {
    E_NOSYS
}

/// Call agent skill
fn sys_agent_call(agent_ptr: *const u8, skill_ptr: *const u8, input: u64) -> SyscallResult {
    E_NOSYS
}

/// Run workflow
fn sys_workflow_run(path_ptr: *const u8) -> SyscallResult {
    E_NOSYS
}

/// Get workflow status
fn sys_workflow_status(workflow_id: u64) -> SyscallResult {
    E_NOSYS
}

/// Get system time (milliseconds since epoch)
fn sys_time() -> SyscallResult {
    // In real implementation, would read RTC
    0
}

/// Get uptime in milliseconds
fn sys_uptime() -> SyscallResult {
    crate::drivers::timer::get_uptime_ms() as i64
}

/// Write to console
fn sys_write(buf_ptr: *const u8, len: u64) -> SyscallResult {
    // In real implementation, would read buffer from user space
    // For now, just return success
    len as i64
}

/// Yield CPU to other tasks
fn sys_yield() -> SyscallResult {
    x86_64::instructions::hlt();
    E_OK
}

/// Sleep for milliseconds
fn sys_sleep(ms: u64) -> SyscallResult {
    crate::drivers::timer::sleep_ms(ms);
    E_OK
}
