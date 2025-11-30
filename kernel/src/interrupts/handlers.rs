//! Interrupt Handlers for BBX OS
//!
//! Individual handlers for CPU exceptions and hardware interrupts

use x86_64::structures::idt::{InterruptStackFrame, PageFaultErrorCode};
use crate::interrupts::{InterruptIndex, end_of_interrupt};
use crate::{kprintln, kprint};

/// Breakpoint exception handler
pub extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    kprintln!("[INT] BREAKPOINT");
    kprintln!("{:#?}", stack_frame);
}

/// Page fault handler
pub extern "x86-interrupt" fn page_fault_handler(
    stack_frame: InterruptStackFrame,
    error_code: PageFaultErrorCode,
) {
    use x86_64::registers::control::Cr2;

    kprintln!("[INT] PAGE FAULT");
    kprintln!("Accessed Address: {:?}", Cr2::read());
    kprintln!("Error Code: {:?}", error_code);
    kprintln!("{:#?}", stack_frame);

    // In a real OS, we would handle page faults (demand paging)
    // For now, halt
    loop {
        x86_64::instructions::hlt();
    }
}

/// Double fault handler (unrecoverable)
pub extern "x86-interrupt" fn double_fault_handler(
    stack_frame: InterruptStackFrame,
    _error_code: u64,
) -> ! {
    kprintln!("[INT] DOUBLE FAULT");
    kprintln!("{:#?}", stack_frame);
    panic!("Double fault - system halted");
}

/// General protection fault handler
pub extern "x86-interrupt" fn general_protection_fault_handler(
    stack_frame: InterruptStackFrame,
    error_code: u64,
) {
    kprintln!("[INT] GENERAL PROTECTION FAULT");
    kprintln!("Error Code: {}", error_code);
    kprintln!("{:#?}", stack_frame);

    loop {
        x86_64::instructions::hlt();
    }
}

/// Invalid opcode handler
pub extern "x86-interrupt" fn invalid_opcode_handler(stack_frame: InterruptStackFrame) {
    kprintln!("[INT] INVALID OPCODE");
    kprintln!("{:#?}", stack_frame);

    loop {
        x86_64::instructions::hlt();
    }
}

/// Divide by zero handler
pub extern "x86-interrupt" fn divide_error_handler(stack_frame: InterruptStackFrame) {
    kprintln!("[INT] DIVIDE BY ZERO");
    kprintln!("{:#?}", stack_frame);

    loop {
        x86_64::instructions::hlt();
    }
}

/// Overflow handler
pub extern "x86-interrupt" fn overflow_handler(stack_frame: InterruptStackFrame) {
    kprintln!("[INT] OVERFLOW");
    kprintln!("{:#?}", stack_frame);
}

/// Timer interrupt handler (IRQ0)
pub extern "x86-interrupt" fn timer_interrupt_handler(_stack_frame: InterruptStackFrame) {
    // Update system tick counter
    crate::drivers::timer::handle_timer_interrupt();

    // Signal end of interrupt
    end_of_interrupt(InterruptIndex::Timer);
}

/// Keyboard interrupt handler (IRQ1)
pub extern "x86-interrupt" fn keyboard_interrupt_handler(_stack_frame: InterruptStackFrame) {
    // Handle keyboard input
    crate::drivers::keyboard::handle_keyboard_interrupt();

    // Signal end of interrupt
    end_of_interrupt(InterruptIndex::Keyboard);
}

/// System call handler (int 0x80)
///
/// This is the main entry point for BBX syscalls from agents
pub extern "x86-interrupt" fn syscall_handler(stack_frame: InterruptStackFrame) {
    // In a real implementation, we would:
    // 1. Read syscall number from register (e.g., rax)
    // 2. Read arguments from registers (rdi, rsi, rdx, etc.)
    // 3. Dispatch to appropriate syscall handler
    // 4. Return result in rax

    kprintln!("[SYS] Syscall received");
    kprintln!("Instruction: {:?}", stack_frame.instruction_pointer);

    // TODO: Implement BBX syscall dispatch
    // crate::syscall::dispatch(syscall_num, args);
}
