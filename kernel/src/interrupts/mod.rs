//! Interrupt Handling for BBX OS
//!
//! IDT (Interrupt Descriptor Table) and interrupt handlers

use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};
use lazy_static::lazy_static;
use pic8259::ChainedPics;
use spin::Mutex;

pub mod handlers;

/// PIC offsets
pub const PIC_1_OFFSET: u8 = 32;
pub const PIC_2_OFFSET: u8 = PIC_1_OFFSET + 8;

/// Hardware interrupt numbers
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum InterruptIndex {
    Timer = PIC_1_OFFSET,
    Keyboard = PIC_1_OFFSET + 1,
}

impl InterruptIndex {
    fn as_u8(self) -> u8 {
        self as u8
    }

    fn as_usize(self) -> usize {
        usize::from(self.as_u8())
    }
}

/// Programmable Interrupt Controllers
pub static PICS: Mutex<ChainedPics> = Mutex::new(unsafe {
    ChainedPics::new(PIC_1_OFFSET, PIC_2_OFFSET)
});

lazy_static! {
    /// Interrupt Descriptor Table
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();

        // CPU Exceptions
        idt.breakpoint.set_handler_fn(handlers::breakpoint_handler);
        idt.page_fault.set_handler_fn(handlers::page_fault_handler);
        idt.general_protection_fault.set_handler_fn(handlers::general_protection_fault_handler);
        idt.invalid_opcode.set_handler_fn(handlers::invalid_opcode_handler);
        idt.divide_error.set_handler_fn(handlers::divide_error_handler);
        idt.overflow.set_handler_fn(handlers::overflow_handler);

        // Double fault with separate stack
        unsafe {
            idt.double_fault
                .set_handler_fn(handlers::double_fault_handler)
                .set_stack_index(crate::cpu::gdt::DOUBLE_FAULT_IST_INDEX);
        }

        // Hardware Interrupts
        idt[InterruptIndex::Timer.as_usize()].set_handler_fn(handlers::timer_interrupt_handler);
        idt[InterruptIndex::Keyboard.as_usize()].set_handler_fn(handlers::keyboard_interrupt_handler);

        // System call interrupt (0x80 like Linux)
        idt[0x80].set_handler_fn(handlers::syscall_handler);

        idt
    };
}

/// Initialize interrupt handling
pub fn init() {
    // Load IDT
    IDT.load();

    // Initialize PIT timer
    crate::drivers::timer::init();

    // Initialize PICs
    unsafe {
        PICS.lock().initialize();
    }

    // Enable interrupts
    x86_64::instructions::interrupts::enable();
}

/// Notify end of interrupt
pub fn end_of_interrupt(interrupt_id: InterruptIndex) {
    unsafe {
        PICS.lock().notify_end_of_interrupt(interrupt_id.as_u8());
    }
}
