//! CPU Module for BBX OS
//!
//! CPU initialization and management

pub mod gdt;

use x86_64::registers::control::{Cr0, Cr0Flags};
use x86_64::registers::model_specific::{Efer, EferFlags};

/// Initialize CPU
pub fn init() {
    // Initialize GDT (Global Descriptor Table)
    gdt::init();

    // Enable CPU features
    enable_cpu_features();
}

/// Enable required CPU features
fn enable_cpu_features() {
    // Enable NX (No-Execute) bit if available
    unsafe {
        let efer = Efer::read();
        Efer::write(efer | EferFlags::NO_EXECUTE_ENABLE);
    }

    // Enable write protection
    unsafe {
        let cr0 = Cr0::read();
        Cr0::write(cr0 | Cr0Flags::WRITE_PROTECT);
    }
}

/// Get CPU vendor string
pub fn get_vendor() -> &'static str {
    // Simplified - in real implementation would use CPUID
    "GenuineIntel"
}

/// Check if CPU supports required features
pub fn check_features() -> bool {
    // Check for essential features
    // In real implementation would check CPUID
    true
}

/// Halt CPU until next interrupt
pub fn halt() {
    x86_64::instructions::hlt();
}

/// Disable interrupts
pub fn disable_interrupts() {
    x86_64::instructions::interrupts::disable();
}

/// Enable interrupts
pub fn enable_interrupts() {
    x86_64::instructions::interrupts::enable();
}
