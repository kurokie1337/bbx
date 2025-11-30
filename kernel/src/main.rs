//! BBX OS Kernel - Operating System for AI Agents
//!
//! Copyright 2025 Ilya Makarov
//! Licensed under BSL-1.1

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]
#![feature(alloc_error_handler)]
#![feature(const_mut_refs)]

extern crate alloc;

mod boot;
mod cpu;
mod memory;
mod interrupts;
mod drivers;
mod scheduler;
mod syscall;

use bootloader_api::{entry_point, BootInfo, BootloaderConfig};
use core::panic::PanicInfo;

/// Bootloader configuration
pub static BOOTLOADER_CONFIG: BootloaderConfig = {
    let mut config = BootloaderConfig::new_default();
    config.mappings.physical_memory = Some(bootloader_api::config::Mapping::Dynamic);
    config.kernel_stack_size = 1024 * 1024; // 1MB kernel stack
    config
};

entry_point!(kernel_main, config = &BOOTLOADER_CONFIG);

/// Kernel entry point - called by bootloader
fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    // Phase 1: Initialize serial console for debugging
    drivers::serial::init();
    kprintln!("BBX OS Kernel v0.1.0");
    kprintln!("Copyright 2025 Ilya Makarov");
    kprintln!("========================================");
    kprintln!("[BOOT] Kernel entry point reached");

    // Phase 2: Initialize CPU features
    kprintln!("[CPU] Initializing CPU...");
    cpu::init();
    kprintln!("[CPU] CPU initialized");

    // Phase 3: Initialize memory management
    kprintln!("[MEM] Initializing memory manager...");
    let phys_mem_offset = boot_info.physical_memory_offset.into_option()
        .expect("Physical memory offset required");
    let memory_regions = &boot_info.memory_regions;
    memory::init(phys_mem_offset, memory_regions);
    kprintln!("[MEM] Memory manager initialized");

    // Phase 4: Initialize heap allocator
    kprintln!("[HEAP] Initializing heap allocator...");
    memory::heap::init_heap();
    kprintln!("[HEAP] Heap allocator initialized");

    // Phase 5: Initialize interrupt handlers
    kprintln!("[INT] Initializing interrupt handlers...");
    interrupts::init();
    kprintln!("[INT] Interrupt handlers initialized");

    // Phase 6: Initialize scheduler
    kprintln!("[SCHED] Initializing scheduler...");
    scheduler::init();
    kprintln!("[SCHED] Scheduler initialized");

    // Phase 7: Initialize system call interface
    kprintln!("[SYS] Initializing syscall interface...");
    syscall::init();
    kprintln!("[SYS] Syscall interface initialized");

    kprintln!("========================================");
    kprintln!("[BBX] Kernel initialization complete!");
    kprintln!("[BBX] Starting Agent Runtime...");
    kprintln!("");

    // Print system info
    print_system_info();

    // Start the main kernel loop
    kernel_loop()
}

/// Print system information
fn print_system_info() {
    kprintln!("=== BBX OS System Information ===");
    kprintln!("Architecture: x86_64");
    kprintln!("Kernel: BBX Kernel v0.1.0");

    let total_mem = memory::get_total_memory();
    let free_mem = memory::get_free_memory();
    kprintln!("Total Memory: {} MB", total_mem / 1024 / 1024);
    kprintln!("Free Memory:  {} MB", free_mem / 1024 / 1024);

    kprintln!("=================================");
    kprintln!("");
}

/// Main kernel loop - runs forever
fn kernel_loop() -> ! {
    kprintln!("[BBX] Entering kernel loop...");
    kprintln!("[BBX] Ready for agent operations");
    kprintln!("");
    kprintln!("BBX> ");

    loop {
        // Process pending interrupts
        x86_64::instructions::interrupts::enable_and_hlt();

        // Run scheduler tick
        scheduler::tick();
    }
}

/// Panic handler - called on kernel panic
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    kprintln!("");
    kprintln!("!!! KERNEL PANIC !!!");
    kprintln!("{}", info);
    kprintln!("");
    kprintln!("System halted.");

    loop {
        x86_64::instructions::hlt();
    }
}

/// Alloc error handler
#[alloc_error_handler]
fn alloc_error_handler(layout: alloc::alloc::Layout) -> ! {
    panic!("Allocation error: {:?}", layout)
}
