//! Kernel Heap Allocator
//!
//! Provides dynamic memory allocation for BBX kernel

use linked_list_allocator::LockedHeap;
use x86_64::structures::paging::{
    mapper::MapToError, FrameAllocator, Mapper, Page, PageTableFlags, Size4KiB,
};
use x86_64::VirtAddr;

/// Heap start address
pub const HEAP_START: usize = 0x_4444_4444_0000;

/// Heap size (16 MB)
pub const HEAP_SIZE: usize = 16 * 1024 * 1024;

/// Global heap allocator
#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

/// Initialize heap allocator
pub fn init_heap() {
    // Map heap pages
    let heap_start = VirtAddr::new(HEAP_START as u64);
    let heap_end = heap_start + HEAP_SIZE - 1u64;

    let heap_start_page = Page::containing_address(heap_start);
    let heap_end_page = Page::containing_address(heap_end);

    // Map all heap pages
    for page in Page::range_inclusive(heap_start_page, heap_end_page) {
        if let Some(frame) = super::frame_allocator::allocate_frame() {
            let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;

            // In a real implementation, we would map through the page table
            // For now, we rely on bootloader's identity mapping

            unsafe {
                // Zero the frame
                let addr = frame.start_address().as_u64() + super::phys_mem_offset();
                core::ptr::write_bytes(addr as *mut u8, 0, 4096);
            }
        }
    }

    // Initialize the allocator
    unsafe {
        ALLOCATOR.lock().init(HEAP_START as *mut u8, HEAP_SIZE);
    }

    crate::kprintln!("[HEAP] Heap initialized: {} MB at {:#x}",
        HEAP_SIZE / 1024 / 1024,
        HEAP_START
    );
}

/// Get heap usage statistics
pub fn heap_stats() -> (usize, usize) {
    let allocator = ALLOCATOR.lock();
    let used = allocator.used();
    let free = allocator.free();
    (used, free)
}
