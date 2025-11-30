//! Page Table Management
//!
//! Virtual memory management for BBX OS

use x86_64::structures::paging::{
    OffsetPageTable, PageTable, PhysFrame, Mapper, Page, FrameAllocator,
    Size4KiB, PageTableFlags,
};
use x86_64::{PhysAddr, VirtAddr};
use spin::Mutex;

/// Global page table mapper
static PAGE_TABLE: Mutex<Option<OffsetPageTable<'static>>> = Mutex::new(None);

/// Physical memory offset
static mut PHYS_MEM_OFFSET: u64 = 0;

/// Initialize paging
pub fn init(level_4_table: &'static mut PageTable, phys_mem_offset: u64) {
    unsafe {
        PHYS_MEM_OFFSET = phys_mem_offset;
        let mapper = OffsetPageTable::new(level_4_table, VirtAddr::new(phys_mem_offset));
        *PAGE_TABLE.lock() = Some(mapper);
    }
}

/// Translate virtual address to physical
pub fn translate_addr(addr: VirtAddr) -> Option<PhysAddr> {
    use x86_64::structures::paging::Translate;

    PAGE_TABLE.lock().as_ref()?.translate_addr(addr)
}

/// Map a page to a frame
pub fn map_page(page: Page, frame: PhysFrame, flags: PageTableFlags) -> Result<(), &'static str> {
    let mut allocator = super::frame_allocator::BootInfoFrameAllocator {
        memory_regions: unsafe { &*(0 as *const _) }, // Placeholder
        next: 0,
    };

    if let Some(ref mut mapper) = *PAGE_TABLE.lock() {
        unsafe {
            mapper.map_to(page, frame, flags, &mut allocator)
                .map_err(|_| "Failed to map page")?
                .flush();
        }
        Ok(())
    } else {
        Err("Page table not initialized")
    }
}

/// Create a new page table for a process/agent
pub fn create_page_table() -> Option<PhysFrame> {
    // Allocate a frame for the new page table
    let frame = super::frame_allocator::allocate_frame()?;

    // Zero out the page table
    let phys_addr = frame.start_address();
    let virt_addr = VirtAddr::new(unsafe { PHYS_MEM_OFFSET } + phys_addr.as_u64());

    unsafe {
        let table: *mut PageTable = virt_addr.as_mut_ptr();
        (*table) = PageTable::new();
    }

    Some(frame)
}

/// Clone current page table for fork-like operation
pub fn clone_page_table() -> Option<PhysFrame> {
    let new_table_frame = create_page_table()?;

    // In a real implementation, we would:
    // 1. Copy kernel mappings
    // 2. Set up Copy-on-Write for user mappings
    // This is inspired by BBX StateSnapshots

    Some(new_table_frame)
}
