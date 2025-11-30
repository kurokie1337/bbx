//! Memory Management for BBX OS
//!
//! Physical and virtual memory management inspired by BBX ContextTiering

pub mod heap;
pub mod frame_allocator;
pub mod paging;

use bootloader_api::info::{MemoryRegionKind, MemoryRegions};
use x86_64::structures::paging::{OffsetPageTable, PageTable};
use x86_64::{PhysAddr, VirtAddr};
use spin::Mutex;

/// Physical memory offset (set by bootloader)
static mut PHYSICAL_MEMORY_OFFSET: Option<u64> = None;

/// Total memory in bytes
static TOTAL_MEMORY: Mutex<u64> = Mutex::new(0);

/// Free memory in bytes
static FREE_MEMORY: Mutex<u64> = Mutex::new(0);

/// Initialize memory management
pub fn init(phys_mem_offset: u64, memory_regions: &'static MemoryRegions) {
    unsafe {
        PHYSICAL_MEMORY_OFFSET = Some(phys_mem_offset);
    }

    // Calculate memory stats
    let mut total: u64 = 0;
    let mut usable: u64 = 0;

    for region in memory_regions.iter() {
        let size = region.end - region.start;
        total += size;

        if region.kind == MemoryRegionKind::Usable {
            usable += size;
        }
    }

    *TOTAL_MEMORY.lock() = total;
    *FREE_MEMORY.lock() = usable;

    // Initialize frame allocator
    frame_allocator::init(memory_regions);

    // Initialize page table
    let level_4_table = unsafe { active_level_4_table(VirtAddr::new(phys_mem_offset)) };
    paging::init(level_4_table, phys_mem_offset);

    crate::kprintln!("[MEM] Total: {} MB, Usable: {} MB",
        total / 1024 / 1024,
        usable / 1024 / 1024
    );
}

/// Get physical memory offset
pub fn phys_mem_offset() -> u64 {
    unsafe { PHYSICAL_MEMORY_OFFSET.expect("Memory not initialized") }
}

/// Get total memory
pub fn get_total_memory() -> u64 {
    *TOTAL_MEMORY.lock()
}

/// Get free memory
pub fn get_free_memory() -> u64 {
    *FREE_MEMORY.lock()
}

/// Get active level 4 page table
unsafe fn active_level_4_table(physical_memory_offset: VirtAddr) -> &'static mut PageTable {
    use x86_64::registers::control::Cr3;

    let (level_4_table_frame, _) = Cr3::read();
    let phys = level_4_table_frame.start_address();
    let virt = physical_memory_offset + phys.as_u64();
    let page_table_ptr: *mut PageTable = virt.as_mut_ptr();

    &mut *page_table_ptr
}

/// Translate virtual address to physical
pub fn translate_addr(addr: VirtAddr) -> Option<PhysAddr> {
    paging::translate_addr(addr)
}

//=============================================================================
// BBX CONTEXT TIERING INTEGRATION
//=============================================================================

/// Memory tier (inspired by BBX ContextTiering)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Hot tier - fastest, uncompressed (L1/L2 cache-like)
    Hot,
    /// Warm tier - fast, potentially compressed (RAM)
    Warm,
    /// Cool tier - slower, compressed (swap-like)
    Cool,
    /// Cold tier - slowest, archived (disk)
    Cold,
}

/// Memory region with tiering support
#[derive(Debug)]
pub struct TieredRegion {
    pub tier: MemoryTier,
    pub start: u64,
    pub size: u64,
    pub compressed: bool,
}

/// Promote memory region to higher tier
pub fn promote_region(region: &mut TieredRegion) {
    region.tier = match region.tier {
        MemoryTier::Cold => MemoryTier::Cool,
        MemoryTier::Cool => MemoryTier::Warm,
        MemoryTier::Warm => MemoryTier::Hot,
        MemoryTier::Hot => MemoryTier::Hot,
    };
}

/// Demote memory region to lower tier
pub fn demote_region(region: &mut TieredRegion) {
    region.tier = match region.tier {
        MemoryTier::Hot => MemoryTier::Warm,
        MemoryTier::Warm => MemoryTier::Cool,
        MemoryTier::Cool => MemoryTier::Cold,
        MemoryTier::Cold => MemoryTier::Cold,
    };
}
