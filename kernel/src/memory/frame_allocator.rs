//! Physical Frame Allocator
//!
//! Manages physical memory frames for BBX OS

use bootloader_api::info::{MemoryRegionKind, MemoryRegions};
use x86_64::structures::paging::{FrameAllocator, PhysFrame, Size4KiB};
use x86_64::PhysAddr;
use spin::Mutex;

/// Global frame allocator
static FRAME_ALLOCATOR: Mutex<Option<BootInfoFrameAllocator>> = Mutex::new(None);

/// Frame allocator that uses memory map from bootloader
pub struct BootInfoFrameAllocator {
    memory_regions: &'static MemoryRegions,
    next: usize,
}

impl BootInfoFrameAllocator {
    /// Create a new frame allocator from memory regions
    pub unsafe fn init(memory_regions: &'static MemoryRegions) -> Self {
        BootInfoFrameAllocator {
            memory_regions,
            next: 0,
        }
    }

    /// Get usable frames iterator
    fn usable_frames(&self) -> impl Iterator<Item = PhysFrame> + '_ {
        // Get usable regions
        let usable_regions = self.memory_regions
            .iter()
            .filter(|r| r.kind == MemoryRegionKind::Usable);

        // Map to address ranges
        let addr_ranges = usable_regions.map(|r| r.start..r.end);

        // Transform to 4KiB aligned frame addresses
        let frame_addresses = addr_ranges.flat_map(|r| r.step_by(4096));

        // Create PhysFrame for each address
        frame_addresses.map(|addr| PhysFrame::containing_address(PhysAddr::new(addr)))
    }
}

unsafe impl FrameAllocator<Size4KiB> for BootInfoFrameAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame> {
        let frame = self.usable_frames().nth(self.next);
        self.next += 1;
        frame
    }
}

/// Initialize frame allocator
pub fn init(memory_regions: &'static MemoryRegions) {
    let allocator = unsafe { BootInfoFrameAllocator::init(memory_regions) };
    *FRAME_ALLOCATOR.lock() = Some(allocator);
}

/// Allocate a physical frame
pub fn allocate_frame() -> Option<PhysFrame> {
    FRAME_ALLOCATOR.lock().as_mut()?.allocate_frame()
}

/// Get number of available frames
pub fn available_frames() -> usize {
    if let Some(ref allocator) = *FRAME_ALLOCATOR.lock() {
        allocator.usable_frames().count() - allocator.next
    } else {
        0
    }
}
