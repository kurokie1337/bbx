//! BBX OS Boot Module
//!
//! Boot information and early initialization

use bootloader_api::BootInfo;

/// Boot information from bootloader
pub struct BbxBootInfo {
    /// Physical memory offset
    pub phys_mem_offset: u64,

    /// Framebuffer info (if available)
    pub framebuffer: Option<FramebufferInfo>,

    /// RSDP address (for ACPI)
    pub rsdp_addr: Option<u64>,
}

/// Framebuffer information
pub struct FramebufferInfo {
    pub buffer_start: u64,
    pub buffer_size: usize,
    pub width: usize,
    pub height: usize,
    pub pitch: usize,
    pub bytes_per_pixel: usize,
}

impl BbxBootInfo {
    /// Create from bootloader BootInfo
    pub fn from_bootloader(info: &BootInfo) -> Self {
        let framebuffer = info.framebuffer.as_ref().map(|fb| {
            let fb_info = fb.info();
            FramebufferInfo {
                buffer_start: fb.buffer().as_ptr() as u64,
                buffer_size: fb.buffer().len(),
                width: fb_info.width,
                height: fb_info.height,
                pitch: fb_info.stride * fb_info.bytes_per_pixel,
                bytes_per_pixel: fb_info.bytes_per_pixel,
            }
        });

        BbxBootInfo {
            phys_mem_offset: info.physical_memory_offset.into_option().unwrap_or(0),
            framebuffer,
            rsdp_addr: info.rsdp_addr.into_option().map(|a| a.as_u64()),
        }
    }
}

/// Print boot banner
pub fn print_banner() {
    crate::kprintln!(r#"
    ____  ____  _  __   ____  _____
   | __ )| __ )\ \/ /  / __ \/ ___/
   |  _ \|  _ \ \  /  / / / /\__ \
   | |_) | |_) |/  \ / /_/ /___/ /
   |____/|____//_/\_\\____//____/

   Operating System for AI Agents
   Copyright 2025 Ilya Makarov
"#);
}

/// Early boot checks
pub fn early_checks() -> Result<(), &'static str> {
    // Check CPU features
    if !crate::cpu::check_features() {
        return Err("CPU does not support required features");
    }

    Ok(())
}
