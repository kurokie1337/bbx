//! Programmable Interval Timer (PIT) Driver
//!
//! Provides system timer for scheduling

use spin::Mutex;
use x86_64::instructions::port::Port;

/// PIT frequency (1.193182 MHz)
const PIT_FREQUENCY: u32 = 1193182;

/// Desired tick frequency (100 Hz = 10ms per tick)
const TICK_FREQUENCY: u32 = 100;

/// PIT divisor
const PIT_DIVISOR: u16 = (PIT_FREQUENCY / TICK_FREQUENCY) as u16;

/// System tick counter
pub static TICKS: Mutex<u64> = Mutex::new(0);

/// PIT ports
const PIT_CHANNEL0: u16 = 0x40;
const PIT_COMMAND: u16 = 0x43;

/// Initialize PIT
pub fn init() {
    let mut command_port = Port::new(PIT_COMMAND);
    let mut data_port = Port::new(PIT_CHANNEL0);

    unsafe {
        // Channel 0, lobyte/hibyte, square wave generator
        command_port.write(0x36u8);

        // Set divisor (low byte then high byte)
        data_port.write((PIT_DIVISOR & 0xFF) as u8);
        data_port.write((PIT_DIVISOR >> 8) as u8);
    }
}

/// Handle timer interrupt
pub fn handle_timer_interrupt() {
    let mut ticks = TICKS.lock();
    *ticks += 1;
}

/// Get current tick count
pub fn get_ticks() -> u64 {
    *TICKS.lock()
}

/// Get uptime in milliseconds
pub fn get_uptime_ms() -> u64 {
    get_ticks() * (1000 / TICK_FREQUENCY as u64)
}

/// Sleep for specified milliseconds (busy wait)
pub fn sleep_ms(ms: u64) {
    let target = get_ticks() + (ms * TICK_FREQUENCY as u64 / 1000);
    while get_ticks() < target {
        x86_64::instructions::hlt();
    }
}
