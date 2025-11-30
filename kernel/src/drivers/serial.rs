//! Serial Port Driver (UART 16550)
//!
//! Provides debug output via COM1 serial port

use spin::Mutex;
use uart_16550::SerialPort;
use lazy_static::lazy_static;
use core::fmt;

lazy_static! {
    /// Global serial port instance
    pub static ref SERIAL1: Mutex<SerialPort> = {
        let mut serial_port = unsafe { SerialPort::new(0x3F8) };
        serial_port.init();
        Mutex::new(serial_port)
    };
}

/// Initialize serial port
pub fn init() {
    // Serial port is initialized lazily
}

/// Write to serial port
#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    use x86_64::instructions::interrupts;

    interrupts::without_interrupts(|| {
        SERIAL1.lock().write_fmt(args).expect("Serial write failed");
    });
}

/// Print to serial console
#[macro_export]
macro_rules! kprint {
    ($($arg:tt)*) => ($crate::drivers::serial::_print(format_args!($($arg)*)));
}

/// Print line to serial console
#[macro_export]
macro_rules! kprintln {
    () => ($crate::kprint!("\n"));
    ($($arg:tt)*) => ($crate::kprint!("{}\n", format_args!($($arg)*)));
}
