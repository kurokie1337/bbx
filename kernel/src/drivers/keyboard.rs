//! PS/2 Keyboard Driver
//!
//! Handles keyboard input for BBX OS

use pc_keyboard::{layouts, DecodedKey, HandleControl, Keyboard, ScancodeSet1};
use spin::Mutex;
use lazy_static::lazy_static;
use x86_64::instructions::port::Port;

lazy_static! {
    /// Global keyboard instance
    pub static ref KEYBOARD: Mutex<Keyboard<layouts::Us104Key, ScancodeSet1>> = {
        Mutex::new(Keyboard::new(
            ScancodeSet1::new(),
            layouts::Us104Key,
            HandleControl::Ignore,
        ))
    };
}

/// Keyboard data port
const KEYBOARD_DATA_PORT: u16 = 0x60;

/// Read scancode from keyboard
pub fn read_scancode() -> u8 {
    let mut port = Port::new(KEYBOARD_DATA_PORT);
    unsafe { port.read() }
}

/// Process keyboard interrupt
pub fn handle_keyboard_interrupt() {
    let scancode = read_scancode();
    let mut keyboard = KEYBOARD.lock();

    if let Ok(Some(key_event)) = keyboard.add_byte(scancode) {
        if let Some(key) = keyboard.process_keyevent(key_event) {
            match key {
                DecodedKey::Unicode(character) => {
                    crate::kprint!("{}", character);
                }
                DecodedKey::RawKey(key) => {
                    crate::kprint!("{:?}", key);
                }
            }
        }
    }
}

/// Initialize keyboard driver
pub fn init() {
    // Keyboard is initialized via PIC
}
