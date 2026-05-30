use std::result::Result;

pub struct DaemonState {
    pub value: i32,
}

impl DaemonState {
    pub fn update_all(&mut self) -> Result<(), String> {
        self.value += 1;
        Ok(())
    }
}

pub fn standalone() -> i32 {
    42
}
