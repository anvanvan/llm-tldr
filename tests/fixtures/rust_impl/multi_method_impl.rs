pub struct Counter {
    count: u32,
}

impl Counter {
    pub fn first(&self) -> u32 {
        self.count
    }

    pub fn second(&mut self) {
        self.count += 1;
    }
}
