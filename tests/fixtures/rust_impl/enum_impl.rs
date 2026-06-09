pub enum Color {
    Red,
    Green,
    Blue,
}

impl Color {
    pub fn rgb(&self) -> (u8, u8, u8) {
        match self {
            Color::Red => (255, 0, 0),
            Color::Green => (0, 255, 0),
            Color::Blue => (0, 0, 255),
        }
    }
}
