// impl appears BEFORE the struct definition (forward reference)
impl Foo {
    pub fn early(&self) -> i32 {
        42
    }
}

pub struct Foo {
    x: i32,
}
