mod foo {
    pub struct Bar {
        x: i32,
    }

    impl Bar {
        pub fn baz(&self) -> i32 {
            self.x
        }
    }
}
