use std::fmt;

pub struct ProtocolMismatch {
    pub expected: String,
}

impl fmt::Display for ProtocolMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mismatch: {}", self.expected)
    }
}
