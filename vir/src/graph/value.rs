use std::fmt;

use ash::vk;

use crate::{Access, Buffer, ClearValue, ImageAttachment};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "%{}", self.0) }
}
impl fmt::Debug for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}

#[derive(Debug, Clone)]
pub enum Value {
    None,
    I32(i32),
    U32(u32),
    Extent2D(vk::Extent2D),
    Extent3D(vk::Extent3D),
    ImageAttachment(ImageAttachment),
    Buffer(Buffer),
    Slice(Vec<ValueId>),
    Reference(ValueId),
    Access(Access),
    ClearValue(ClearValue),
}

pub trait FromValue {
    fn from_value(value: &Value) -> Self;
}

impl FromValue for i32 {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::I32(v) => *v,
            _ => panic!("expected I32, got {:?}", value),
        }
    }
}

impl FromValue for u32 {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::U32(v) => *v,
            _ => panic!("expected U32, got {:?}", value),
        }
    }
}

impl FromValue for vk::Extent2D {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Extent2D(v) => *v,
            _ => panic!("expected Extent2D, got {:?}", value),
        }
    }
}

impl FromValue for vk::Extent3D {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Extent3D(v) => *v,
            _ => panic!("expected Extent3D, got {:?}", value),
        }
    }
}

impl FromValue for ImageAttachment {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::ImageAttachment(v) => v.clone(),
            _ => panic!("expected ImageAttachment, got {:?}", value),
        }
    }
}

impl FromValue for Buffer {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Buffer(v) => *v,
            _ => panic!("expected Buffer, got {:?}", value),
        }
    }
}

impl FromValue for Vec<ValueId> {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Slice(v) => v.clone(),
            _ => panic!("expected Slice, got {:?}", value),
        }
    }
}

impl FromValue for ValueId {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Reference(v) => *v,
            _ => panic!("expected Reference, got {:?}", value),
        }
    }
}

impl FromValue for Access {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Access(v) => *v,
            _ => panic!("expected Access, got {:?}", value),
        }
    }
}

impl FromValue for ClearValue {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::ClearValue(v) => *v,
            _ => panic!("expected ClearValue, got {:?}", value),
        }
    }
}
