use ash::vk;

use crate::{ImageAttachment, ValueId};

#[derive(Debug, Clone)]
pub enum Value {
    None,
    U32(u32),
    Extent3D(vk::Extent3D),
    ImageAttachment(ImageAttachment),
    Slice(Vec<ValueId>),
}

pub trait FromValue {
    fn from_value(value: &Value) -> Self;
}

impl FromValue for u32 {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::U32(v) => *v,
            _ => panic!("expected U32, got {:?}", value),
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

impl FromValue for Vec<ValueId> {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Slice(v) => v.clone(),
            _ => panic!("expected Slice, got {:?}", value),
        }
    }
}
