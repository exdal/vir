use ash::vk;

use crate::{ImageAttachment, ValueId};

pub enum Value {
    ImageAttachment(ImageAttachment),
}

pub trait FromValue {
    fn from_value(value: &Value) -> Self;
}
