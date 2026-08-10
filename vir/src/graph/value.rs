use std::{fmt, sync::Arc};

use ash::vk;

use crate::{Access, Buffer, ClearValue, ImageAttachment, PassCallback, SwapchainBinding, graph::ir::VariableKind};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "%{}", self.0) }
}
impl fmt::Debug for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LabelId(pub u32);

impl fmt::Display for LabelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "label {}", self.0) }
}
impl fmt::Debug for LabelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}

#[derive(Debug, Clone)]
pub enum Value {
    None,
    Bool(bool),
    I32(i32),
    U32(u32),
    Size(usize),
    Extent2D(vk::Extent2D),
    Extent3D(vk::Extent3D),
    ImageAttachment(ImageAttachment),
    Buffer(Buffer),
    Bytes(Arc<[u8]>),
    Swapchain(SwapchainBinding),
    Slice(Vec<ValueId>),
    Reference(ValueId),
    Access(Access),
    ClearValue(ClearValue),
    Callback(PassCallback),
}

impl Value {
    pub fn kind(&self) -> Option<VariableKind> {
        match self {
            Value::Bool(_) => Some(VariableKind::Bool),
            Value::I32(_) => Some(VariableKind::I32),
            Value::U32(_) => Some(VariableKind::U32),
            Value::Extent2D(_) => Some(VariableKind::Extent2D),
            Value::Extent3D(_) => Some(VariableKind::Extent3D),
            Value::ClearValue(_) => Some(VariableKind::ClearValue),
            Value::Bytes(bytes) => Some(VariableKind::Bytes(bytes.len() as u32)),
            Value::Swapchain(_) => Some(VariableKind::Swapchain),
            Value::Buffer(_) => Some(VariableKind::Buffer),
            Value::ImageAttachment(_) => Some(VariableKind::ImageAttachment),
            Value::Callback(_) => Some(VariableKind::Callback),
            _ => None,
        }
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self { Value::Bool(v) }
}
impl From<i32> for Value {
    fn from(v: i32) -> Self { Value::I32(v) }
}
impl From<u32> for Value {
    fn from(v: u32) -> Self { Value::U32(v) }
}
impl From<usize> for Value {
    fn from(v: usize) -> Self { Value::Size(v) }
}
impl From<vk::Extent2D> for Value {
    fn from(v: vk::Extent2D) -> Self { Value::Extent2D(v) }
}
impl From<vk::Extent3D> for Value {
    fn from(v: vk::Extent3D) -> Self { Value::Extent3D(v) }
}
impl From<ClearValue> for Value {
    fn from(v: ClearValue) -> Self { Value::ClearValue(v) }
}
impl From<&[u8]> for Value {
    fn from(v: &[u8]) -> Self { Value::Bytes(Arc::from(v)) }
}
impl From<SwapchainBinding> for Value {
    fn from(v: SwapchainBinding) -> Self { Value::Swapchain(v) }
}
impl From<Buffer> for Value {
    fn from(v: Buffer) -> Self { Value::Buffer(v) }
}
impl From<&Buffer> for Value {
    fn from(v: &Buffer) -> Self { Value::Buffer(*v) }
}
impl From<ImageAttachment> for Value {
    fn from(v: ImageAttachment) -> Self { Value::ImageAttachment(v) }
}
impl From<&ImageAttachment> for Value {
    fn from(v: &ImageAttachment) -> Self { Value::ImageAttachment(v.clone()) }
}
impl From<PassCallback> for Value {
    fn from(v: PassCallback) -> Self { Value::Callback(v) }
}

pub trait FromValue {
    fn from_value(value: &Value) -> Self;
}

impl FromValue for bool {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Bool(v) => *v,
            _ => panic!("expected Bool, got {:?}", value),
        }
    }
}

impl FromValue for SwapchainBinding {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Swapchain(v) => v.clone(),
            _ => panic!("expected Swapchain, got {:?}", value),
        }
    }
}

impl FromValue for Arc<[u8]> {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Bytes(v) => v.clone(),
            _ => panic!("expected Bytes, got {:?}", value),
        }
    }
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

impl FromValue for usize {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Size(v) => *v,
            _ => panic!("expected Size, got {:?}", value),
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

impl FromValue for PassCallback {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Callback(v) => v.clone(),
            _ => panic!("expected Callback, got {:?}", value),
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
