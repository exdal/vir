use std::rc::Rc;

use crate::{CommandBuffer, ValueId};

pub type PassCallback = Rc<dyn Fn(&mut CommandBuffer, &[ValueId]) -> Vec<ValueId>>;
