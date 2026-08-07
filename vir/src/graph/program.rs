use std::collections::HashMap;

use crate::{
    IR,
    LabelId,
    Value,
    ValueId,
    graph::{
        dump,
        ir::{Instr, Name, VariableKind},
    },
};

#[derive(Clone)]
pub struct Variable {
    pub kind: VariableKind,
    pub name: Name,
    pub value: Value,
}

#[derive(Default, Clone)]
pub struct Program {
    instructions: Vec<Instr>,
    variables: Vec<Variable>,
    slots: HashMap<ValueId, u32>,
    labels: HashMap<LabelId, usize>,
}

impl Program {
    pub(crate) fn new(instructions: Vec<Instr>, variables: Vec<Variable>, slots: HashMap<ValueId, u32>) -> Self {
        let labels = instructions
            .iter()
            .enumerate()
            .filter_map(|(index, (_, ir))| match ir {
                IR::Label { label, .. } => Some((*label, index)),
                _ => None,
            })
            .collect();

        Self {
            instructions,
            variables,
            slots,
            labels,
        }
    }

    pub fn instructions(&self) -> &[Instr] { &self.instructions }

    pub(crate) fn label_index(&self, label: LabelId) -> Option<usize> { self.labels.get(&label).copied() }

    pub fn variables(&self) -> &[Variable] { &self.variables }

    pub(crate) fn variable(&self, slot: u32) -> Option<&Variable> { self.variables.get(slot as usize) }

    pub fn set(&mut self, variable: ValueId, value: impl Into<Value>) {
        let value = value.into();
        let Some(slot) = self.slots.get(&variable).copied() else {
            panic!("{variable} is not a variable of this program");
        };

        let declared = &mut self.variables[slot as usize];
        match value.kind() {
            Some(kind) if kind == declared.kind => declared.value = value,
            kind => panic!("variable {variable} holds {:?} but was given {kind:?}", declared.kind),
        }
    }

    pub fn set_bytes<T: Copy>(&mut self, variable: ValueId, value: &T) {
        let bytes = unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.set(variable, bytes);
    }

    pub fn dump(&self) -> String { dump::dump(&self.instructions) }
}
