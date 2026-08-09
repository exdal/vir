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
    pub resource: Option<ValueId>,
}

#[derive(Default, Clone)]
pub struct Program {
    instructions: Vec<Instr>,
    variables: Vec<Variable>,
    slots: HashMap<ValueId, u32>,
    labels: HashMap<LabelId, usize>,
    bound: HashMap<ValueId, u32>,
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

        let bound = variables
            .iter()
            .enumerate()
            .filter_map(|(slot, variable)| Some((variable.resource?, slot as u32)))
            .filter(|(resource, _)| instructions.iter().any(|(id, _)| id == resource))
            .collect();

        Self {
            instructions,
            variables,
            slots,
            labels,
            bound,
        }
    }

    pub(crate) fn bound(&self, resource: &ValueId) -> Option<&Variable> {
        self.variables.get(*self.bound.get(resource)? as usize)
    }

    pub(crate) fn bound_variables(&self) -> impl Iterator<Item = (ValueId, &Variable)> + '_ {
        self.bound
            .iter()
            .filter_map(|(resource, slot)| Some((*resource, self.variables.get(*slot as usize)?)))
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

        let declared = &self.variables[slot as usize];
        match value.kind() {
            Some(kind) if kind == declared.kind => {},
            kind => panic!("variable {variable} holds {:?} but was given {kind:?}", declared.kind),
        }

        if let Some(resource) = declared.resource {
            self.check_bound(resource, &value);
        }

        self.variables[slot as usize].value = value;
    }

    fn check_bound(&self, resource: ValueId, value: &Value) {
        let Value::ImageAttachment(attachment) = value else {
            return;
        };
        let Some((
            _,
            IR::ConstructImage {
                format,
                samples,
                initial_layout,
                ..
            },
        )) = self.instructions.iter().find(|(id, _)| *id == resource)
        else {
            return;
        };

        assert!(
            attachment.format() == *format && attachment.samples() == *samples,
            "{resource} was declared {format:?} samples={samples:?} but was given {:?} samples={:?}",
            attachment.format(),
            attachment.samples(),
        );

        if attachment.layout() != *initial_layout {
            tracing::warn!(
                %resource,
                declared = ?initial_layout,
                given = ?attachment.layout(),
                "bound image does not rest at the layout it was declared with"
            );
        }
    }

    pub fn set_bytes<T: Copy>(&mut self, variable: ValueId, value: &T) {
        let bytes = unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) };
        self.set(variable, bytes);
    }

    pub fn dump(&self) -> String { dump::dump(&self.instructions, self.bound.keys().copied()) }
}
